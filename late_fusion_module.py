
import numpy as np
import pandas as pd
import torch
import cv2
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
from enum import Enum
import json
import time
from datetime import datetime
import logging

# Import the existing classifiers
import sys
import os
sys.path.append('active')
sys.path.append('passive/model/network')

from active.emotion_detector import EmotionDetector
from passive.model.network.CNN import EmotionCNN


class FusionStrategy(Enum):
    """Available fusion strategies"""
    WEIGHTED_AVERAGE = "weighted_average"
    RULE_BASED = "rule_based"
    CONFIDENCE_BASED = "confidence_based"
    ADAPTIVE_WEIGHTED = "adaptive_weighted"


@dataclass
class BiosignalPrediction:
    """Biosignal prediction structure"""
    valence: int  # 0 or 1 (binary classification)
    arousal: int  # 0 or 1 (binary classification)
    confidence: float 
    timestamp: float
    method: str = "cnn_biosignal"


@dataclass
class VisualPrediction:
    """Visual/facial expression prediction structure"""
    emotion: str  # Discrete emotion label
    valence: float  # Continuous valence (-1 to 1)
    arousal: float  # Continuous arousal (-1 to 1)
    confidence: float  # Prediction confidence (0-1)
    faces_detected: int
    timestamp: float
    method: str = "facial_expression"


@dataclass
class FusedPrediction:
    """Final fused prediction"""
    valence: float  # Final valence (-1 to 1)
    arousal: float  # Final arousal (-1 to 1)
    discrete_emotion: str  # Most likely discrete emotion
    fusion_confidence: float  # Overall confidence
    fusion_strategy: str
    contributing_modalities: List[str]
    timestamp: float
    
    # Individual contributions
    biosignal_contribution: float = 0.0
    visual_contribution: float = 0.0
    
    # Metadata
    metadata: Dict = None


class LateFusionModule:
    """
    Late Fusion Module for combining biosignal and facial expression emotion predictions
    
    Supports multiple fusion strategies:
    1. Weighted Average: Simple linear combination with fixed weights
    2. Rule-based: Decision rules based on confidence and modality reliability
    3. Confidence-based: Dynamic weighting based on prediction confidence
    4. Adaptive Weighted: Learning-based weight adaptation
    """
    
    def __init__(self, 
                 biosignal_model_path: str = "passive/model/network/emotion_cnn.pth",
                 enable_visual: bool = True,
                 fusion_strategy: FusionStrategy = FusionStrategy.WEIGHTED_AVERAGE,
                 weights: Dict[str, float] = None,
                 confidence_threshold: float = 0.5,
                 logger: logging.Logger = None):
        """
        Initialize the Late Fusion Module
        
        Args:
            biosignal_model_path: Path to trained biosignal CNN model
            enable_visual: Whether to enable visual emotion detection
            fusion_strategy: Strategy for combining predictions
            weights: Manual weights for weighted average (biosignal, visual)
            confidence_threshold: Minimum confidence for predictions
            logger: Logger instance
        """
        self.fusion_strategy = fusion_strategy
        self.confidence_threshold = confidence_threshold
        self.logger = logger or self._setup_logger()
        
        # Default weights if not provided
        self.weights = weights or {"biosignal": 0.6, "visual": 0.4}
        
        # Valence-Arousal to emotion mapping (Russell's Circumplex Model)
        self.va_to_emotion_map = self._create_va_emotion_mapping()
        
        # Initialize biosignal classifier (gracefully handle missing model for manual-input scenarios)
        try:
            self.biosignal_model = self._load_biosignal_model(biosignal_model_path)
        except Exception as e:
            self.logger.warning(
                f"Biosignal model unavailable; continuing without it (manual inputs still supported). Reason: {e}"
            )
            self.biosignal_model = None
        
        # Initialize visual classifier
        self.visual_detector = None
        if enable_visual:
            try:
                self.visual_detector = EmotionDetector(enable_valence_arousal=True)
                self.logger.info("Visual emotion detector initialized successfully")
            except Exception as e:
                self.logger.error(f"Failed to initialize visual detector: {e}")
                
        # Fusion history for adaptive weighting
        self.fusion_history = []
        self.performance_metrics = {"biosignal": [], "visual": [], "fused": []}
        
    def _setup_logger(self) -> logging.Logger:
        """Setup logging"""
        logger = logging.getLogger("LateFusionModule")
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            
        return logger
        
    def _load_biosignal_model(self, model_path: str) -> torch.nn.Module:
        """Load the trained biosignal CNN model"""
        try:
            model = EmotionCNN()
            model.load_state_dict(torch.load(model_path, map_location='cpu'))
            model.eval()
            self.logger.info(f"Biosignal model loaded from {model_path}")
            return model
        except Exception as e:
            self.logger.error(f"Failed to load biosignal model: {e}")
            raise
            
    def _create_va_emotion_mapping(self) -> Dict[Tuple[int, int], str]:
        """Create mapping from valence-arousal quadrants to emotions"""
        return {
            (0, 0): "sad",        # Low valence, Low arousal
            (0, 1): "angry",      # Low valence, High arousal  
            (1, 0): "calm",       # High valence, Low arousal
            (1, 1): "happy"       # High valence, High arousal
        }
        
    def predict_biosignal_emotion(self, signal_data: np.ndarray) -> BiosignalPrediction:
        """
        Predict emotion from biosignal data
        
        Args:
            signal_data: PPG/pulse signal segments (shape: [batch_size, 140])
            
        Returns:
            BiosignalPrediction object
        """
        # If the biosignal model isn't loaded, return a neutral low-confidence default
        if self.biosignal_model is None:
            return BiosignalPrediction(
                valence=0, arousal=0, confidence=0.0, timestamp=time.time()
            )

        try:
            # Convert to tensor
            if len(signal_data.shape) == 1:
                signal_data = signal_data.reshape(1, -1)
                
            X_tensor = torch.tensor(signal_data, dtype=torch.float32)
            
            with torch.no_grad():
                valence_logits, arousal_logits = self.biosignal_model(X_tensor)
                
                # Get predictions and confidence
                valence_probs = torch.softmax(valence_logits, dim=1)
                arousal_probs = torch.softmax(arousal_logits, dim=1)
                
                valence_pred = torch.argmax(valence_probs, dim=1).item()
                arousal_pred = torch.argmax(arousal_probs, dim=1).item()
                
                # Calculate confidence as max probability
                valence_conf = torch.max(valence_probs, dim=1)[0].item()
                arousal_conf = torch.max(arousal_probs, dim=1)[0].item()
                overall_conf = (valence_conf + arousal_conf) / 2.0
                
            return BiosignalPrediction(
                valence=valence_pred,
                arousal=arousal_pred,
                confidence=overall_conf,
                timestamp=time.time()
            )
            
        except Exception as e:
            self.logger.error(f"Biosignal prediction failed: {e}")
            # Return default prediction
            return BiosignalPrediction(
                valence=0, arousal=0, confidence=0.0, timestamp=time.time()
            )
    
    def predict_visual_emotion(self, frame: np.ndarray) -> Optional[VisualPrediction]:
        """
        Predict emotion from visual/facial data
        
        Args:
            frame: Video frame/image (BGR format)
            
        Returns:
            VisualPrediction object or None if no faces detected
        """
        if self.visual_detector is None:
            return None
            
        try:
            processed_frame, results = self.visual_detector.detect_emotions(frame)
            
            if not results:
                return None
                
            # Average across all detected faces
            avg_valence = np.mean([r.get("valence", 0) for r in results])
            avg_arousal = np.mean([r.get("arousal", 0) for r in results])
            avg_confidence = np.mean([r.get("confidence", 0) for r in results])
            
            # Get most confident emotion
            best_result = max(results, key=lambda x: x.get("confidence", 0))
            
            return VisualPrediction(
                emotion=best_result["emotion"],
                valence=avg_valence,
                arousal=avg_arousal,
                confidence=avg_confidence,
                faces_detected=len(results),
                timestamp=time.time()
            )
            
        except Exception as e:
            self.logger.error(f"Visual prediction failed: {e}")
            return None
    
    def fuse_predictions(self, 
                        biosignal_pred: Optional[BiosignalPrediction],
                        visual_pred: Optional[VisualPrediction]) -> FusedPrediction:
        """
        Fuse biosignal and visual predictions using the selected strategy
        
        Args:
            biosignal_pred: Biosignal prediction
            visual_pred: Visual prediction
            
        Returns:
            FusedPrediction object
        """
        timestamp = time.time()
        
        # Handle cases where one or both predictions are missing
        if biosignal_pred is None and visual_pred is None:
            return self._create_default_prediction(timestamp)
        
        if biosignal_pred is None:
            return self._visual_only_prediction(visual_pred, timestamp)
            
        if visual_pred is None:
            return self._biosignal_only_prediction(biosignal_pred, timestamp)
        
        # Both predictions available - apply fusion strategy
        if self.fusion_strategy == FusionStrategy.WEIGHTED_AVERAGE:
            return self._weighted_average_fusion(biosignal_pred, visual_pred, timestamp)
        elif self.fusion_strategy == FusionStrategy.RULE_BASED:
            return self._rule_based_fusion(biosignal_pred, visual_pred, timestamp)
        elif self.fusion_strategy == FusionStrategy.CONFIDENCE_BASED:
            return self._confidence_based_fusion(biosignal_pred, visual_pred, timestamp)
        elif self.fusion_strategy == FusionStrategy.ADAPTIVE_WEIGHTED:
            return self._adaptive_weighted_fusion(biosignal_pred, visual_pred, timestamp)
        else:
            # Default to weighted average
            return self._weighted_average_fusion(biosignal_pred, visual_pred, timestamp)
    
    def _weighted_average_fusion(self, 
                                biosignal: BiosignalPrediction,
                                visual: VisualPrediction,
                                timestamp: float) -> FusedPrediction:
        """Weighted average fusion strategy"""
        
        # Convert biosignal binary predictions to continuous scale (-1 to 1)
        bio_valence = (biosignal.valence * 2) - 1  # 0->-1, 1->1
        bio_arousal = (biosignal.arousal * 2) - 1
        
        # Weight by predefined weights
        w_bio = self.weights["biosignal"]
        w_vis = self.weights["visual"]
        
        # Fused valence and arousal
        fused_valence = w_bio * bio_valence + w_vis * visual.valence
        fused_arousal = w_bio * bio_arousal + w_vis * visual.arousal
        
        # Overall confidence
        fusion_confidence = w_bio * biosignal.confidence + w_vis * visual.confidence
        
        # Map to discrete emotion
        discrete_emotion = self._map_va_to_emotion(fused_valence, fused_arousal)
        
        return FusedPrediction(
            valence=fused_valence,
            arousal=fused_arousal,
            discrete_emotion=discrete_emotion,
            fusion_confidence=fusion_confidence,
            fusion_strategy="weighted_average",
            contributing_modalities=["biosignal", "visual"],
            timestamp=timestamp,
            biosignal_contribution=w_bio,
            visual_contribution=w_vis,
            metadata={
                "biosignal": {
                    "valence": biosignal.valence,
                    "arousal": biosignal.arousal,
                    "confidence": biosignal.confidence
                },
                "visual": {
                    "emotion": visual.emotion,
                    "valence": visual.valence,
                    "arousal": visual.arousal,
                    "confidence": visual.confidence,
                    "faces": visual.faces_detected
                }
            }
        )
    
    def _confidence_based_fusion(self,
                               biosignal: BiosignalPrediction,
                               visual: VisualPrediction,
                               timestamp: float) -> FusedPrediction:
        """Confidence-based dynamic weighting fusion"""
        
        # Normalize confidences
        total_conf = biosignal.confidence + visual.confidence
        if total_conf > 0:
            w_bio = biosignal.confidence / total_conf
            w_vis = visual.confidence / total_conf
        else:
            w_bio = w_vis = 0.5
        
        # Convert biosignal predictions
        bio_valence = (biosignal.valence * 2) - 1
        bio_arousal = (biosignal.arousal * 2) - 1
        
        # Fused predictions
        fused_valence = w_bio * bio_valence + w_vis * visual.valence
        fused_arousal = w_bio * bio_arousal + w_vis * visual.arousal
        
        # Confidence as weighted average
        fusion_confidence = w_bio * biosignal.confidence + w_vis * visual.confidence
        
        discrete_emotion = self._map_va_to_emotion(fused_valence, fused_arousal)
        
        return FusedPrediction(
            valence=fused_valence,
            arousal=fused_arousal,
            discrete_emotion=discrete_emotion,
            fusion_confidence=fusion_confidence,
            fusion_strategy="confidence_based",
            contributing_modalities=["biosignal", "visual"],
            timestamp=timestamp,
            biosignal_contribution=w_bio,
            visual_contribution=w_vis,
            metadata={
                "dynamic_weights": {"biosignal": w_bio, "visual": w_vis},
                "original_confidences": {
                    "biosignal": biosignal.confidence,
                    "visual": visual.confidence
                }
            }
        )
    
    def _rule_based_fusion(self,
                          biosignal: BiosignalPrediction,
                          visual: VisualPrediction,
                          timestamp: float) -> FusedPrediction:
        """Rule-based fusion with heuristics"""
        
        # Rule 1: If one prediction has very low confidence, trust the other more
        if biosignal.confidence < 0.3 and visual.confidence > 0.7:
            return self._visual_only_prediction(visual, timestamp, "rule_based_visual_dominant")
            
        if visual.confidence < 0.3 and biosignal.confidence > 0.7:
            return self._biosignal_only_prediction(biosignal, timestamp, "rule_based_biosignal_dominant")
        
        # Rule 2: If both are confident but disagree strongly, use biosignal for arousal and visual for valence
        bio_valence = (biosignal.valence * 2) - 1
        bio_arousal = (biosignal.arousal * 2) - 1
        
        valence_disagreement = abs(bio_valence - visual.valence)
        arousal_disagreement = abs(bio_arousal - visual.arousal)
        
        if (biosignal.confidence > 0.6 and visual.confidence > 0.6 and 
            (valence_disagreement > 1.0 or arousal_disagreement > 1.0)):
            
            # Trust visual for valence (facial expressions are good for positive/negative)
            # Trust biosignal for arousal (physiological signals are good for activation)
            fused_valence = visual.valence
            fused_arousal = bio_arousal
            fusion_strategy = "rule_based_split_decision"
            
        else:
            # Rule 3: Default to confidence-weighted combination
            total_conf = biosignal.confidence + visual.confidence
            w_bio = biosignal.confidence / total_conf if total_conf > 0 else 0.5
            w_vis = visual.confidence / total_conf if total_conf > 0 else 0.5
            
            fused_valence = w_bio * bio_valence + w_vis * visual.valence
            fused_arousal = w_bio * bio_arousal + w_vis * visual.arousal
            fusion_strategy = "rule_based_weighted"
        
        fusion_confidence = max(biosignal.confidence, visual.confidence)
        discrete_emotion = self._map_va_to_emotion(fused_valence, fused_arousal)
        
        return FusedPrediction(
            valence=fused_valence,
            arousal=fused_arousal,
            discrete_emotion=discrete_emotion,
            fusion_confidence=fusion_confidence,
            fusion_strategy=fusion_strategy,
            contributing_modalities=["biosignal", "visual"],
            timestamp=timestamp,
            metadata={
                "rule_applied": fusion_strategy,
                "disagreement_scores": {
                    "valence": valence_disagreement,
                    "arousal": arousal_disagreement
                }
            }
        )
    
    def _adaptive_weighted_fusion(self,
                                biosignal: BiosignalPrediction,
                                visual: VisualPrediction,
                                timestamp: float) -> FusedPrediction:
        """Adaptive weighted fusion that learns from performance history"""
        
        # Start with base weights
        w_bio = self.weights["biosignal"]
        w_vis = self.weights["visual"]
        
        # Adjust based on recent performance if we have history
        if len(self.performance_metrics["biosignal"]) > 5:
            bio_recent_perf = np.mean(self.performance_metrics["biosignal"][-5:])
            vis_recent_perf = np.mean(self.performance_metrics["visual"][-5:])
            
            total_perf = bio_recent_perf + vis_recent_perf
            if total_perf > 0:
                w_bio = bio_recent_perf / total_perf
                w_vis = vis_recent_perf / total_perf
        
        # Apply confidence modulation
        conf_factor = 0.2  # How much confidence affects weights
        w_bio = w_bio * (1 + conf_factor * (biosignal.confidence - 0.5))
        w_vis = w_vis * (1 + conf_factor * (visual.confidence - 0.5))
        
        # Normalize weights
        total_w = w_bio + w_vis
        w_bio /= total_w
        w_vis /= total_w
        
        # Fuse predictions
        bio_valence = (biosignal.valence * 2) - 1
        bio_arousal = (biosignal.arousal * 2) - 1
        
        fused_valence = w_bio * bio_valence + w_vis * visual.valence
        fused_arousal = w_bio * bio_arousal + w_vis * visual.arousal
        
        fusion_confidence = w_bio * biosignal.confidence + w_vis * visual.confidence
        discrete_emotion = self._map_va_to_emotion(fused_valence, fused_arousal)
        
        return FusedPrediction(
            valence=fused_valence,
            arousal=fused_arousal,
            discrete_emotion=discrete_emotion,
            fusion_confidence=fusion_confidence,
            fusion_strategy="adaptive_weighted",
            contributing_modalities=["biosignal", "visual"],
            timestamp=timestamp,
            biosignal_contribution=w_bio,
            visual_contribution=w_vis,
            metadata={
                "adaptive_weights": {"biosignal": w_bio, "visual": w_vis},
                "base_weights": self.weights,
                "performance_history_length": len(self.performance_metrics["biosignal"])
            }
        )
    
    def _visual_only_prediction(self, visual: VisualPrediction, timestamp: float, 
                               strategy: str = "visual_only") -> FusedPrediction:
        """Create prediction from visual data only"""
        discrete_emotion = visual.emotion
        
        return FusedPrediction(
            valence=visual.valence,
            arousal=visual.arousal,
            discrete_emotion=discrete_emotion,
            fusion_confidence=visual.confidence,
            fusion_strategy=strategy,
            contributing_modalities=["visual"],
            timestamp=timestamp,
            visual_contribution=1.0,
            biosignal_contribution=0.0
        )
    
    def _biosignal_only_prediction(self, biosignal: BiosignalPrediction, timestamp: float,
                                  strategy: str = "biosignal_only") -> FusedPrediction:
        """Create prediction from biosignal data only"""
        bio_valence = (biosignal.valence * 2) - 1
        bio_arousal = (biosignal.arousal * 2) - 1
        
        discrete_emotion = self._map_va_to_emotion(bio_valence, bio_arousal)
        
        return FusedPrediction(
            valence=bio_valence,
            arousal=bio_arousal,
            discrete_emotion=discrete_emotion,
            fusion_confidence=biosignal.confidence,
            fusion_strategy=strategy,
            contributing_modalities=["biosignal"],
            timestamp=timestamp,
            biosignal_contribution=1.0,
            visual_contribution=0.0
        )
    
    def _create_default_prediction(self, timestamp: float) -> FusedPrediction:
        """Create default neutral prediction when no data available"""
        return FusedPrediction(
            valence=0.0,
            arousal=0.0,
            discrete_emotion="neutral",
            fusion_confidence=0.0,
            fusion_strategy="default",
            contributing_modalities=[],
            timestamp=timestamp
        )
    
    def _map_va_to_emotion(self, valence: float, arousal: float) -> str:
        """Map continuous valence-arousal to discrete emotion"""
        # Convert to binary for lookup
        v_bin = 1 if valence > 0 else 0
        a_bin = 1 if arousal > 0 else 0
        
        # Use lookup table
        base_emotion = self.va_to_emotion_map.get((v_bin, a_bin), "neutral")
        
        # Refine based on magnitude
        v_mag = abs(valence)
        a_mag = abs(arousal)
        
        if v_mag < 0.2 and a_mag < 0.2:
            return "neutral"
        
        # High arousal emotions
        if a_mag > 0.7:
            if v_mag > 0.5:
                return "happy" if valence > 0 else "angry"
            else:
                return "surprise" if arousal > 0 else "fear"
        
        # Low arousal emotions  
        if a_mag < 0.3:
            if v_mag > 0.3:
                return "content" if valence > 0 else "sad"
            else:
                return "calm"
        
        return base_emotion
    
    def update_performance_metrics(self, prediction: FusedPrediction, ground_truth: Dict = None):
        """Update performance metrics for adaptive weighting"""
        if ground_truth is None:
            return
            
        # Calculate performance scores (placeholder - would need actual metrics)
        # This is a simplified version - in practice you'd use proper evaluation metrics
        if "biosignal" in prediction.contributing_modalities:
            bio_score = 1.0 - abs(prediction.valence - ground_truth.get("valence", 0))
            self.performance_metrics["biosignal"].append(max(0, bio_score))
            
        if "visual" in prediction.contributing_modalities:
            vis_score = 1.0 - abs(prediction.arousal - ground_truth.get("arousal", 0))
            self.performance_metrics["visual"].append(max(0, vis_score))
        
        # Keep only recent history
        max_history = 100
        for key in self.performance_metrics:
            if len(self.performance_metrics[key]) > max_history:
                self.performance_metrics[key] = self.performance_metrics[key][-max_history:]
    
    def save_fusion_results(self, predictions: List[FusedPrediction], filepath: str):
        """Save fusion results to JSON file"""
        results = []
        for pred in predictions:
            result = {
                "timestamp": pred.timestamp,
                "valence": pred.valence,
                "arousal": pred.arousal,
                "discrete_emotion": pred.discrete_emotion,
                "fusion_confidence": pred.fusion_confidence,
                "fusion_strategy": pred.fusion_strategy,
                "contributing_modalities": pred.contributing_modalities,
                "biosignal_contribution": pred.biosignal_contribution,
                "visual_contribution": pred.visual_contribution,
                "metadata": pred.metadata
            }
            results.append(result)
        
        with open(filepath, 'w') as f:
            json.dump(results, f, indent=2)
        
        self.logger.info(f"Saved {len(results)} fusion results to {filepath}")
    
    def get_fusion_statistics(self) -> Dict:
        """Get statistics about fusion performance"""
        if not self.fusion_history:
            return {}
        
        stats = {
            "total_predictions": len(self.fusion_history),
            "fusion_strategies_used": {},
            "average_confidence": np.mean([p.fusion_confidence for p in self.fusion_history]),
            "modality_usage": {"biosignal_only": 0, "visual_only": 0, "both": 0},
            "emotion_distribution": {}
        }
        
        for pred in self.fusion_history:
            # Strategy usage
            strategy = pred.fusion_strategy
            stats["fusion_strategies_used"][strategy] = stats["fusion_strategies_used"].get(strategy, 0) + 1
            
            # Modality usage
            if len(pred.contributing_modalities) == 1:
                if "biosignal" in pred.contributing_modalities:
                    stats["modality_usage"]["biosignal_only"] += 1
                else:
                    stats["modality_usage"]["visual_only"] += 1
            else:
                stats["modality_usage"]["both"] += 1
            
            # Emotion distribution
            emotion = pred.discrete_emotion
            stats["emotion_distribution"][emotion] = stats["emotion_distribution"].get(emotion, 0) + 1
        
        return stats


# Example usage and testing functions
def test_late_fusion():
    """Test the late fusion module with sample data"""
    print("Testing Late Fusion Module...")
    
    # Initialize fusion module
    fusion_module = LateFusionModule(
        fusion_strategy=FusionStrategy.CONFIDENCE_BASED,
        weights={"biosignal": 0.6, "visual": 0.4}
    )
    
    # Test with sample data
    sample_signal = np.random.randn(140)  # Sample PPG signal
    sample_frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)  # Sample video frame
    
    # Get individual predictions
    bio_pred = fusion_module.predict_biosignal_emotion(sample_signal)
    print(f"Biosignal prediction: Valence={bio_pred.valence}, Arousal={bio_pred.arousal}, Conf={bio_pred.confidence:.3f}")
    
    # Note: Visual prediction might fail without actual face in the frame
    visual_pred = fusion_module.predict_visual_emotion(sample_frame)
    if visual_pred:
        print(f"Visual prediction: {visual_pred.emotion}, V={visual_pred.valence:.3f}, A={visual_pred.arousal:.3f}, Conf={visual_pred.confidence:.3f}")
    else:
        print("No face detected in sample frame")
    
    # Fuse predictions
    fused_pred = fusion_module.fuse_predictions(bio_pred, visual_pred)
    print(f"Fused prediction: {fused_pred.discrete_emotion}, V={fused_pred.valence:.3f}, A={fused_pred.arousal:.3f}")
    print(f"Fusion confidence: {fused_pred.fusion_confidence:.3f}, Strategy: {fused_pred.fusion_strategy}")
    
    return fusion_module, fused_pred


if __name__ == "__main__":
    # Run test
    fusion_module, prediction = test_late_fusion()
    print("\nLate Fusion Module test completed successfully!")
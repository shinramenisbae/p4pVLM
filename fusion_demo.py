
import numpy as np
import cv2
import matplotlib.pyplot as plt
import pandas as pd
import time
from pathlib import Path
import json

from late_fusion_module import (
    LateFusionModule, 
    FusionStrategy, 
    BiosignalPrediction, 
    VisualPrediction
)


class FusionDemo:
    """Demo class for Late Fusion Module"""
    
    def __init__(self):
        """Initialize the demo"""
        self.results = []
        
    def demo_fusion_strategies(self):
        """Demonstrate different fusion strategies"""
        print("=== Late Fusion Module Demo ===\n")
        
        # Test different fusion strategies
        strategies = [
            FusionStrategy.WEIGHTED_AVERAGE,
            FusionStrategy.CONFIDENCE_BASED, 
            FusionStrategy.RULE_BASED,
            FusionStrategy.ADAPTIVE_WEIGHTED
        ]
        
        for strategy in strategies:
            print(f"Testing {strategy.value} fusion:")
            self._test_strategy(strategy)
            print("-" * 50)
    
    def _test_strategy(self, strategy: FusionStrategy):
        """Test a specific fusion strategy"""
        
        # Initialize fusion module with current strategy
        fusion_module = LateFusionModule(
            fusion_strategy=strategy,
            weights={"biosignal": 0.7, "visual": 0.3}
        )
        
        # Simulate different scenarios
        scenarios = [
            {
                "name": "High confidence both modalities",
                "bio_data": {"valence": 1, "arousal": 1, "confidence": 0.85},
                "vis_data": {"emotion": "happy", "valence": 0.8, "arousal": 0.7, "confidence": 0.90}
            },
            {
                "name": "Low confidence biosignal",
                "bio_data": {"valence": 0, "arousal": 1, "confidence": 0.25},
                "vis_data": {"emotion": "angry", "valence": -0.7, "arousal": 0.8, "confidence": 0.88}
            },
            {
                "name": "Conflicting predictions",
                "bio_data": {"valence": 1, "arousal": 0, "confidence": 0.75},
                "vis_data": {"emotion": "sad", "valence": -0.6, "arousal": -0.3, "confidence": 0.82}
            },
            {
                "name": "Visual only (no biosignal)",
                "bio_data": None,
                "vis_data": {"emotion": "surprise", "valence": 0.2, "arousal": 0.9, "confidence": 0.78}
            }
        ]
        
        for scenario in scenarios:
            # Create prediction objects
            bio_pred = None
            if scenario["bio_data"]:
                bio_pred = BiosignalPrediction(
                    valence=scenario["bio_data"]["valence"],
                    arousal=scenario["bio_data"]["arousal"], 
                    confidence=scenario["bio_data"]["confidence"],
                    timestamp=time.time()
                )
            
            vis_pred = None
            if scenario["vis_data"]:
                vis_pred = VisualPrediction(
                    emotion=scenario["vis_data"]["emotion"],
                    valence=scenario["vis_data"]["valence"],
                    arousal=scenario["vis_data"]["arousal"],
                    confidence=scenario["vis_data"]["confidence"],
                    faces_detected=1,
                    timestamp=time.time()
                )
            
            # Fuse predictions
            fused = fusion_module.fuse_predictions(bio_pred, vis_pred)
            
            # Print results
            print(f"  Scenario: {scenario['name']}")
            if bio_pred:
                print(f"    Biosignal: V={bio_pred.valence}, A={bio_pred.arousal}, Conf={bio_pred.confidence:.3f}")
            if vis_pred:
                print(f"    Visual: {vis_pred.emotion}, V={vis_pred.valence:.2f}, A={vis_pred.arousal:.2f}, Conf={vis_pred.confidence:.3f}")
            
            print(f"    Fused: {fused.discrete_emotion}, V={fused.valence:.2f}, A={fused.arousal:.2f}")
            print(f"    Confidence: {fused.fusion_confidence:.3f}, Strategy: {fused.fusion_strategy}")
            print(f"    Contributions: Bio={fused.biosignal_contribution:.2f}, Vis={fused.visual_contribution:.2f}")
            print()
    
    def demo_realtime_simulation(self):
        """Simulate real-time fusion with synthetic data"""
        print("\n=== Real-time Simulation Demo ===")
        
        # Initialize fusion module
        fusion_module = LateFusionModule(
            fusion_strategy=FusionStrategy.CONFIDENCE_BASED
        )
        
        # Simulate 30 seconds of data (1Hz sampling)
        num_samples = 30
        results = []
        
        print(f"Simulating {num_samples} seconds of multimodal emotion recognition...")
        
        for i in range(num_samples):
            # Simulate biosignal data (PPG-like signal)
            t = np.linspace(0, 2*np.pi, 140)
            # Add some emotion-related variation
            base_signal = np.sin(t) + 0.3 * np.sin(3*t) + 0.1 * np.random.randn(140)
            
            # Simulate emotional state changes over time
            if i < 10:  # Start calm
                emotion_state = {"valence": 0.2, "arousal": -0.3}
            elif i < 20:  # Become excited/happy
                emotion_state = {"valence": 0.7, "arousal": 0.6}
            else:  # End tired/sad
                emotion_state = {"valence": -0.4, "arousal": -0.5}
            
            # Get biosignal prediction
            bio_pred = fusion_module.predict_biosignal_emotion(base_signal)
            
            # Simulate visual prediction (would normally come from camera)
            # Add some noise and variation to the target emotion state
            vis_valence = emotion_state["valence"] + 0.2 * np.random.randn()
            vis_arousal = emotion_state["arousal"] + 0.2 * np.random.randn()
            vis_confidence = 0.6 + 0.3 * np.random.random()
            
            # Map to discrete emotion for visual
            if vis_valence > 0.3 and vis_arousal > 0.3:
                vis_emotion = "happy"
            elif vis_valence < -0.3 and vis_arousal > 0.3:
                vis_emotion = "angry"
            elif vis_valence < -0.3 and vis_arousal < -0.3:
                vis_emotion = "sad"
            elif abs(vis_valence) < 0.3 and abs(vis_arousal) < 0.3:
                vis_emotion = "neutral"
            else:
                vis_emotion = "surprise"
            
            vis_pred = VisualPrediction(
                emotion=vis_emotion,
                valence=vis_valence,
                arousal=vis_arousal,
                confidence=vis_confidence,
                faces_detected=1,
                timestamp=time.time()
            )
            
            # Fuse predictions
            fused = fusion_module.fuse_predictions(bio_pred, vis_pred)
            
            # Store results
            results.append({
                "time": i,
                "biosignal_valence": (bio_pred.valence * 2) - 1,
                "biosignal_arousal": (bio_pred.arousal * 2) - 1,
                "biosignal_confidence": bio_pred.confidence,
                "visual_valence": vis_pred.valence,
                "visual_arousal": vis_pred.arousal,
                "visual_confidence": vis_pred.confidence,
                "visual_emotion": vis_pred.emotion,
                "fused_valence": fused.valence,
                "fused_arousal": fused.arousal,
                "fused_emotion": fused.discrete_emotion,
                "fused_confidence": fused.fusion_confidence,
                "bio_contribution": fused.biosignal_contribution,
                "vis_contribution": fused.visual_contribution
            })
            
            # Print progress
            if i % 5 == 0:
                print(f"  t={i:2d}s: {fused.discrete_emotion:>8s} | V={fused.valence:5.2f} A={fused.arousal:5.2f} | Conf={fused.fusion_confidence:.3f}")
        
        self.results = results
        return results
    
    def plot_fusion_results(self, results=None):
        """Plot the fusion results"""
        if results is None:
            results = self.results
            
        if not results:
            print("No results to plot. Run demo_realtime_simulation() first.")
            return
        
        df = pd.DataFrame(results)
        
        # Create subplots
        fig, axes = plt.subplots(3, 2, figsize=(15, 12))
        fig.suptitle('Late Fusion Multimodal Emotion Recognition Results', fontsize=16)
        
        # Plot 1: Valence over time
        axes[0, 0].plot(df['time'], df['biosignal_valence'], 'b-', label='Biosignal', alpha=0.7)
        axes[0, 0].plot(df['time'], df['visual_valence'], 'r-', label='Visual', alpha=0.7)
        axes[0, 0].plot(df['time'], df['fused_valence'], 'g-', label='Fused', linewidth=2)
        axes[0, 0].set_title('Valence Over Time')
        axes[0, 0].set_ylabel('Valence')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Plot 2: Arousal over time  
        axes[0, 1].plot(df['time'], df['biosignal_arousal'], 'b-', label='Biosignal', alpha=0.7)
        axes[0, 1].plot(df['time'], df['visual_arousal'], 'r-', label='Visual', alpha=0.7)
        axes[0, 1].plot(df['time'], df['fused_arousal'], 'g-', label='Fused', linewidth=2)
        axes[0, 1].set_title('Arousal Over Time')
        axes[0, 1].set_ylabel('Arousal')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Plot 3: Confidence levels
        axes[1, 0].plot(df['time'], df['biosignal_confidence'], 'b-', label='Biosignal')
        axes[1, 0].plot(df['time'], df['visual_confidence'], 'r-', label='Visual')
        axes[1, 0].plot(df['time'], df['fused_confidence'], 'g-', label='Fused')
        axes[1, 0].set_title('Confidence Levels')
        axes[1, 0].set_ylabel('Confidence')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # Plot 4: Modality contributions
        axes[1, 1].plot(df['time'], df['bio_contribution'], 'b-', label='Biosignal Weight')
        axes[1, 1].plot(df['time'], df['vis_contribution'], 'r-', label='Visual Weight')
        axes[1, 1].set_title('Modality Contribution Weights')
        axes[1, 1].set_ylabel('Weight')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        # Plot 5: Valence-Arousal space
        scatter = axes[2, 0].scatter(df['fused_valence'], df['fused_arousal'], 
                                   c=df['time'], cmap='viridis', alpha=0.7)
        axes[2, 0].set_title('Emotion Trajectory in Valence-Arousal Space')
        axes[2, 0].set_xlabel('Valence')
        axes[2, 0].set_ylabel('Arousal')
        axes[2, 0].grid(True, alpha=0.3)
        plt.colorbar(scatter, ax=axes[2, 0], label='Time (s)')
        
        # Plot 6: Emotion distribution
        emotion_counts = df['fused_emotion'].value_counts()
        axes[2, 1].pie(emotion_counts.values, labels=emotion_counts.index, autopct='%1.1f%%')
        axes[2, 1].set_title('Fused Emotion Distribution')
        
        plt.tight_layout()
        plt.show()
    
    def compare_fusion_strategies(self):
        """Compare different fusion strategies on the same data"""
        print("\n=== Fusion Strategy Comparison ===")
        
        strategies = [
            FusionStrategy.WEIGHTED_AVERAGE,
            FusionStrategy.CONFIDENCE_BASED,
            FusionStrategy.RULE_BASED
        ]
        
        # Test scenario with conflicting predictions
        bio_pred = BiosignalPrediction(
            valence=1, arousal=0, confidence=0.75, timestamp=time.time()
        )
        vis_pred = VisualPrediction(
            emotion="sad", valence=-0.6, arousal=-0.3, 
            confidence=0.85, faces_detected=1, timestamp=time.time()
        )
        
        print("Test scenario: Conflicting predictions")
        print(f"Biosignal: Valence=1, Arousal=0, Confidence=0.75")
        print(f"Visual: sad, Valence=-0.6, Arousal=-0.3, Confidence=0.85\n")
        
        comparison_results = []
        
        for strategy in strategies:
            fusion_module = LateFusionModule(fusion_strategy=strategy)
            fused = fusion_module.fuse_predictions(bio_pred, vis_pred)
            
            result = {
                "strategy": strategy.value,
                "valence": fused.valence,
                "arousal": fused.arousal,
                "emotion": fused.discrete_emotion,
                "confidence": fused.fusion_confidence,
                "bio_weight": fused.biosignal_contribution,
                "vis_weight": fused.visual_contribution
            }
            comparison_results.append(result)
            
            print(f"{strategy.value:>18s}: {fused.discrete_emotion:>8s} | V={fused.valence:5.2f} A={fused.arousal:5.2f} | "
                  f"Bio={fused.biosignal_contribution:.2f} Vis={fused.visual_contribution:.2f}")
        
        return comparison_results
    
    def save_demo_results(self, filename="fusion_demo_results.json"):
        """Save demo results to file"""
        if not self.results:
            print("No results to save. Run demo_realtime_simulation() first.")
            return
        
        output_data = {
            "metadata": {
                "timestamp": time.time(),
                "num_samples": len(self.results),
                "demo_type": "realtime_simulation"
            },
            "results": self.results
        }
        
        with open(filename, 'w') as f:
            json.dump(output_data, f, indent=2)
        
        print(f"Demo results saved to {filename}")


def main():
    """Main demo function"""
    # Create demo instance
    demo = FusionDemo()
    
    # Run different demos
    print("Starting Late Fusion Module demonstrations...\n")
    
    # 1. Test different fusion strategies
    demo.demo_fusion_strategies()
    
    # 2. Run real-time simulation
    results = demo.demo_realtime_simulation()
    
    # 3. Compare fusion strategies
    demo.compare_fusion_strategies()
    
    # 4. Plot results (if matplotlib is available)
    try:
        demo.plot_fusion_results()
    except ImportError:
        print("\nMatplotlib not available. Skipping plots.")
    
    # 5. Save results
    demo.save_demo_results()
    
    print("\n=== Demo Complete ===")
    print("The Late Fusion Module successfully combines biosignal and visual emotion predictions")
    print("using various fusion strategies including weighted average, confidence-based,")
    print("rule-based, and adaptive weighting approaches.")


if __name__ == "__main__":
    main()
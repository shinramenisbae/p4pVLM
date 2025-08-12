
import numpy as np
import cv2
import torch
import time
import os
from pathlib import Path

# Add local imports
import sys
sys.path.append('active')
sys.path.append('passive/model/network')

from late_fusion_module import LateFusionModule, FusionStrategy

def simulate_ppg_signal(emotion_state="neutral", duration_seconds=2, sampling_rate=128):
    """
    Simulate a PPG signal with emotional characteristics
    
    Args:
        emotion_state: Target emotional state to simulate
        duration_seconds: Length of signal in seconds
        sampling_rate: Sampling rate in Hz
    
    Returns:
        numpy array of simulated PPG signal
    """
    num_samples = int(duration_seconds * sampling_rate)
    t = np.linspace(0, duration_seconds, num_samples)
    
    # Base PPG signal (simulated heartbeat)
    base_freq = 1.2  # ~72 BPM baseline
    
    # Emotional modulation of heart rate
    if emotion_state == "happy":
        heart_rate_factor = 1.1  # Slightly elevated
        variability = 0.15
    elif emotion_state == "angry":
        heart_rate_factor = 1.3  # Elevated heart rate
        variability = 0.25
    elif emotion_state == "sad":
        heart_rate_factor = 0.9  # Slightly lower
        variability = 0.08
    elif emotion_state == "fear":
        heart_rate_factor = 1.4  # High heart rate
        variability = 0.3
    else:  # neutral
        heart_rate_factor = 1.0
        variability = 0.1
    
    # Generate PPG-like signal
    signal = np.sin(2 * np.pi * base_freq * heart_rate_factor * t)
    signal += 0.3 * np.sin(2 * np.pi * base_freq * heart_rate_factor * 2 * t)  # Harmonics
    signal += variability * np.random.randn(num_samples)  # Noise and variability
    
    # Add respiratory modulation
    resp_freq = 0.25  # ~15 breaths per minute
    signal += 0.1 * np.sin(2 * np.pi * resp_freq * t)
    
    # Normalize to reasonable range
    signal = (signal - np.mean(signal)) / np.std(signal)
    signal = signal * 100 + 1000  # Scale to typical PPG range
    
    return signal

def create_synthetic_face_image(emotion="neutral", image_size=(480, 640)):
    """
    Create a synthetic face-like image for testing
    (In practice, this would be replaced with actual camera input)
    """
    height, width = image_size
    image = np.zeros((height, width, 3), dtype=np.uint8)
    
    # Create a simple face-like structure
    center_x, center_y = width // 2, height // 2
    
    # Face outline (circle)
    cv2.circle(image, (center_x, center_y), 100, (180, 150, 120), -1)
    
    # Eyes
    cv2.circle(image, (center_x - 30, center_y - 20), 8, (50, 50, 50), -1)
    cv2.circle(image, (center_x + 30, center_y - 20), 8, (50, 50, 50), -1)
    
    # Nose
    cv2.circle(image, (center_x, center_y), 5, (160, 130, 100), -1)
    
    # Mouth (emotion-dependent)
    mouth_y = center_y + 30
    if emotion == "happy":
        # Smile
        cv2.ellipse(image, (center_x, mouth_y), (25, 12), 0, 0, 180, (50, 50, 50), 2)
    elif emotion == "sad":
        # Frown
        cv2.ellipse(image, (center_x, mouth_y + 10), (25, 12), 0, 180, 360, (50, 50, 50), 2)
    elif emotion == "angry":
        # Straight line
        cv2.line(image, (center_x - 20, mouth_y), (center_x + 20, mouth_y), (50, 50, 50), 3)
    else:  # neutral
        cv2.ellipse(image, (center_x, mouth_y), (15, 5), 0, 0, 180, (50, 50, 50), 2)
    
    return image

def run_multimodal_session(duration_minutes=2):
    """
    Run a multimodal emotion recognition session
    
    Args:
        duration_minutes: Duration of the session in minutes
    """
    print("=== Multimodal Emotion Recognition Session ===\n")
    
    # Initialize fusion module
    fusion_module = LateFusionModule(
        fusion_strategy=FusionStrategy.CONFIDENCE_BASED,
        weights={"biosignal": 0.6, "visual": 0.4}
    )
    
    # Session parameters
    sample_interval = 2.0  # seconds between samples
    total_samples = int(duration_minutes * 60 / sample_interval)
    
    # Simulate emotional states over time
    emotion_timeline = [
        (0.0, 0.3, "neutral"),
        (0.3, 0.6, "happy"), 
        (0.6, 0.8, "angry"),
        (0.8, 1.0, "sad")
    ]
    
    results = []
    
    print(f"Running {total_samples} samples over {duration_minutes} minutes...")
    print("Time | Bio Pred    | Visual Pred | Fused Result | Confidence | Strategy")
    print("-" * 80)
    
    for i in range(total_samples):
        # Determine current emotional state
        progress = i / total_samples
        current_emotion = "neutral"
        for start, end, emotion in emotion_timeline:
            if start <= progress < end:
                current_emotion = emotion
                break
        
        # Generate biosignal data
        ppg_signal = simulate_ppg_signal(current_emotion, duration_seconds=2)
        # Take 140 samples (typical window size)
        signal_window = ppg_signal[:140]
        
        # Generate visual data
        face_image = create_synthetic_face_image(current_emotion)
        
        # Get predictions
        bio_pred = fusion_module.predict_biosignal_emotion(signal_window)
        visual_pred = fusion_module.predict_visual_emotion(face_image)
        
        # Fuse predictions
        fused_pred = fusion_module.fuse_predictions(bio_pred, visual_pred)
        
        # Format biosignal prediction
        bio_str = f"V:{bio_pred.valence} A:{bio_pred.arousal} ({bio_pred.confidence:.2f})"
        
        # Format visual prediction  
        if visual_pred:
            vis_str = f"{visual_pred.emotion[:4]} V:{visual_pred.valence:.1f} A:{visual_pred.arousal:.1f} ({visual_pred.confidence:.2f})"
        else:
            vis_str = "No face detected"
        
        # Format fused result
        fused_str = f"{fused_pred.discrete_emotion[:4]} V:{fused_pred.valence:.1f} A:{fused_pred.arousal:.1f}"
        
        # Print results
        elapsed_time = i * sample_interval
        print(f"{elapsed_time:4.0f}s| {bio_str:11s} | {vis_str:11s} | {fused_str:12s} | "
              f"{fused_pred.fusion_confidence:.3f}      | {fused_pred.fusion_strategy[:4]}")
        
        # Store results
        results.append({
            "time": elapsed_time,
            "target_emotion": current_emotion,
            "biosignal": bio_pred,
            "visual": visual_pred,
            "fused": fused_pred
        })
        
        # Small delay to simulate real-time processing
        time.sleep(0.1)
    
    return results

def analyze_session_results(results):
    """Analyze the results from a multimodal session"""
    print("\n=== Session Analysis ===")
    
    # Basic statistics
    total_predictions = len(results)
    visual_detections = sum(1 for r in results if r["visual"] is not None)
    avg_confidence = np.mean([r["fused"].fusion_confidence for r in results])
    
    print(f"Total predictions: {total_predictions}")
    print(f"Visual detections: {visual_detections}/{total_predictions} ({visual_detections/total_predictions*100:.1f}%)")
    print(f"Average fusion confidence: {avg_confidence:.3f}")
    
    # Emotion distribution
    emotions = [r["fused"].discrete_emotion for r in results]
    unique_emotions = list(set(emotions))
    print(f"\nEmotion distribution:")
    for emotion in unique_emotions:
        count = emotions.count(emotion)
        print(f"  {emotion}: {count} ({count/total_predictions*100:.1f}%)")
    
    # Fusion strategy usage
    strategies = [r["fused"].fusion_strategy for r in results]
    unique_strategies = list(set(strategies))
    print(f"\nFusion strategy usage:")
    for strategy in unique_strategies:
        count = strategies.count(strategy)
        print(f"  {strategy}: {count} ({count/total_predictions*100:.1f}%)")
    
    # Modality contributions
    bio_contributions = [r["fused"].biosignal_contribution for r in results if r["fused"].biosignal_contribution > 0]
    vis_contributions = [r["fused"].visual_contribution for r in results if r["fused"].visual_contribution > 0]
    
    if bio_contributions:
        print(f"\nAverage biosignal contribution: {np.mean(bio_contributions):.3f}")
    if vis_contributions:
        print(f"Average visual contribution: {np.mean(vis_contributions):.3f}")

def test_different_strategies():
    """Test all fusion strategies on the same data"""
    print("\n=== Fusion Strategy Comparison ===")
    
    strategies = [
        FusionStrategy.WEIGHTED_AVERAGE,
        FusionStrategy.CONFIDENCE_BASED,
        FusionStrategy.RULE_BASED,
        FusionStrategy.ADAPTIVE_WEIGHTED
    ]
    
    # Generate test data
    ppg_signal = simulate_ppg_signal("happy")[:140]
    face_image = create_synthetic_face_image("happy")
    
    print("Test data: Happy emotion scenario")
    print(f"Strategy            | Emotion | Valence | Arousal | Confidence | Bio Weight | Vis Weight")
    print("-" * 90)
    
    for strategy in strategies:
        fusion_module = LateFusionModule(fusion_strategy=strategy)
        
        bio_pred = fusion_module.predict_biosignal_emotion(ppg_signal)
        visual_pred = fusion_module.predict_visual_emotion(face_image)
        fused_pred = fusion_module.fuse_predictions(bio_pred, visual_pred)
        
        print(f"{strategy.value:18s} | {fused_pred.discrete_emotion:7s} | "
              f"{fused_pred.valence:7.2f} | {fused_pred.arousal:7.2f} | "
              f"{fused_pred.fusion_confidence:10.3f} | {fused_pred.biosignal_contribution:10.2f} | "
              f"{fused_pred.visual_contribution:10.2f}")

def main():
    """Main example function"""
    print("Late Fusion Module - Example Integration\n")
    
    # Check if model files exist
    model_path = "passive/model/network/emotion_cnn.pth"
    if not os.path.exists(model_path):
        print(f"Warning: Biosignal model not found at {model_path}")
        print("This example will use synthetic data for demonstration.\n")
    
    try:
        # Test different fusion strategies
        test_different_strategies()
        
        # Run a short multimodal session
        results = run_multimodal_session(duration_minutes=0.5)  # 30 seconds for demo
        
        # Analyze results
        analyze_session_results(results)
        
        print("\n=== Integration Example Complete ===")
        print("The Late Fusion Module successfully demonstrated:")
        print("- Multiple fusion strategies")
        print("- Real-time multimodal processing simulation")
        print("- Confidence-based weighting")
        print("- Comprehensive result analysis")
        
    except Exception as e:
        print(f"Error during integration example: {e}")
        import traceback
        traceback.print_exc()
        print("\nNote: This example uses simulated data for demonstration.")
        print("For real data, connect actual PPG sensors and camera feeds.")

if __name__ == "__main__":
    main()
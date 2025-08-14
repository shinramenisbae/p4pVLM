"""
Interactive test for the Late Fusion Module.

Prompts the user to:
- Select a fusion strategy (weighted average, confidence-based, or rule-based)
- Enter passive (biosignal) predictions: valence (0..1), arousal (0..1), confidence (0..1)
- Enter active (visual) predictions: valence (-1..1), arousal (-1..1), confidence (0..1)

Then uses late_fusion_module.LateFusionModule to produce the final fused prediction.
"""

import time
from typing import Optional

from late_fusion_module import (
    LateFusionModule,
    FusionStrategy,
    BiosignalPrediction,
    VisualPrediction,
)


def prompt_choice(prompt: str, options: dict) -> str:
    while True:
        print(prompt)
        for key, label in options.items():
            print(f"  {key}) {label}")
        raw = input("> ").strip().lower()
        if raw in options:
            return options[raw]
        # allow entering the label directly
        for label in options.values():
            if raw == label.lower():
                return label
        print("Invalid selection. Please try again.\n")


def prompt_int(prompt: str, valid_values: Optional[list] = None) -> int:
    while True:
        raw = input(f"{prompt} ").strip()
        try:
            value = int(raw)
            if valid_values is None or value in valid_values:
                return value
            print(f"Please enter one of: {valid_values}")
        except ValueError:
            print("Invalid integer. Please try again.")


def prompt_float(prompt: str, min_value: Optional[float] = None, max_value: Optional[float] = None) -> float:
    while True:
        raw = input(f"{prompt} ").strip()
        try:
            value = float(raw)
            if (min_value is None or value >= min_value) and (max_value is None or value <= max_value):
                return value
            bounds = []
            if min_value is not None:
                bounds.append(f">= {min_value}")
            if max_value is not None:
                bounds.append(f"<= {max_value}")
            print("Please enter a value " + " and ".join(bounds))
        except ValueError:
            print("Invalid number. Please try again.")


def main():
    print("=== Interactive Late Fusion Test ===\n")

    # Select fusion strategy
    strategy_label = prompt_choice(
        "Select fusion strategy:",
        {
            "1": "weighted_average",
            "2": "confidence_based",
            "3": "rule_based",
        },
    )

    strategy_map = {
        "weighted_average": FusionStrategy.WEIGHTED_AVERAGE,
        "confidence_based": FusionStrategy.CONFIDENCE_BASED,
        "rule_based": FusionStrategy.RULE_BASED,
    }
    fusion_strategy = strategy_map[strategy_label]

    # Collect passive (biosignal) inputs
    print("\nEnter passive (biosignal) predictions:")
    bio_valence = prompt_float("- Valence (0..1):", 0.0, 1.0)
    bio_arousal = prompt_float("- Arousal (0..1):", 0.0, 1.0)
    bio_conf = prompt_float("- Confidence (0..1):", 0.0, 1.0)

    # Collect active (visual) inputs
    print("\nEnter active (visual) predictions:")
    vis_valence = prompt_float("- Valence (-1..1):", -1.0, 1.0)
    vis_arousal = prompt_float("- Arousal (-1..1):", -1.0, 1.0)
    vis_conf = prompt_float("- Confidence (0..1):", 0.0, 1.0)

    # Initialize fusion module
    try:
        fusion_module = LateFusionModule(
            fusion_strategy=fusion_strategy,
            weights={"biosignal": 0.6, "visual": 0.4},
            enable_visual=False,  # skip heavy visual model init; we provide visual inputs manually
        )
    except Exception as e:
        print(f"Failed to initialize LateFusionModule: {e}")
        return

    # Derive a discrete emotion label for the visual input using the module's mapping
    try:
        vis_emotion = fusion_module._map_va_to_emotion(vis_valence, vis_arousal)  # type: ignore[attr-defined]
    except Exception:
        vis_emotion = "neutral"

    # Build prediction objects
    bio_pred = BiosignalPrediction(
        valence=bio_valence,
        arousal=bio_arousal,
        confidence=bio_conf,
        timestamp=time.time(),
    )

    vis_pred = VisualPrediction(
        emotion=vis_emotion,
        valence=vis_valence,
        arousal=vis_arousal,
        confidence=vis_conf,
        faces_detected=1,
        timestamp=time.time(),
    )

    # Fuse
    fused = fusion_module.fuse_predictions(bio_pred, vis_pred)

    # Display results
    print("\n=== Fused Prediction ===")
    print(f"Strategy          : {fused.fusion_strategy}")
    print(f"Discrete Emotion  : {fused.discrete_emotion}")
    print(f"Valence (final)   : {fused.valence:.3f}")
    print(f"Arousal (final)   : {fused.arousal:.3f}")
    print(f"Confidence        : {fused.fusion_confidence:.3f}")
    if fused.biosignal_contribution or fused.visual_contribution:
        print(f"Bio Contribution  : {fused.biosignal_contribution:.2f}")
        print(f"Vis Contribution  : {fused.visual_contribution:.2f}")


if __name__ == "__main__":
    main()
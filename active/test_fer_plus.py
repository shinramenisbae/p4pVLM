from emotion_detector import EmotionDetector


def main():
    detector = EmotionDetector(use_vllm=False)

    fer_plus_path = "./train.csv"

    results = detector.evaluate_on_dataset(fer_plus_path, subset_size=28000)

    if results and results["accuracy"] >= 0.70:
        print("✅ Milestone achieved! Accuracy >= 70%")
    else:
        accuracy = results["accuracy"] if results else 0
        print(f"❌ Milestone not met. Accuracy: {accuracy:.2%}")


if __name__ == "__main__":
    main()

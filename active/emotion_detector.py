import cv2
import torch
from PIL import Image
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import time
import os
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report
import numpy as np
from tqdm import tqdm
from datetime import datetime


class EmotionDetector:
    def __init__(
        self, use_vllm=False, output_dir="webcam_captures", enable_valence_arousal=True
    ):
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )
        self.use_vllm = use_vllm
        self.output_dir = output_dir
        self.enable_valence_arousal = enable_valence_arousal

        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
            print(f"Created output directory: {self.output_dir}")

        print("Loading emotion recognition model...")
        try:
            if use_vllm:
                self.model = AutoModelForCausalLM.from_pretrained(
                    "microsoft/kosmos-2-patch14-224",
                    torch_dtype=torch.float16,
                    device_map="auto",
                )
                self.tokenizer = AutoTokenizer.from_pretrained(
                    "microsoft/kosmos-2-patch14-224"
                )
            else:
                # load emotion classifier, note that this code snippet is part of the class constructor __init__
                self.emotion_classifier = pipeline(
                    "image-classification",
                    model="dima806/facial_emotions_image_detection",
                    use_fast=True,
                    device="cuda" if torch.cuda.is_available() else "cpu",
                )
        except Exception as e:
            print(f"Error loading model: {e}")
            raise

        self.emotion_mapping = {
            "angry": "angry",
            "anger": "angry",
            "disgust": "disgust",
            "disgusted": "disgust",
            "fear": "fear",
            "fearful": "fear",
            "afraid": "fear",
            "happy": "happy",
            "happiness": "happy",
            "joy": "happy",
            "joyful": "happy",
            "sad": "sad",
            "sadness": "sad",
            "sorrow": "sad",
            "surprise": "surprise",
            "surprised": "surprise",
            "neutral": "neutral",
            "contempt": "contempt",
            "excited": "happy",
            "disappointed": "sad",
            "worried": "fear",
            "calm": "neutral",
            "content": "happy",
        }

        # Valence-Arousal mapping for late fusion (Russell's Circumplex Model)
        # Valence: -1 (negative) to +1 (positive)
        # Arousal: -1 (calm) to +1 (excited)
        self.emotion_to_valence_arousal = {
            "angry": {"valence": -0.8, "arousal": 0.8},  # Negative, High arousal
            "disgust": {"valence": -0.7, "arousal": 0.3},  # Negative, Medium arousal
            "fear": {
                "valence": -0.9,
                "arousal": 0.9,
            },  # Very negative, Very high arousal
            "happy": {"valence": 0.8, "arousal": 0.6},  # Positive, Medium-high arousal
            "sad": {"valence": -0.8, "arousal": -0.4},  # Negative, Low arousal
            "surprise": {
                "valence": 0.1,
                "arousal": 0.8,
            },  # Slightly positive, High arousal
            "neutral": {"valence": 0.0, "arousal": 0.0},  # Neutral valence and arousal
            "contempt": {"valence": -0.5, "arousal": 0.5},  # Negative, Medium arousal
        }

    def save_classified_frame(self, frame, emotions_detected):
        """Save the frame with detected emotions to the output directory"""
        try:
            # Skip saving if no faces/emotions detected
            if not emotions_detected:
                return

            # Generate timestamp for filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]

            # Create emotion summary for filename
            emotion_summary = "_".join(
                [result["emotion"] for result in emotions_detected]
            )
            # Sanitize filename by removing/replacing invalid characters
            emotion_summary = "".join(
                c for c in emotion_summary if c.isalnum() or c in "_ -"
            ).strip()

            filename = f"webcam_{timestamp}_{emotion_summary}.jpg"
            filepath = os.path.join(self.output_dir, filename)

            cv2.imwrite(filepath, frame)
            print(f"Saved classified frame: {filename}")

        except Exception as e:
            print(f"Error saving frame: {e}")

    def detect_emotions(self, frame):
        # convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # detect faces
        faces = self.face_cascade.detectMultiScale(gray, 1.1, 4)

        # store results
        results = []
        for x, y, w, h in faces:
            # crop face
            face_img = frame[y : y + h, x : x + w]

            # convert to PIL image
            pil_img = Image.fromarray(cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB))

            try:
                predictions = self.emotion_classifier(pil_img)
                emotion = predictions[0]["label"]
                confidence = predictions[0]["score"]

                # Get valence/arousal values for late fusion
                valence_arousal = self.map_emotion_to_valence_arousal(
                    emotion, confidence
                )

                # Create display text with valence/arousal if available
                if valence_arousal and self.enable_valence_arousal:
                    display_text = f"{emotion} ({confidence:.2f}) V:{valence_arousal['valence']:.2f} A:{valence_arousal['arousal']:.2f}"
                else:
                    display_text = f"{emotion} ({confidence:.2f})"

                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
                cv2.putText(
                    frame,
                    display_text,
                    (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.9,
                    (36, 255, 12),
                    2,
                )

                result_data = {
                    "position": (x, y, w, h),
                    "emotion": emotion,
                    "confidence": confidence,
                }

                if valence_arousal:
                    result_data.update(valence_arousal)

                results.append(result_data)
            except Exception as e:
                print(f"Error analyzing emotion: {e}")
                # Print more detailed debug info
                import traceback

                traceback.print_exc()

        return frame, results

    def start_webcam(self, camera_index=0):
        # Access webcam with specified index
        cap = cv2.VideoCapture(camera_index)

        if not cap.isOpened():
            print(f"Error: Could not open webcam at index {camera_index}.")
            return

        print(
            f"Starting webcam feed from camera {camera_index}. Press 'q' to quit, 'c' to change camera."
        )
        print(f"All classified frames will be saved to: {self.output_dir}")

        last_process_time = time.time()
        process_interval = 1.0  # Process every 1 second to avoid overloading

        # Variables to store the latest detection results
        latest_results = []

        while True:
            # Read frame from webcam
            ret, frame = cap.read()

            if not ret:
                print("Error: Failed to capture image")
                break

            current_time = time.time()

            # Process frames at intervals to avoid overloading
            if current_time - last_process_time >= process_interval:
                # Process the frame
                processed_frame, emotions = self.detect_emotions(
                    frame.copy()
                )  # Work on a copy

                # Save the processed frame with detected emotions
                self.save_classified_frame(processed_frame, emotions)

                latest_results = emotions  # Store the latest results
                frame = processed_frame
                last_process_time = current_time
            else:
                # Draw the previously detected results on the current frame
                for result in latest_results:
                    x, y, w, h = result["position"]
                    emotion = result["emotion"]
                    confidence = result.get("confidence", 0.0)

                    # Create display text with valence/arousal if available
                    if (
                        self.enable_valence_arousal
                        and "valence" in result
                        and "arousal" in result
                    ):
                        display_text = f"{emotion} ({confidence:.2f}) V:{result['valence']:.2f} A:{result['arousal']:.2f}"
                    else:
                        display_text = f"{emotion} ({confidence:.2f})"

                    # Draw rectangle and emotion on the frame
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
                    cv2.putText(
                        frame,
                        display_text,
                        (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.9,
                        (36, 255, 12),
                        2,
                    )

            # Display the frame
            cv2.imshow("Emotion Detection", frame)

            # Check for key presses
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == ord("c"):
                # Release current camera
                cap.release()

                # Switch to next camera
                new_camera = (camera_index + 1) % 10  # Try up to 10 cameras
                cap = cv2.VideoCapture(new_camera)

                if cap.isOpened():
                    camera_index = new_camera
                    print(f"Switched to camera {camera_index}")
                else:
                    print(f"Could not open camera {new_camera}, trying again...")
                    # Try to reopen the previous camera
                    cap = cv2.VideoCapture(camera_index)
                    if not cap.isOpened():
                        print("Error reopening previous camera. Exiting.")
                        break

        # Release resources
        cap.release()
        cv2.destroyAllWindows()

        print(
            f"Webcam session ended. All captured frames are saved in: {self.output_dir}"
        )

    def load_fer_plus_dataset(self, data_path):
        """Load FER+ dataset for testing - handles CSV format"""
        if not os.path.exists(data_path):
            print(f"Dataset path {data_path} does not exist")
            return None, None

        try:
            # Read the CSV file
            df = pd.read_csv(data_path)

            # Check if required columns exist
            if "pixels" not in df.columns:
                print("CSV file must contain 'pixels' column")
                return None, None

            images = []
            labels = []

            # Emotion mapping for FER2013: 0=Angry, 1=Disgust, 2=Fear, 3=Happy, 4=Sad, 5=Surprise, 6=Neutral
            emotion_map = {
                0: "angry",
                1: "disgust",
                2: "fear",
                3: "happy",
                4: "sad",
                5: "surprise",
                6: "neutral",
            }

            print(f"Processing {len(df)} samples from CSV...")

            for idx, row in df.iterrows():
                try:
                    # Convert pixel string to numpy array
                    pixel_values = np.array(
                        [int(x) for x in row["pixels"].split()], dtype=np.uint8
                    )

                    # Reshape to 48x48 image
                    if len(pixel_values) != 48 * 48:
                        print(
                            f"Skipping sample {idx}: incorrect pixel count ({len(pixel_values)})"
                        )
                        continue

                    img_array = pixel_values.reshape(48, 48)

                    # Convert to PIL Image and then to RGB (the model expects RGB)
                    pil_img = Image.fromarray(img_array, mode="L").convert("RGB")
                    images.append(pil_img)

                    # Get emotion label if available (for train.csv)
                    if "emotion" in df.columns:
                        emotion_id = int(row["emotion"])
                        emotion_name = emotion_map.get(emotion_id, "unknown")
                        labels.append(emotion_name)
                    else:
                        # For test.csv without labels
                        labels.append("unknown")

                except Exception as e:
                    print(f"Error processing sample {idx}: {e}")
                    continue

            print(f"Successfully loaded {len(images)} images")
            return images, labels

        except Exception as e:
            print(f"Error reading CSV file: {e}")
            return None, None

    def preprocess_image_for_testing(self, image):
        """Preprocess image for model testing - handles both file paths and PIL images"""
        try:
            # If it's already a PIL image (from CSV), use it directly
            if isinstance(image, Image.Image):
                pil_img = image
            else:
                # If it's a file path, load it
                img = cv2.imread(image)
                if img is None:
                    return None

                # Convert to grayscale for face detection
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

                # Detect faces
                faces = self.face_cascade.detectMultiScale(gray, 1.1, 4)

                if len(faces) == 0:
                    # If no face detected, use whole image
                    face_img = img
                else:
                    # Use the largest face
                    x, y, w, h = max(faces, key=lambda face: face[2] * face[3])
                    face_img = img[y : y + h, x : x + w]

                # Convert to PIL for model
                pil_img = Image.fromarray(cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB))

            return pil_img

        except Exception as e:
            print(f"Error preprocessing image: {e}")
            return None

    def predict_emotion_batch(self, images):
        """Predict emotions for a batch of images"""
        predictions = []
        raw_predictions = []  # Debug: store raw predictions

        for img in tqdm(images, desc="Processing images"):
            pil_img = self.preprocess_image_for_testing(img)
            if pil_img is None:
                predictions.append("neutral")
                raw_predictions.append("neutral")
                continue

            try:
                if self.use_vllm:
                    # Use Visual LLM for prediction
                    prompt = "What emotion is shown in this face? Answer with one word: angry, disgust, fear, happy, neutral, sad, or surprise."
                    prediction = self.predict_with_vllm(pil_img, prompt)
                else:
                    # Use current CNN model
                    result = self.emotion_classifier(pil_img)
                    raw_prediction = result[0]["label"].lower()
                    raw_predictions.append(
                        raw_prediction
                    )  # Debug: store raw prediction

                    # Normalize prediction to match FER2013 labels
                    prediction = self.emotion_mapping.get(
                        raw_prediction, raw_prediction
                    )

                predictions.append(prediction)

            except Exception as e:
                print(f"Error predicting emotion: {e}")
                predictions.append("neutral")
                raw_predictions.append("error")

        # Debug: Print unique predictions
        unique_raw = set(raw_predictions)
        unique_mapped = set(predictions)
        print(f"\nDEBUG - Raw model outputs: {unique_raw}")
        print(f"DEBUG - Mapped predictions: {unique_mapped}")

        return predictions

    def evaluate_on_dataset(self, data_path, subset_size=None):
        """Evaluate model on FER+ dataset"""
        print("Loading dataset...")
        images, true_labels = self.load_fer_plus_dataset(data_path)

        if images is None:
            print("Failed to load dataset")
            return None

        # Check if we have labeled data
        has_labels = True if true_labels and true_labels[0] != "unknown" else False

        if not has_labels:
            print(
                "⚠️  Dataset appears to be unlabeled (test set). Will return predictions without accuracy metrics."
            )

        # Use subset if specified
        if subset_size and subset_size < len(images):
            indices = np.random.choice(len(images), subset_size, replace=False)
            images = [images[i] for i in indices]
            if has_labels:
                true_labels = [true_labels[i] for i in indices]

        print(f"Testing on {len(images)} images...")

        # Get predictions
        predicted_labels = self.predict_emotion_batch(images)

        if has_labels:
            # Debug: Print unique labels
            unique_true = set(true_labels)
            unique_pred = set(predicted_labels)
            print(f"\nDEBUG - True labels: {unique_true}")
            print(f"DEBUG - Predicted labels: {unique_pred}")

            # Calculate accuracy
            accuracy = accuracy_score(true_labels, predicted_labels)

            # Get the union of all labels for classification report
            all_labels = sorted(list(unique_true.union(unique_pred)))

            # Generate detailed report with dynamic labels
            report = classification_report(
                true_labels,
                predicted_labels,
                labels=all_labels,  # Use actual labels present in data
                target_names=all_labels,  # Use same labels as target names
                zero_division=0,
            )

            print(f"\nAccuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
            print("\nDetailed Classification Report:")
            print(report)

            return {
                "accuracy": accuracy,
                "predictions": predicted_labels,
                "true_labels": true_labels,
                "report": report,
            }
        else:
            # For unlabeled data, just return predictions and summary
            from collections import Counter

            prediction_counts = Counter(predicted_labels)

            print(f"\nPrediction Summary:")
            print(f"Total predictions: {len(predicted_labels)}")
            for emotion, count in sorted(prediction_counts.items()):
                percentage = (count / len(predicted_labels)) * 100
                print(f"  {emotion}: {count} ({percentage:.1f}%)")

            return {
                "accuracy": None,
                "predictions": predicted_labels,
                "true_labels": None,
                "report": None,
                "prediction_summary": prediction_counts,
            }

    def predict_with_vllm(self, image, prompt):
        """Placeholder for Visual LLM inference"""
        # This would need to be implemented based on the specific VLM you choose
        # For now, fall back to the CNN model
        result = self.emotion_classifier(image)
        return result[0]["label"].lower()

    def map_emotion_to_valence_arousal(self, emotion, confidence):
        """Map discrete emotion to continuous valence/arousal values for late fusion"""
        if not self.enable_valence_arousal:
            return None

        base_mapping = self.emotion_to_valence_arousal.get(
            emotion, {"valence": 0.0, "arousal": 0.0}
        )

        # Scale by confidence - less confident predictions move toward neutral
        confidence_factor = confidence  # Use confidence to modulate intensity

        valence = base_mapping["valence"] * confidence_factor
        arousal = base_mapping["arousal"] * confidence_factor

        return {
            "valence": valence,
            "arousal": arousal,
            "confidence": confidence,
            "method": "classifier_mapping",
        }

    def predict_valence_arousal_with_vllm(self, image):
        """Use VLM to directly predict valence and arousal"""
        if not self.use_vllm:
            return None

        prompt = """Analyze this facial expression and rate:
        1. Valence (emotional positivity): -1 (very negative) to +1 (very positive)
        2. Arousal (emotional intensity): -1 (very calm) to +1 (very excited)
        
        Respond in format: valence=X.X, arousal=Y.Y, confidence=Z.Z"""

        # This would be implemented based on your chosen VLM
        # For now, placeholder that falls back to classifier
        emotion_result = self.emotion_classifier(image)
        emotion = emotion_result[0]["label"].lower()
        confidence = emotion_result[0]["score"]

        return self.map_emotion_to_valence_arousal(emotion, confidence)

    def export_fusion_data(self, results, timestamp, bio_signals=None):
        """Export emotion data in format suitable for late fusion with bio-signals"""
        fusion_data = {
            "timestamp": timestamp,
            "visual_modality": {"faces_detected": len(results), "emotions": []},
        }

        # Add bio-signal data if provided
        if bio_signals:
            fusion_data["bio_modality"] = bio_signals

        # Process each detected face
        for i, result in enumerate(results):
            face_data = {
                "face_id": i,
                "emotion_discrete": result["emotion"],
                "confidence": result["confidence"],
                "position": result["position"],
            }

            # Add valence/arousal if available
            if "valence" in result and "arousal" in result:
                face_data.update(
                    {
                        "valence": result["valence"],
                        "arousal": result["arousal"],
                        "method": result.get("method", "classifier_mapping"),
                    }
                )

            fusion_data["visual_modality"]["emotions"].append(face_data)

        return fusion_data

    def late_fusion_example(self, visual_results, bio_signals):
        """Example of how to perform late fusion with bio-signals"""
        # This is a conceptual example - you'd implement your actual fusion logic

        # Extract visual features
        if visual_results:
            # Average across all detected faces
            avg_valence = np.mean([r.get("valence", 0) for r in visual_results])
            avg_arousal = np.mean([r.get("arousal", 0) for r in visual_results])
            avg_confidence = np.mean([r.get("confidence", 0) for r in visual_results])
        else:
            avg_valence = avg_arousal = avg_confidence = 0

        # Example bio-signal features (you'd extract these from your bio-signal ML model)
        bio_valence = bio_signals.get("predicted_valence", 0)
        bio_arousal = bio_signals.get("predicted_arousal", 0)
        bio_confidence = bio_signals.get("confidence", 0)

        # Late fusion - weighted combination
        visual_weight = avg_confidence
        bio_weight = bio_confidence
        total_weight = visual_weight + bio_weight

        if total_weight > 0:
            fused_valence = (
                avg_valence * visual_weight + bio_valence * bio_weight
            ) / total_weight
            fused_arousal = (
                avg_arousal * visual_weight + bio_arousal * bio_weight
            ) / total_weight
        else:
            fused_valence = fused_arousal = 0

        return {
            "fused_valence": fused_valence,
            "fused_arousal": fused_arousal,
            "visual_contribution": (
                visual_weight / total_weight if total_weight > 0 else 0
            ),
            "bio_contribution": bio_weight / total_weight if total_weight > 0 else 0,
            "fusion_confidence": total_weight / 2,  # Average of both confidences
        }


def list_available_cameras():
    """List all available camera devices"""
    available_cameras = []
    for i in range(10):  # Check first 10 camera indexes
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            ret, _ = cap.read()
            if ret:
                available_cameras.append(i)
            cap.release()
    return available_cameras


if __name__ == "__main__":
    detector = EmotionDetector()

    # List available cameras
    available_cameras = list_available_cameras()
    if not available_cameras:
        print("No cameras detected. Exiting.")
        exit()

    print("Available cameras:", available_cameras)

    # Ask user to select a camera
    if len(available_cameras) > 1:
        selected_camera = input(
            f"Select camera index {available_cameras} (default=0): "
        )
        try:
            selected_camera = int(selected_camera)
            if selected_camera not in available_cameras:
                print(f"Invalid camera index. Using default (0).")
                selected_camera = 0
        except ValueError:
            print("Invalid input. Using default camera (0).")
            selected_camera = 0
    else:
        selected_camera = available_cameras[0]
        print(f"Only one camera detected. Using camera {selected_camera}.")

    # Start the webcam with selected camera
    detector.start_webcam(selected_camera)

import cv2
import torch
from PIL import Image
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import pipeline
import time


class EmotionDetector:
    def __init__(self):
        # Initialize face detector
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )

        # Load a proper emotion recognition model
        print("Loading emotion recognition model...")
        try:
            # Option 1: Use a dedicated emotion recognition model
            self.emotion_classifier = pipeline(
                "image-classification", model="dima806/facial_emotions_image_detection"
            )

            # Remove the Gemma 3 model initialization
            self.model = None
            self.tokenizer = None
            self.vl_pipeline = None

        except Exception as e:
            print(f"Error loading model: {e}")
            raise

        print("Model loaded successfully!")

    def detect_emotions(self, frame):
        # Convert frame to grayscale for face detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect faces
        faces = self.face_cascade.detectMultiScale(gray, 1.1, 4)

        results = []
        for x, y, w, h in faces:
            # Extract face region
            face_img = frame[y : y + h, x : x + w]

            # Convert the face image to PIL format for the model
            pil_img = Image.fromarray(cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB))

            try:
                # Process with emotion classifier
                predictions = self.emotion_classifier(pil_img)
                # Get the top prediction
                emotion = predictions[0]["label"]
                confidence = predictions[0]["score"]

                # Format text to display
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

                results.append({"position": (x, y, w, h), "emotion": emotion})
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
                processed_frame, emotions = self.detect_emotions(frame)
                latest_results = emotions  # Store the latest results
                frame = processed_frame
                last_process_time = current_time
            else:
                # Draw the previously detected results on the current frame
                for result in latest_results:
                    x, y, w, h = result["position"]
                    emotion = result["emotion"]

                    # If we have confidence info (from the updated code)
                    if (
                        isinstance(emotion, dict)
                        and "label" in emotion
                        and "score" in emotion
                    ):
                        display_text = f"{emotion['label']} ({emotion['score']:.2f})"
                    elif hasattr(result, "confidence"):
                        display_text = f"{emotion} ({result['confidence']:.2f})"
                    else:
                        display_text = emotion

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

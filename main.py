import cv2
import dlib
import numpy as np
import os
import pickle
import threading

# Initialize dlib models
detector = dlib.get_frontal_face_detector()
shape_predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
face_recognizer = dlib.face_recognition_model_v1("dlib_face_recognition_resnet_model_v1.dat")

ENCODINGS_FILE = "encodings.pkl"

# Threaded video capture class
class VideoCaptureThreaded:
    def __init__(self, src=0):
        self.cap = cv2.VideoCapture(src)
        self.ret = False
        self.frame = None
        self.running = True
        self.thread = threading.Thread(target=self.update)
        self.thread.start()

    def update(self):
        while self.running:
            if self.cap.isOpened():
                self.ret, self.frame = self.cap.read()
            else:
                self.ret = False

    def read(self):
        return self.ret, self.frame

    def release(self):
        self.running = False
        self.thread.join()
        self.cap.release()

# Functions for face encoding
def compute_face_encodings(image):
    detections = detector(image, 1)
    encodings = []
    for detection in detections:
        try:
            shape = shape_predictor(image, detection)
            encoding = np.array(face_recognizer.compute_face_descriptor(image, shape))
            encodings.append((detection, encoding))
        except Exception as e:
            print(f"Error processing face: {e}")
    return encodings

def load_encodings(file_path):
    with open(file_path, 'rb') as file:
        return pickle.load(file)

# Precompute encodings if not done yet
if not os.path.exists(ENCODINGS_FILE):
    print(f"Encodings file {ENCODINGS_FILE} is missing.")
    exit(1)

# Load encodings
known_face_encodings, known_face_names = load_encodings(ENCODINGS_FILE)

# Initialize video capture
video_capture = VideoCaptureThreaded()
frame_counter = 0
skip_rate = 5  # Process every nth frame
frame_resize_factor = 0.4  # Resize factor for faster processing
tracked_faces = []  # Store tracked faces

# Helper function to check bounding box overlap
def is_overlapping(box1, box2):
    x1, y1, x2, y2 = box1
    x3, y3, x4, y4 = box2

    # Calculate overlap area
    dx = min(x2, x4) - max(x1, x3)
    dy = min(y2, y4) - max(y1, y3)

    if dx >= 0 and dy >= 0:
        return True  # Overlap exists
    return False

# Main loop
try:
    while True:
        ret, frame = video_capture.read()
        if not ret or frame is None:
            print("Failed to capture frame. Retrying...")
            continue

        # Resize frame for faster processing
        small_frame = cv2.resize(frame, (0, 0), fx=frame_resize_factor, fy=frame_resize_factor)
        rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

        # Process every nth frame
        if frame_counter % skip_rate == 0:
            # Detect faces
            detected_faces = compute_face_encodings(rgb_small_frame)

            # Update tracked faces
            for detection, face_encoding in detected_faces:
                # Scale box back to original frame size
                left, top, right, bottom = (
                    int(detection.left() / frame_resize_factor),
                    int(detection.top() / frame_resize_factor),
                    int(detection.right() / frame_resize_factor),
                    int(detection.bottom() / frame_resize_factor),
                )

                # Check if this face overlaps with any tracked face
                overlapping = any(is_overlapping((left, top, right, bottom), face["box"]) for face in tracked_faces)

                if not overlapping:
                    # Find the best match for this new face
                    distances = np.linalg.norm(known_face_encodings - face_encoding, axis=1)
                    min_distance_index = np.argmin(distances)
                    name = "Unknown"
                    color = (0, 0, 255)  # Red for unknown

                    if distances[min_distance_index] < 0.6:  # Adjust threshold
                        name = known_face_names[min_distance_index]
                        color = (0, 255, 0)  # Green for recognized

                    tracked_faces.append({
                        "box": (left, top, right, bottom),
                        "name": name,
                        "color": color,
                        "frame_life": 10  # Frames to persist tracking
                    })

        # Update face tracking
        for face in tracked_faces:
            face["frame_life"] -= 1
            left, top, right, bottom = face["box"]
            cv2.rectangle(frame, (left, top), (right, bottom), face["color"], 2)
            cv2.putText(frame, face["name"], (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, face["color"], 1)

        # Remove expired faces
        tracked_faces = [face for face in tracked_faces if face["frame_life"] > 0]

        # Display the resulting frame
        cv2.imshow("Video", frame)
        frame_counter += 1

        # Quit with 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

except KeyboardInterrupt:
    print("Exiting program...")

finally:
    # Cleanup
    video_capture.release()
    cv2.destroyAllWindows()

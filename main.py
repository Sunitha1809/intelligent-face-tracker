
import cv2
from ultralytics import YOLO

# -------------------------------
# 1. Load the face detection model
# -------------------------------
print("Loading YOLO face model...")
model = YOLO("models/yolov8n-face.pt")
print("Face model loaded successfully!")

# -------------------------------
# 2. Open the video
# -------------------------------
video_path = "data/sample.mp4"
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print(f"ERROR: Could not open video: {video_path}")
    raise SystemExit(1)

total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
fps = cap.get(cv2.CAP_PROP_FPS)

print(f"Video opened successfully! Frames: {total_frames}, FPS: {fps:.2f}")
print("Face tracking started. Press Q in the video window to quit.")

frame_number = 0

# -------------------------------
# 3. Detect and track faces
# -------------------------------
while cap.isOpened():
    ret, frame = cap.read()

    if not ret:
        print("Reached the end of the video.")
        break

    frame_number += 1

    # Track faces between consecutive frames
    results = model.track(
        source=frame,
        persist=True,
        tracker="bytetrack.yaml",
        conf=0.25,
        verbose=False
    )

    face_count = 0
    track_ids = []

    for result in results:
        if result.boxes is None:
            continue

        for box in result.boxes:
            x1, y1, x2, y2 = map(
                int, box.xyxy[0].tolist()
            )
            confidence = float(box.conf[0])
            face_count += 1

            # Read the temporary tracking ID
            if box.id is not None:
                track_id = int(box.id[0].item())
                track_ids.append(track_id)
                label = f"Person {track_id} ({confidence:.2f})"
            else:
                label = f"Face ({confidence:.2f})"

            # Draw a green box around the face
            cv2.rectangle(
                frame,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                2
            )

            # Draw the label above the box
            cv2.putText(
                frame,
                label,
                (x1, max(y1 - 10, 25)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2
            )

    # Show status on the video
    cv2.putText(
        frame,
        f"Frame: {frame_number}/{total_frames} | Faces: {face_count}",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 255),
        2
    )

    cv2.imshow("Intelligent Face Tracker", frame)

    # Print progress every 100 frames
    if frame_number % 100 == 0:
        print(
            f"Frame {frame_number}/{total_frames} | "
            f"Faces: {face_count} | Tracking IDs: {track_ids}"
        )

    # Wait briefly and check whether Q was pressed
    if cv2.waitKey(1) & 0xFF == ord("q"):
        print("Quit requested by user.")
        break

# -------------------------------
# 4. Release resources
# -------------------------------
cap.release()
cv2.destroyAllWindows()

print(f"Tracking finished. Total frames processed: {frame_number}")

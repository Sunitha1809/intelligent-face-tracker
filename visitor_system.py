
import cv2
import json
import sqlite3
import numpy as np
from pathlib import Path
from datetime import datetime
from ultralytics import YOLO
from insightface.app import FaceAnalysis

# -------------------------------
# Project paths and configuration
# -------------------------------
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "yolov8n-face.pt"
VIDEO_PATH = BASE_DIR / "data" / "sample.mp4"
DB_PATH = BASE_DIR / "visitors.db"
FACES_DIR = BASE_DIR / "logs" / "registered_faces"
CONFIG_PATH = BASE_DIR / "config.json"

FACES_DIR.mkdir(parents=True, exist_ok=True)

# Similarity threshold: higher means stricter matching
SIMILARITY_THRESHOLD = 0.45

# -------------------------------
# Load configuration
# -------------------------------
default_config = {"frame_skip": 0}

if CONFIG_PATH.exists():
    with open(CONFIG_PATH, "r", encoding="utf-8") as file:
        config = json.load(file)
else:
    config = default_config.copy()
    with open(CONFIG_PATH, "w", encoding="utf-8") as file:
        json.dump(config, file, indent=4)

FRAME_SKIP = max(0, int(config.get("frame_skip", 0)))

# -------------------------------
# Initialize SQLite database
# -------------------------------
connection = sqlite3.connect(DB_PATH)
connection.execute("""
    CREATE TABLE IF NOT EXISTS visitors (
        visitor_id INTEGER PRIMARY KEY AUTOINCREMENT,
        embedding TEXT NOT NULL,
        image_path TEXT NOT NULL,
        first_seen TEXT NOT NULL
    )
""")
connection.commit()

# -------------------------------
# Load AI models
# -------------------------------
print("Loading YOLO face detector...")
detector = YOLO(str(MODEL_PATH))

print("Loading InsightFace recognition model...")
recognizer = FaceAnalysis(name="buffalo_l")
recognizer.prepare(ctx_id=-1, det_size=(640, 640))

print("Models loaded successfully!")

# -------------------------------
# Visitor recognition helpers
# -------------------------------
def get_registered_visitors():
    rows = connection.execute(
        "SELECT visitor_id, embedding FROM visitors"
    ).fetchall()

    visitors = []
    for visitor_id, embedding_json in rows:
        embedding = np.array(
            json.loads(embedding_json), dtype=np.float32
        )
        visitors.append((visitor_id, embedding))

    return visitors


def match_visitor(embedding):
    best_id = None
    best_similarity = -1.0

    # Normalize embeddings before comparing them
    embedding = embedding / (np.linalg.norm(embedding) + 1e-10)

    for visitor_id, known_embedding in get_registered_visitors():
        known_embedding = known_embedding / (
            np.linalg.norm(known_embedding) + 1e-10
        )
        similarity = float(np.dot(embedding, known_embedding))

        if similarity > best_similarity:
            best_similarity = similarity
            best_id = visitor_id

    if best_id is not None and best_similarity >= SIMILARITY_THRESHOLD:
        return best_id, best_similarity

    return None, best_similarity


def register_visitor(embedding, face_crop):
    now = datetime.now()
    timestamp = now.strftime("%Y-%m-%d_%H-%M-%S-%f")
    image_path = FACES_DIR / f"visitor_{timestamp}.jpg"

    if not cv2.imwrite(str(image_path), face_crop):
        raise RuntimeError(f"Could not save face image: {image_path}")

    normalized = embedding / (np.linalg.norm(embedding) + 1e-10)

    cursor = connection.execute(
        """
        INSERT INTO visitors (embedding, image_path, first_seen)
        VALUES (?, ?, ?)
        """,
        (
            json.dumps(normalized.tolist()),
            str(image_path.relative_to(BASE_DIR)),
            now.isoformat(timespec="seconds"),
        ),
    )
    connection.commit()

    visitor_id = cursor.lastrowid
    print(f"Registered visitor V{visitor_id}")
    return visitor_id


# -------------------------------
# Open input video
# -------------------------------
cap = cv2.VideoCapture(str(VIDEO_PATH))

if not cap.isOpened():
    connection.close()
    raise SystemExit(f"ERROR: Cannot open video: {VIDEO_PATH}")

total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
frame_number = 0

# Keep a mapping from temporary track IDs to visitor IDs
track_to_visitor = {}

print(f"Video opened: {total_frames} frames")
print("Visitor recognition started. Press Q to quit.")

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Reached end of video.")
            break

        frame_number += 1

        # Process every (FRAME_SKIP + 1) frames
        if FRAME_SKIP and (frame_number - 1) % (FRAME_SKIP + 1) != 0:
            cv2.imshow("Intelligent Visitor System", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
            continue

        # YOLO detects and tracks faces
        results = detector.track(
            frame,
            persist=True,
            tracker="bytetrack.yaml",
            conf=0.25,
            verbose=False,
        )

        # InsightFace produces recognition embeddings
        detected_faces = recognizer.get(frame)
        assigned_in_frame = set()

        for result in results:
            if result.boxes is None:
                continue

            for box in result.boxes:
                x1, y1, x2, y2 = map(
                    int, box.xyxy[0].tolist()
                )
                confidence = float(box.conf[0])

                x1 = max(0, x1)
                y1 = max(0, y1)
                x2 = min(frame.shape[1], x2)
                y2 = min(frame.shape[0], y2)

                if x2 <= x1 or y2 <= y1:
                    continue

                # Match YOLO box to the nearest InsightFace detection
                box_center = np.array(
                    [(x1 + x2) / 2, (y1 + y2) / 2]
                )
                best_face = None
                best_distance = float("inf")

                for face in detected_faces:
                    fx1, fy1, fx2, fy2 = face.bbox
                    face_center = np.array(
                        [(fx1 + fx2) / 2, (fy1 + fy2) / 2]
                    )
                    distance = float(
                        np.linalg.norm(box_center - face_center)
                    )
                    if distance < best_distance:
                        best_distance = distance
                        best_face = face

                # Require the InsightFace detection to be nearby
                if best_face is None or best_distance > max(
                    x2 - x1, y2 - y1
                ) * 0.75:
                    label = f"Detecting ({confidence:.2f})"
                    cv2.rectangle(
                        frame, (x1, y1), (x2, y2), (0, 165, 255), 2
                    )
                    cv2.putText(
                        frame, label, (x1, max(y1 - 10, 20)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                        (0, 165, 255), 2
                    )
                    continue

                embedding = best_face.embedding.astype(np.float32)

                # Get temporary tracker ID, if available
                track_id = None
                if box.id is not None:
                    track_id = int(box.id[0].item())

                visitor_id = (
                    track_to_visitor.get(track_id)
                    if track_id is not None else None
                )

                # Verify the track's identity against stored embeddings
                matched_id, similarity = match_visitor(embedding)

                if matched_id is not None:
                    visitor_id = matched_id
                else:
                    # Register only once per current track in this run
                    if (
                        track_id is not None
                        and track_id in track_to_visitor
                    ):
                        visitor_id = track_to_visitor[track_id]
                    else:
                        face_crop = frame[y1:y2, x1:x2]
                        if face_crop.size == 0:
                            continue
                        visitor_id = register_visitor(
                            embedding, face_crop
                        )

                if track_id is not None:
                    track_to_visitor[track_id] = visitor_id

                assigned_in_frame.add(visitor_id)

                label = f"Visitor V{visitor_id} ({confidence:.2f})"
                cv2.rectangle(
                    frame, (x1, y1), (x2, y2), (0, 255, 0), 2
                )
                cv2.putText(
                    frame, label, (x1, max(y1 - 10, 20)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                    (0, 255, 0), 2
                )

        unique_visitors = connection.execute(
            "SELECT COUNT(*) FROM visitors"
        ).fetchone()[0]

        cv2.putText(
            frame,
            f"Unique registered visitors: {unique_visitors}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 255),
            2,
        )

        cv2.imshow("Intelligent Visitor System", frame)

        if frame_number % 100 == 0:
            print(
                f"Frame {frame_number}/{total_frames} | "
                f"Registered visitors: {unique_visitors}"
            )

        if cv2.waitKey(1) & 0xFF == ord("q"):
            print("Quit requested.")
            break

finally:
    cap.release()
    cv2.destroyAllWindows()
    connection.close()
    print(f"Finished. Frames processed: {frame_number}")

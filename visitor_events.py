
import cv2
import json
import logging
import sqlite3
import time
from datetime import datetime
from pathlib import Path

import numpy as np
from ultralytics import YOLO
from insightface.app import FaceAnalysis


# -------------------- PROJECT SETTINGS --------------------

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "yolov8n-face.pt"
VIDEO_PATH = BASE_DIR / "data" / "sample.mp4"
DB_PATH = BASE_DIR / "visitors.db"
CONFIG_PATH = BASE_DIR / "config.json"

LOG_DIR = BASE_DIR / "logs"
ENTRY_DIR = LOG_DIR / "entries"
EXIT_DIR = LOG_DIR / "exits"
REGISTERED_DIR = LOG_DIR / "registered_faces"
EVENT_LOG = LOG_DIR / "events.log"

SIMILARITY_THRESHOLD = 0.45
MATCH_DISTANCE = 120
LINE_POSITION_RATIO = 0.50


# -------------------- CREATE FOLDERS AND LOGGING --------------------

for folder in (LOG_DIR, ENTRY_DIR, EXIT_DIR, REGISTERED_DIR):
    folder.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    filename=str(EVENT_LOG),
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    force=True,
)

def log_event(message):
    logging.info(message)
    print(message)


# -------------------- LOAD CONFIGURATION --------------------

if not CONFIG_PATH.exists():
    CONFIG_PATH.write_text(
        json.dumps({"frame_skip": 0}, indent=4),
        encoding="utf-8",
    )

try:
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    FRAME_SKIP = max(0, int(config.get("frame_skip", 0)))
except (ValueError, TypeError, json.JSONDecodeError):
    FRAME_SKIP = 0

log_event(f"SYSTEM_START | frame_skip={FRAME_SKIP}")


# -------------------- DATABASE --------------------

def connect_db():
    connection = sqlite3.connect(str(DB_PATH))
    connection.execute("""
        CREATE TABLE IF NOT EXISTS visitors (
            visitor_id INTEGER PRIMARY KEY AUTOINCREMENT,
            embedding TEXT,
            image_path TEXT,
            first_seen TEXT
        )
    """)
    connection.execute("""
        CREATE TABLE IF NOT EXISTS visitor_events (
            event_id INTEGER PRIMARY KEY AUTOINCREMENT,
            visitor_id INTEGER NOT NULL,
            event_type TEXT NOT NULL,
            event_time TEXT NOT NULL,
            image_path TEXT,
            track_id TEXT
        )
    """)
    connection.commit()
    return connection


db = connect_db()


# -------------------- FACE RECOGNITION --------------------

def cosine_similarity(a, b):
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    denominator = np.linalg.norm(a) * np.linalg.norm(b)

    if denominator == 0:
        return -1.0

    return float(np.dot(a, b) / denominator)


def find_or_register_visitor(embedding, face_crop):
    rows = db.execute(
        "SELECT visitor_id, embedding FROM visitors"
    ).fetchall()

    best_id = None
    best_score = -1.0

    for visitor_id, saved_embedding in rows:
        if not saved_embedding:
            continue

        try:
            saved = json.loads(saved_embedding)
            score = cosine_similarity(embedding, saved)
        except (ValueError, TypeError, json.JSONDecodeError):
            continue

        if score > best_score:
            best_score = score
            best_id = visitor_id

    if best_id is not None and best_score >= SIMILARITY_THRESHOLD:
        log_event(
            f"RECOGNITION | visitor=V{best_id} | "
            f"similarity={best_score:.3f}"
        )
        return best_id

    now = datetime.now().astimezone().isoformat(timespec="seconds")
    cursor = db.execute(
        """
        INSERT INTO visitors (embedding, image_path, first_seen)
        VALUES (?, ?, ?)
        """,
        (json.dumps(np.asarray(embedding).tolist()), "", now),
    )
    visitor_id = cursor.lastrowid

    image_path = REGISTERED_DIR / f"visitor_{visitor_id}.jpg"
    if face_crop is not None and face_crop.size:
        cv2.imwrite(str(image_path), face_crop)

    db.execute(
        "UPDATE visitors SET image_path = ? WHERE visitor_id = ?",
        (str(image_path.relative_to(BASE_DIR)), visitor_id),
    )
    db.commit()

    log_event(
        f"REGISTRATION | visitor=V{visitor_id} | "
        f"image={image_path.relative_to(BASE_DIR)}"
    )
    return visitor_id


# -------------------- SAVE ENTRY/EXIT EVENTS --------------------

def save_event(visitor_id, event_type, track_id, face_crop):
    now = datetime.now().astimezone()
    timestamp = now.isoformat(timespec="seconds")
    date_folder = now.strftime("%Y-%m-%d")

    if event_type == "ENTRY":
        folder = ENTRY_DIR / date_folder
    else:
        folder = EXIT_DIR / date_folder

    folder.mkdir(parents=True, exist_ok=True)

    filename = (
        f"{now.strftime('%H%M%S_%f')}_"
        f"V{visitor_id}_{event_type.lower()}.jpg"
    )
    image_path = folder / filename

    saved_path = ""
    if face_crop is not None and face_crop.size:
        if cv2.imwrite(str(image_path), face_crop):
            saved_path = str(image_path.relative_to(BASE_DIR))

    db.execute(
        """
        INSERT INTO visitor_events
            (visitor_id, event_type, event_time, image_path, track_id)
        VALUES (?, ?, ?, ?, ?)
        """,
        (visitor_id, event_type, timestamp, saved_path, str(track_id)),
    )
    db.commit()

    log_event(
        f"{event_type} | visitor=V{visitor_id} | "
        f"track={track_id} | time={timestamp} | image={saved_path}"
    )


# -------------------- LOAD MODELS --------------------

if not MODEL_PATH.exists():
    raise FileNotFoundError(f"YOLO model not found: {MODEL_PATH}")

if not VIDEO_PATH.exists():
    raise FileNotFoundError(f"Sample video not found: {VIDEO_PATH}")

log_event("MODEL_LOAD | Loading YOLO model")
detector = YOLO(str(MODEL_PATH))

log_event("MODEL_LOAD | Loading InsightFace buffalo_l on CPU")
recognizer = FaceAnalysis(name="buffalo_l")
recognizer.prepare(ctx_id=-1, det_size=(640, 640))

# Maps temporary tracker IDs to visitor database IDs.
track_to_visitor = {}
previous_positions = {}
last_event_direction = {}
frame_number = 0
processed_frames = 0


# -------------------- PROCESS VIDEO --------------------

cap = cv2.VideoCapture(str(VIDEO_PATH))

if not cap.isOpened():
    db.close()
    raise RuntimeError(f"Could not open video: {VIDEO_PATH}")

video_fps = cap.get(cv2.CAP_PROP_FPS)
if not video_fps or video_fps <= 0:
    video_fps = 25.0

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
line_x = int(width * LINE_POSITION_RATIO)

log_event(
    f"VIDEO_START | width={width} | height={height} | "
    f"fps={video_fps:.2f} | virtual_line_x={line_x}"
)

print("\nStarting visitor event tracking.")
print("Move across the vertical line to demonstrate ENTRY or EXIT.")
print("Press Q in the video window to stop early.\n")

try:
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        frame_number += 1

        # Keep the last known tracking position on skipped frames.
        if FRAME_SKIP > 0 and frame_number % (FRAME_SKIP + 1) != 1:
            continue

        processed_frames += 1

        # YOLO detection and ByteTrack tracking.
        result = detector.track(
            frame,
            persist=True,
            tracker="bytetrack.yaml",
            conf=0.25,
            verbose=False,
        )[0]

        # InsightFace supplies face embeddings.
        detected_faces = recognizer.get(frame)

        face_info = []
        for face in detected_faces:
            box = face.bbox.astype(int)
            x1, y1, x2, y2 = box.tolist()
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(width, x2), min(height, y2)

            if x2 <= x1 or y2 <= y1:
                continue

            crop = frame[y1:y2, x1:x2].copy()
            center = ((x1 + x2) // 2, (y1 + y2) // 2)
            face_info.append((center, face.embedding, crop, box))

        boxes = result.boxes
        current_tracks = []

        if boxes is not None and boxes.id is not None:
            coordinates = boxes.xyxy.cpu().numpy().astype(int)
            track_ids = boxes.id.cpu().numpy().astype(int)

            for coords, track_id in zip(coordinates, track_ids):
                x1, y1, x2, y2 = coords.tolist()
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(width, x2), min(height, y2)

                if x2 <= x1 or y2 <= y1:
                    continue

                center = ((x1 + x2) // 2, (y1 + y2) // 2)
                track_crop = frame[y1:y2, x1:x2].copy()

                # Associate the YOLO track with the nearest InsightFace face.
                best_face = None
                best_distance = float("inf")

                for info in face_info:
                    face_center = info[0]
                    distance = (
                        (center[0] - face_center[0]) ** 2
                        + (center[1] - face_center[1]) ** 2
                    ) ** 0.5

                    if distance < best_distance:
                        best_distance = distance
                        best_face = info

                visitor_id = track_to_visitor.get(int(track_id))

                if best_face is not None and best_distance <= MATCH_DISTANCE:
                    embedding = best_face[1]
                    face_crop = best_face[2]

                    if visitor_id is None:
                        log_event(
                            f"EMBEDDING | track={track_id} | "
                            f"dimensions={len(embedding)}"
                        )
                        visitor_id = find_or_register_visitor(
                            embedding, face_crop
                        )
                        track_to_visitor[int(track_id)] = visitor_id

                if visitor_id is None:
                    # Wait until a usable face embedding is available.
                    label = f"Track {track_id} | face pending"
                else:
                    label = f"Visitor V{visitor_id} | Track {track_id}"

                    # Detect crossing of the virtual vertical boundary.
                    previous_x = previous_positions.get(int(track_id))
                    current_x = int(center[0])
                    direction = None

                    if previous_x is not None:
                        if previous_x < line_x <= current_x:
                            direction = "ENTRY"
                        elif previous_x >= line_x > current_x:
                            direction = "EXIT"

                    debug_message = (
                        f"DEBUG_CROSSING | track={track_id} | "
                        f"visitor={visitor_id} | previous_x={previous_x} | "
                        f"current_x={current_x} | line_x={line_x} | "
                        f"direction={direction}"
                    )
                    log_event(debug_message)
                    print(debug_message)

                    if direction is not None:
                        previous_direction = last_event_direction.get(int(track_id))
                        if previous_direction != direction:
                            save_event(
                                visitor_id,
                                direction,
                                int(track_id),
                                track_crop,
                            )
                            last_event_direction[int(track_id)] = direction

                    previous_positions[int(track_id)] = current_x

                current_tracks.append((coords, label))

                log_event(
                    f"TRACKING | track={track_id} | "
                    f"visitor={visitor_id if visitor_id is not None else 'pending'}"
                )

        # Draw the virtual line and tracking labels.
        cv2.line(
            frame, (line_x, 0), (line_x, height), (255, 200, 0), 2
        )

        for coords, label in current_tracks:
            x1, y1, x2, y2 = coords.tolist()
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(
                frame,
                label,
                (x1, max(25, y1 - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                2,
            )

        unique_visitors = db.execute(
            "SELECT COUNT(*) FROM visitors"
        ).fetchone()[0]

        entry_count = db.execute(
            "SELECT COUNT(*) FROM visitor_events WHERE event_type='ENTRY'"
        ).fetchone()[0]

        exit_count = db.execute(
            "SELECT COUNT(*) FROM visitor_events WHERE event_type='EXIT'"
        ).fetchone()[0]

        cv2.putText(
            frame,
            f"Visitors registered: {unique_visitors}",
            (10, 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 255, 255),
            2,
        )
        cv2.putText(
            frame,
            f"Entries: {entry_count}  Exits: {exit_count}",
            (10, 52),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 255, 255),
            2,
        )

        cv2.imshow("Intelligent Face Tracker - Visitor Events", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            log_event("VIDEO_STOP | stopped by user")
            break

finally:
    cap.release()
    cv2.destroyAllWindows()

    log_event(
        f"VIDEO_FINISHED | frames_read={frame_number} | "
        f"frames_processed={processed_frames}"
    )

    final_visitors = db.execute(
        "SELECT COUNT(*) FROM visitors"
    ).fetchone()[0]
    final_entries = db.execute(
        "SELECT COUNT(*) FROM visitor_events WHERE event_type='ENTRY'"
    ).fetchone()[0]
    final_exits = db.execute(
        "SELECT COUNT(*) FROM visitor_events WHERE event_type='EXIT'"
    ).fetchone()[0]

    print("\n---------- FINAL SUMMARY ----------")
    print(f"Registered visitor records: {final_visitors}")
    print(f"Entry events: {final_entries}")
    print(f"Exit events: {final_exits}")
    print(f"Event log: {EVENT_LOG}")
    print(f"Database: {DB_PATH}")
    print("-----------------------------------")

    db.close()

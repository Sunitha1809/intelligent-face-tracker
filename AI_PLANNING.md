# AI Planning and Implementation Notes

## Project objective

Build a video-based face-tracking prototype that associates detected faces with persistent visitor records and records entry/exit events when tracks cross a virtual line.

## Planned processing flow

1. Read video frames with OpenCV.
2. Run YOLO face detection on the configured processing schedule.
3. Track detections across frames using ByteTrack.
4. Generate face embeddings with InsightFace/ArcFace for usable face crops.
5. Compare each embedding with stored visitor embeddings using cosine similarity.
6. Reuse a stored visitor ID when the best match meets the configured threshold; otherwise create a new visitor record.
7. Track each object's position relative to a virtual line.
8. When a track crosses the line, classify the direction as ENTRY or EXIT.
9. Save an event image, timestamp, visitor ID, event type, and track ID in SQLite and write diagnostic events to the log.
10. Display the current video and counters.

## Components and responsibilities

- `main.py`: YOLO-based detection and tracking visualization.
- `visitor_events.py`: visitor matching/registration, event persistence, and entry/exit crossing logic.
- `visitor_system.py`: an additional visitor-system implementation retained in the repository.
- `config.json`: runtime configuration, including frame skipping.
- `visitors.db`: persistent visitor and event records.
- `logs/`: event logs and saved face crops.
- `models/yolov8n-face.pt`: local YOLO face-detection weights.

## Identity and tracking assumptions

- A tracker ID is temporary and may change when tracking restarts or a person is lost and reacquired.
- A visitor ID is the persistent database identity assigned through embedding matching.
- Embedding similarity is an approximate biometric match, not proof of identity.
- The similarity threshold must be tested against representative footage to understand false matches and missed matches.
- The virtual line and crossing direction determine entry/exit classification.
- A person who never crosses the line in the tested direction will not produce that crossing event.

## Database design

The current visitor database contains a `visitors` table for registered visitor embeddings, image paths, and first-seen timestamps, and a `visitor_events` table for event type, timestamp, image path, visitor ID, and track ID.

## Testing plan

- Verify Python syntax for the application scripts.
- Confirm the local YOLO model can load.
- Confirm InsightFace can generate a 512-dimensional embedding.
- Run the sample video and inspect the generated event log.
- Inspect SQLite event counts and recent rows.
- Test left-to-right and right-to-left line crossings using footage or controlled test cases.
- Check that repeated frames do not create duplicate events for the same crossing.
- Review saved face crops and check for false visitor matches.
- Record limitations honestly in the final submission.

## Current known validation evidence

The development checks performed so far indicate that the scripts compile, the YOLO and InsightFace components have been exercised, an embedding log with 512 dimensions has been observed, and SQLite contains registered visitor records and ENTRY events.

The sample-video run has not yet demonstrated an EXIT event. This should be treated as an unverified behavior for the available footage, not as proof that exit detection works end to end.

## Responsible use

Face embeddings and face crops are sensitive biometric information. Obtain appropriate consent, limit access, protect stored data, and define a retention policy before any real-world deployment.

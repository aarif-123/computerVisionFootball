import cv2
import json
from ultralytics import YOLO
from player_utils import get_feature_extractor, extract_feature, assign_id

# Load YOLOv11 detection model
yolo_model = YOLO("yolov11.pt")

# Load feature extractor model
resnet = get_feature_extractor()

# Input and output video
cap = cv2.VideoCapture("15sec_input_720p.mp4")
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)
out = cv2.VideoWriter("output.avi", cv2.VideoWriter_fourcc(*"XVID"), fps, (width, height))

# Tracking data
players = {}  # {id: {feature, last_seen}}
player_id_counter = 0
frame_no = 0
log = []  # For optional output_log.json

while True:
    ret, frame = cap.read()
    if not ret:
        break

    detections = yolo_model(frame)[0].boxes.xyxy.cpu().numpy()
    used_ids = set()
    frame_data = []

    for det in detections:
        x1, y1, x2, y2 = map(int, det[:4])
        crop = frame[y1:y2, x1:x2]
        feature = extract_feature(crop, resnet)
        if feature is None:
            continue

        pid = assign_id(feature, players, used_ids, frame_no)

        if pid is None:
            player_id_counter += 1
            pid = player_id_counter
            players[pid] = {'feature': feature, 'last_seen': frame_no}

        used_ids.add(pid)

        # Draw and log
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f"ID: {pid}", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        frame_data.append({'id': pid, 'bbox': [int(x1), int(y1), int(x2), int(y2)]})

    log.append({'frame': frame_no, 'players': frame_data})
    out.write(frame)
    frame_no += 1

cap.release()
out.release()
cv2.destroyAllWindows()

# Save ID assignment log
with open("output_log.json", "w") as f:
    json.dump(log, f, indent=2)

print("✅ Output video saved as: output.avi")
print("✅ Player tracking log saved as: output_log.json")

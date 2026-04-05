from ultralytics import YOLO
import cv2
import numpy as np
import random

# Load YOLO model
model = YOLO("yolov8n.pt")

# Open video
cap = cv2.VideoCapture("football.mp4")

# Store previous ball position
prev_ball_pos = None
event_text = "Starting match..."

# Output video setup
width = int(cap.get(3))
height = int(cap.get(4))

out = cv2.VideoWriter(
    "output.mp4",
    cv2.VideoWriter_fourcc(*'mp4v'),
    30,
    (width, height)
)

# 🎙️ Commentary generator
def generate_commentary(event):
    if event == "pass":
        return random.choice([
            "Nice pass between players!",
            "Quick ball movement!",
            "Smooth passing play!",
            "Great teamwork on display!"
        ])
    
    elif event == "shot":
        return random.choice([
            "Shot taken with power!",
            "What a strike!",
            "A strong attempt on goal!",
            "Goal opportunity created!"
        ])
    
    else:
        return random.choice([
            "Game is in progress...",
            "Players are positioning themselves...",
            "Ball is under control...",
            "Action continues on the field..."
        ])

# Process video
while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model.track(frame, persist=True)

    ball_pos = None

    # Detect ball
    for r in results:
        boxes = r.boxes
        for box in boxes:
            cls = int(box.cls[0])

            # COCO class 32 = sports ball
            if cls == 32:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                ball_pos = ((x1 + x2)//2, (y1 + y2)//2)

    # 🎯 Event detection logic
    if prev_ball_pos is not None and ball_pos is not None:
        dx = ball_pos[0] - prev_ball_pos[0]
        dy = ball_pos[1] - prev_ball_pos[1]
        speed = np.sqrt(dx*dx + dy*dy)

        if speed > 30:
            event_text = generate_commentary("shot")
        elif speed > 5:
            event_text = generate_commentary("pass")
        else:
            event_text = generate_commentary("normal")

    prev_ball_pos = ball_pos

    # Draw detections
    annotated = results[0].plot()

    # Add commentary text on video
    cv2.putText(
        annotated,
        event_text,
        (50, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2
    )

    # Save output
    out.write(annotated)

    # Show video
    cv2.imshow("Football Analysis", annotated)

    if cv2.waitKey(1) & 0xFF == 27:
        break

# Release everything
cap.release()
out.release()
cv2.destroyAllWindows()
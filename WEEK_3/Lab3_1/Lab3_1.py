"""
YOLOv8 Person Detection in Video
Detects and counts people in video frames using YOLOv8 nano model
"""
import cv2
import time
from ultralytics import YOLO

# ============================================================================
# CONFIGURATION
# ============================================================================
VIDEO_PATH = r"C:\Users\HP\Documents\EY-ASSIGNMENTS\WEEK_3\Lab3_1\street.mp4"
OUTPUT_PATH = "output.mp4"

# ============================================================================
# INITIALIZE MODEL AND VIDEO
# ============================================================================
# Load YOLOv8 nano model for person detection
model = YOLO("yolov8n.pt")

# Open video file for processing
cap = cv2.VideoCapture(VIDEO_PATH, cv2.CAP_FFMPEG)

if not cap.isOpened():
    print("Error: Could not open video file.")
    exit()

# Extract video properties for output
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
fps_video = cap.get(cv2.CAP_PROP_FPS)

# Initialize video writer with same properties as input
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(OUTPUT_PATH, fourcc, fps_video,
                      (frame_width, frame_height))

prev_time = 0

# ============================================================================
# PROCESS VIDEO FRAMES
# ============================================================================
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Resize frame for faster inference while maintaining aspect
    frame = cv2.resize(frame, (640, 480))

    # Run YOLOv8 detection
    results = model(frame, verbose=False)
    person_count = 0

    # Process detections - extract person bounding boxes
    for box in results[0].boxes:
        cls = int(box.cls[0])
        
        # COCO dataset class 0 represents 'person'
        if cls == 0:
            person_count += 1
            
            # Extract bounding box coordinates and confidence
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            
            # Draw bounding box (green rectangle)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            # Draw confidence label above box
            cv2.putText(frame, f"Person {conf:.2f}",
                        (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (0, 255, 0), 2)

    # Calculate frames per second
    current_time = time.time()
    fps = 1 / (current_time - prev_time) if (current_time - prev_time) > 0 else 0
    prev_time = current_time

    # Display person count on frame (red text)
    cv2.putText(frame, f"People Count: {person_count}",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1, (0, 0, 255), 3)

    # Display FPS on frame (blue text)
    cv2.putText(frame, f"FPS: {fps:.2f}",
                (20, 80),
                cv2.FONT_HERSHEY_SIMPLEX,
                1, (255, 0, 0), 3)

    # Display processed frame in window
    cv2.imshow("YOLOv8 Person Detection", frame)

    # Write frame to output video file
    out.write(frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()

print("Processing complete.")
print("Saved output to:", OUTPUT_PATH)
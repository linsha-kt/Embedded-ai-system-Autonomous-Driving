main.py
import cv2
import torch
from ultralytics import YOLO
from picamera2 import Picamera2

picam = Picamera2()
picam.preview_configuration.main.size = (480, 480)
picam.preview_configuration.main.format = "RGB888"
picam.preview_configuration.main.align()
picam.configure("preview")
picam.start()

# Load two YOLOv8 models
model1 = YOLO("/home/pi/Autonomous_Driving/runs1/detect/train/weights/last.pt")
model2 = YOLO("/home/pi/Autonomous_Driving/runs2/detect/train/weights/last.pt")


threshold1 = 0.8
threshold2 = 0.4

while True:
    frame = picam.capture_array()

    # Run inference with both models
    results1 = model1(frame)[0]
    results2 = model2(frame)[0]

    for result in results1.boxes.data.tolist():
        x1, y1, x2, y2, score, class_id = result

        if score > threshold1:
            # Draw bounding box and label
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 4)
            cv2.putText(frame, results1.names[int(class_id)].upper(), (int(x1), int(y1 - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3, cv2.LINE_AA)
            objects=results1.names[int(class_id)]
            print(objects)

    for result in results2.boxes.data.tolist():
        x1, y1, x2, y2, score, class_id = result

        if score > threshold2:
            # Draw bounding box and label
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 4)
            cv2.putText(frame, results2.names[int(class_id)].upper(), (int(x1), int(y1 - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3, cv2.LINE_AA)
            objects=results2.names[int(class_id)]
            print(objects)

    # Show the frame
    cv2.imshow("Live Detection", frame)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
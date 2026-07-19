detection.py
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

model1 = YOLO("/home/pi/Autonomous_Driving/runs1/detect/train/weights/last.pt")
model2 = YOLO("/home/pi/Autonomous_Driving/runs2/detect/train/weights/last.pt")
model3 = YOLO("yolov8n.pt")

threshold1 = 0.8
threshold2 = 0.4
threshold3 = 0.5

while True:

    frame = picam.capture_array()

    results1 = model1(frame)[0]

    for result in results1.boxes.data.tolist():
        x1, y1, x2, y2, score, class_id = result

        if score > threshold1:
            label = results1.names[int(class_id)]

            cv2.rectangle(frame,(int(x1),int(y1)),(int(x2),int(y2)),(0,255,0),3)
            cv2.putText(frame,label.upper(),(int(x1),int(y1-10)),
                        cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)

            print("Model1:", label)

    results2 = model2(frame)[0]

    for result in results2.boxes.data.tolist():
        x1, y1, x2, y2, score, class_id = result

        if score > threshold2:
            label = results2.names[int(class_id)]

            cv2.rectangle(frame,(int(x1),int(y1)),(int(x2),int(y2)),(255,0,0),3)
            cv2.putText(frame,label.upper(),(int(x1),int(y1-10)),
                        cv2.FONT_HERSHEY_SIMPLEX,1,(255,0,0),2)

            print("Model2:", label)

    results3 = model3(frame)[0]

    for result in results3.boxes.data.tolist():
        x1, y1, x2, y2, score, class_id = result

        if score > threshold3:
            label = results3.names[int(class_id)]

            cv2.rectangle(frame,(int(x1),int(y1)),(int(x2),int(y2)),(0,0,255),3)
            cv2.putText(frame,label.upper(),(int(x1),int(y1-10)),
                        cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),2)

            print("Model3:", label)


    cv2.imshow("Live Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


cv2.destroyAllWindows()
picam.stop()
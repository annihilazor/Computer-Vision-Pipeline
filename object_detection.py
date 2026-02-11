import cv2
from ultralytics import YOLO
import numpy as np

model = YOLO('yolov8n.pt')

cap = cv2.VideoCapture('street.mp4')

while True:
    ret, frame = cap.read()
    results = model(frame, classes = [0])  # class 0 is for personṇ
    annotated_frame= results[0].plot()
    annotated_frame = cv2.resize(annotated_frame, (400, 300))
    cv2.imshow('Live camer detection', annotated_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break   
cap.release()
cv2.destroyAllWindows()





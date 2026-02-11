import cv2
from ultralytics import YOLO
import numpy as np

model  = YOLO('yolov8n.pt')

vid = cv2.VideoCapture('bottles.mp4')

unique_id = set()

while True:
    ret, frame = vid.read()
    results = model.track(frame, classes=[39], persist=True, verbose=False)  # class 39 is for bottle
    annotated_frame = results[0].plot()

    if results[0].boxes and results[0].boxes.id is not None:
        ids = results[0].boxes.id.numpy()
        for oid in ids:
            unique_id.add(oid)
        cv2.putText(annotated_frame, f'Count: {len(unique_id)}', (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow('Object counting', annotated_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break  
 

vid.release()
cv2.destroyAllWindows()

# import cv2
# from ultralytics import YOLO
# import numpy as np

# model = YOLO('yolov8n.pt')

# vid = cv2.VideoCapture('bottles.mp4')
# unique_id = set()
# bottle_count = 0  # <-- count variable

# while True:
#     ret, frame = vid.read()
#     if not ret:
#         break

#     # Run detection
#     results = model.track(frame, classes=[39], persist = True)  # class 39 is for bottle

#     annotated_frame = results[0].plot()

#     # Check if boxes exist
#     if hasattr(results[0].boxes, 'id') and results[0].boxes.id is not None:
#         ids = results[0].boxes.id.cpu().numpy()
#         for oid in ids:
#             if oid not in unique_id:
#                 unique_id.add(oid)
#                 bottle_count += 1  # increment count only for new bottles

#     # Display count
#     cv2.putText(annotated_frame, f'Bottles Count: {bottle_count}', (20, 50), 
#                 cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
#     cv2.imshow('Object counting', annotated_frame)

#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# vid.release()
# cv2.destroyAllWindows()

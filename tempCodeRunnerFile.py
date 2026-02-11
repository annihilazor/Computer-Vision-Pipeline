
cap = cv2.VideoCapture(0)

while True:
    ret, frame =cap.read()
    results = model(frame)
    annotated_frame= results[0].plot()
    cv2.imshow('Live camer detection', annotated_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break   
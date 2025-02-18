import cv2


cap = cv2.VideoCapture(0)  # Open the default webcam


while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break


    cv2.imshow("Camera Feed", frame)


    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()



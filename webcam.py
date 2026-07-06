from ultralytics import YOLO
import cv2

if __name__ == '__main__':
    model = YOLO(r'C:\Users\ASUS\runs\detect\action_recognition_v1-4\weights\best.pt')

    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model.predict(frame, device=0, conf=0.15, verbose=False)
        annotated_frame = results[0].plot(line_width=3, font_size=1.5)

        if len(results[0].boxes) > 0:
            print("DETECTED:", results[0].names[int(results[0].boxes.cls[0])])

        cv2.imshow("Action Recognition", annotated_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
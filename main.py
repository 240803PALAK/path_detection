import cv2
import numpy as np


def detect_and_highlight_path(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    lower_black = np.array([0, 0, 0])
    upper_black = np.array([180, 255, 50])

    mask = cv2.inRange(hsv, lower_black, upper_black)

    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if len(contours) > 0:
        largest_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest_contour)
        center_x = x + (w // 2)

        frame_center_x = frame.shape[1] // 2
        direction = "center"
        if center_x < frame_center_x - 20:
            direction = "left"
        elif center_x > frame_center_x + 20:
            direction = "right"

        cv2.drawContours(frame, [largest_contour], 0, (0, 255, 0), 2)
        print(direction)

    cv2.imshow("Path Detection", frame)


cap = cv2.VideoCapture(0)

while True:

    ret, frame = cap.read()

    detect_and_highlight_path(frame.copy())

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
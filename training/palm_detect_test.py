import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2
from vision.palmtracker import findPalm


def main():
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    while True:
        success, img = cap.read()
        if not success:
            break

        img, info = findPalm(img)

        status = info[2] > 0
        cv2.putText(img, f"Status: {'DETECTED' if status else 'NO HAND'}",
                     (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0) if status else (0, 0, 255), 2)

        cv2.putText(img, f"cx:{info[0]} cy:{info[1]} area:{info[2]}",
                     (10, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps > 0:
            cv2.putText(img, f"FPS: {int(fps)}",
                         (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        print(f"Center: ({info[0]}, {info[1]}) Area: {info[2]}", end="\r")

        cv2.imshow("Palm Detection (MediaPipe)", img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

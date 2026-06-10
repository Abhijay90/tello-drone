import sys
import os
import time
import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from vision.palmtracker import findPalm
from vision.gestures import (
    Gesture,
    classify,
    get_thumb_distance,
    get_finger_count,
    print_thresholds,
)

DATA_SOURCE = "webcam"  # "webcam" | "drone"
WEBCAM_IDX = 0

GESTURE_DRONE_MAP = {
    Gesture.OPEN_PALM: ("→", "FORWARD"),
    Gesture.CLOSED_FIST: ("←", "BACKWARD"),
    Gesture.THUMBS_UP: ("↑", "UP"),
    Gesture.THUMBS_DOWN: ("↓", "DOWN"),
    Gesture.PALM_DOWN: ("↑", "DIST HOLD"),
    Gesture.PALM_UP: ("HOVER", "HOVER"),
    Gesture.UNKNOWN: ("○", "HOVER"),
}

# No Hand info (used when no hand detected)
NO_HAND_INFO = ("—", "NO HAND")


def get_frame(me, cap):
    if DATA_SOURCE == "drone" and me:
        frame = me.get_frame_read().frame
        return frame
    elif cap is not None:
        ret, frame = cap.read()
        if ret and frame is not None and frame.size > 0:
            return frame
    return None


def draw_top_bar(img, gesture, cx, cy, area, thumb_px, finger_count):
    """Draw 2-line text bar at top of frame."""
    line1 = f"Gesture: {gesture}  │ cx:{cx} cy:{cy}  │ area:{area}"
    line2 = f"THUMB: {thumb_px:.1f}px  │ FINGERS: {finger_count[0]}/{finger_count[1]}"

    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.6
    color = (255, 255, 255)
    thickness = 1

    y1 = 25
    y2 = 50

    cv2.putText(img, line1, (10, y1), font, scale, color, thickness, cv2.LINE_AA)
    cv2.putText(img, line2, (10, y2), font, scale, color, thickness, cv2.LINE_AA)


def draw_gesture_table(img, active_gesture):
    """Draw gesture table panel in top-right corner."""
    img_h, img_w = img.shape[:2]
    panel_x = img_w - 240
    panel_y = 80
    panel_w = 220
    panel_h = 260

    # Panel background
    panel_bg = img[panel_y:panel_y + panel_h, panel_x:panel_x + panel_w].copy()

    for y in range(panel_h):
        for x in range(panel_w):
            panel_bg[y, x] = (40, 40, 40)  # Dark bg

    # Panel border
    cv2.rectangle(img, (panel_x, panel_y), (panel_x + panel_w, panel_y + panel_h), (100, 100, 100), 2)

    # Title
    cv2.putText(img, "GESTURES", (panel_x + 60, panel_y + 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)

    # Draw all gestures
    gestures = [
        Gesture.OPEN_PALM,
        Gesture.CLOSED_FIST,
        Gesture.THUMBS_UP,
        Gesture.THUMBS_DOWN,
        Gesture.PALM_DOWN,
        Gesture.PALM_UP,
        Gesture.UNKNOWN,
    ]

    start_y = panel_y + 50
    line_h = 28

    for i, g in enumerate(gestures):
        y = start_y + i * line_h
        active = (g == active_gesture)

        # Marker
        if active:
            marker_color = (0, 255, 0)  # Green
            marker = "●"
        else:
            marker_color = (128, 128, 128)  # Gray
            marker = "○"

        cv2.putText(img, marker, (panel_x + 8, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, marker_color, 2, cv2.LINE_AA)

        # Name
        name = g.replace("_", " ").title()
        cv2.putText(img, name, (panel_x + 35, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1, cv2.LINE_AA)

        # Arrow + drone direction
        if active:
            arrow, direction = GESTURE_DRONE_MAP.get(g, ("?", "?"))
            text_color = (0, 255, 0)
            cv2.putText(img, arrow, (panel_x + panel_w - 90, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, text_color, 2, cv2.LINE_AA)
            cv2.putText(img, direction, (panel_x + panel_w - 65, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 1, cv2.LINE_AA)
        else:
            cv2.putText(img, "-", (panel_x + panel_w - 70, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (80, 80, 80), 1, cv2.LINE_AA)


def draw_status_footer(img):
    """Draw status bar at bottom of frame."""
    img_h, img_w = img.shape[:2]
    y = img_h - 15

    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.55
    color = (200, 200, 200)

    text = f"GESTURE TESTER v1 | [{DATA_SOURCE.upper()}] | Q: Exit | T: Toggle Source"
    cv2.putText(img, text, (10, y), font, scale, color, 1, cv2.LINE_AA)


def main():
    global DATA_SOURCE
    cap = None
    me = None

    initialized = False

    # Initialize based on DATA_SOURCE
    if DATA_SOURCE == "webcam":
        print("[webcam] Starting webcam capture...")
        cap = cv2.VideoCapture(WEBCAM_IDX)
        if not cap.isOpened():
            print(f"Cannot open webcam index {WEBCAM_IDX}")
            sys.exit(1)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        initialized = True
    else:
        from djitellopy import tello
        print("[drone] Initializing Tello...")
        try:
            me = tello.Tello()
            me.connect()
            me.streamon()
            initialized = True
        except Exception as e:
            print(f"[drone] Cannot connect to drone: {e}")
            print("[drone] Falling back to webcam")
            DATA_SOURCE = "webcam"
            cap = cv2.VideoCapture(WEBCAM_IDX)
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            initialized = True
            me = None

    if not initialized:
        print("Failed to initialize any data source.")
        sys.exit(1)

    print(f"Gesture Tester v1 starting. DATA_SOURCE={DATA_SOURCE.upper()}")
    print("Press 'q' to exit, 't' to toggle source, 'p' to print thresholds")

    frame_count = 0
    landmarks = None  # Track landmarks across loops
    gesture = Gesture.UNKNOWN

    while True:
        if not initialized:
            break

        img = get_frame(me, cap)

        if img is None or img.size == 0:
            fallback = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(fallback, "NO FRAME", (150, 200),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.imshow("Gesture Tester", fallback)
        else:
            # Run palm tracking
            img, info = findPalm(img)

            # Extract info
            cx, cy, area, landmarks = info

            gesture = Gesture.UNKNOWN
            thumb_px = 0.0
            finger_ext, finger_total = 0, 4

            if landmarks is not None:
                gesture = classify(landmarks, img.shape[1], img.shape[0])
                thumb_px = get_thumb_distance(landmarks, img.shape[1], img.shape[0])
                finger_ext, finger_total = get_finger_count(landmarks, img.shape[1], img.shape[0])
            else:
                gesture = Gesture.UNKNOWN
                thumb_px = 0.0
                finger_ext, finger_total = 0, 4

            # Draw overlays
            draw_top_bar(img, gesture, cx, cy, area, thumb_px, (finger_ext, finger_total))
            draw_gesture_table(img, gesture)
            draw_status_footer(img)

            # Add source indicator
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(img, f"[{DATA_SOURCE.upper()}]", (10, 70),
                        font, 0.5, (0, 0, 255), 1, cv2.LINE_AA)

            # Add threshold info
            status_y = int(img.shape[0]) - 30
            palm_dir = "—"
            if gesture == Gesture.PALM_DOWN:
                palm_dir = "↓"
            elif gesture == Gesture.PALM_UP:
                palm_dir = "↑"
            elif gesture == Gesture.OPEN_PALM:
                palm_dir = "●"
            threshold_text = f"[{gesture:15s}] THUMB: {thumb_px:6.1f}px  PALS: {palm_dir:3s}  EXT: {finger_ext}/{finger_total}"
            cv2.putText(img, threshold_text, (10, status_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1, cv2.LINE_AA)

            cv2.imshow("Gesture Tester", img)

        # Keyboard — handled at end of every iteration
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('t'):
            # Toggle data source
            DATA_SOURCE = "drone" if DATA_SOURCE == "webcam" else "webcam"
            print(f"Switching to [{DATA_SOURCE.upper()}]...")
            if cap is not None:
                cap.release()
                cap = None
            if 'me' in locals() and me:
                try:
                    me.streamoff()
                except Exception:
                    pass
            me = None
            initialized = False
            time.sleep(0.3)

            if DATA_SOURCE == "webcam":
                cap = cv2.VideoCapture(WEBCAM_IDX)
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                initialized = cap.isOpened()
                print("Using webcam")
            else:
                from djitellopy import tello
                try:
                    me = tello.Tello()
                    me.connect()
                    me.streamon()
                    initialized = True
                    print("Using drone")
                except Exception as e:
                    print(f"Drone connection failed ({e}), falling back to webcam")
                    DATA_SOURCE = "webcam"
                    cap = cv2.VideoCapture(WEBCAM_IDX)
                    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                    initialized = cap.isOpened()
                    me = None
        elif key == ord('p'):
            if landmarks is not None:
                print_thresholds(landmarks, img.shape[1], img.shape[0])
            else:
                print("NO HAND detected")

        frame_count += 1
        if frame_count % 30 == 0:
            palm_dir = "—"
            if gesture == Gesture.PALM_DOWN:
                palm_dir = "↓"
            elif gesture == Gesture.PALM_UP:
                palm_dir = "↑"
            print(f"[{gesture:15s}] THUMB:{thumb_px:7.1f}px  PALS:{palm_dir:3s}  EXT:{finger_ext}/{finger_total}")


    # Cleanup
    print(f"\n[{DATA_SOURCE.upper()}] Cleaning up...")
    if 'me' in locals() and me:
        try:
            me.streamoff()
            me.end()
        except Exception:
            pass
    if cap:
        cap.release()

    cv2.destroyAllWindows()
    print("Done.")


if __name__ == "__main__":
    main()

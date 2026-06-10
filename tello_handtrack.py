# palm gesture flight control using hand landmark detection + vision.gestures_v2 engine
from djitellopy import tello
from time import sleep, time, perf_counter
from vision.palmtracker import findPalm
from vision.gestures_v2 import Gesture, classify_and_debug, DEADZONE_CENTER, DEADZONE_WIDTH
import cv2
import sys
import os

DEBUG_DIR = "debug_frames"
os.makedirs(DEBUG_DIR, exist_ok=True)
_proof_count = 0
_prev_gesture = Gesture.UNKNOWN
_prev_classify_time = 0.0  # last classify() call time

# ============== CONSTANTS ================

# Display
RES_W, RES_H = 960, 720
WINDOW_NAME = "Palm Flight"

# Detection targets
TARGET_AREA = 4000
TARGET_X = RES_W // 2

# Gesture-to-RC speed mapping: (speed_y, speed_z)
# Y axis = forward/backward (positive=closer to palm, negative=farther)
# Z axis = up/down (positive=away=up, negative=toward=back)
# Base scales: y_scale=35 px/s (speed=1→35), z_scale=0.4 m/s (speed=1→0.4 m/s)
GESTURE_RC = {
    Gesture.OPEN_PALM:   (0, 0),
    Gesture.CLOSED_FIST: (0, 0),
    Gesture.THUMBS_UP:   (0, 1),
    Gesture.THUMBS_DOWN: (0, -1),
    Gesture.PALM_UP:     (0, 0),
    Gesture.PALM_DOWN:   (0, 0),
    Gesture.PALM_LEFT:   (-1, 0),
    Gesture.PALM_RIGHT:  (1, 0),
    Gesture.UNKNOWN:     (0, 0)
}

# Gesture-to-overlay labels: (arrow, command_text)
GESTURE_CMD = {
    Gesture.OPEN_PALM:   ("\u2299", "HOVER"),
    Gesture.CLOSED_FIST: ("\u2299", "HOVER"),
    Gesture.THUMBS_UP:   ("\u2191", "UP"),
    Gesture.THUMBS_DOWN: ("\u2193", "DOWN"),
    Gesture.PALM_UP:     ("\u2299", "HOVER"),
    Gesture.PALM_DOWN:   ("\u2299", "HOVER"),
    Gesture.PALM_LEFT:   ("\u2190", "LEFT"),
    Gesture.PALM_RIGHT:  ("\u2192", "RIGHT"),
    Gesture.UNKNOWN:     ("\u25cb", "NO TRACK")
}

# Display constants
OVERLAY_FONT = cv2.FONT_HERSHEY_SIMPLEX
OVERLAY_SCALE_GESTURE = 0.7
OVERLAY_SCALE_INFO = 0.5
OVERLAY_THICKNESS = 2
OVERLAY_THICKNESS_THIN = 1
COLOR_GREEN = (0, 255, 0)
COLOR_WHITE = (255, 255, 255)
COLOR_YELLOW = (255, 255, 0)
COLOR_RED = (0, 0, 255)
COLOR_CYAN = (255, 255, 0)

# Position constants (x, y) for overlay text layers
POS_GESTURE = (20, 30)
POS_CMD = (20, 60)
POS_SPEED = (10, 90)
POS_MODE = (10, 110)

# Drone settings
DEFAULT_WEBCAM_IDX = 0
LAND_WAIT_SEC = 2
EXIT_WAIT_SEC = 1

# PID state (kept for compatibility)
ZERO_PID = (0, 0)

# Gesture classification debounce: minimum time between classify() calls (seconds)
CLASSIFY_DEBOUNCE_SEC = 0.5

# Motion modes: toggled with 'm' key
MOTION_MODE_DEPLOY = "deploy"      # Hand position (left/right of center) controls direction
MOTION_MODE_CLASSIFY = "classify"   # Discrete gesture recognition (thumbs, palms, etc.)

# Motion mode constants
MOTION_MODE_DEADZONE = "deadzone"   # Hovering within deadzone in deploy mode
MOTION_MODE_DIR = "dir"             # Directional control based on hand placement
MOTION_MODE_CLASSIFY_GESTURE = "gesture"  # Gesture classification state

# Deploy deadzone threshold in pixels (from center)
# Below this: deadzone (hover). Above this: directional (PALM_LEFT/PALM_RIGHT)
DEPLOY_DEADZONE_PX = 30


# ============= HELPER FUNCTIONS =============

def gesture_to_rc(gesture):
    """Get (speed_y, speed_z) from a gesture."""
    return GESTURE_RC.get(gesture, (0, 0))


def gesture_to_cmd_text(gesture):
    """Get (arrow, command) labels from a gesture."""
    return GESTURE_CMD.get(gesture, ("\u25cb", "UNKNOWN"))


def draw_overlays(img, gesture, speed_y, speed_z, mode, sub_mode=None):
    """Draw gesture overlay info on frame. Returns the modified image."""
    arrow, cmd = gesture_to_cmd_text(gesture)

    # Gesture line
    cv2.putText(img, f"Gesture: {gesture}", POS_GESTURE,
                OVERLAY_FONT, OVERLAY_SCALE_GESTURE, COLOR_GREEN, OVERLAY_THICKNESS)
    cv2.putText(img, f"{arrow} {cmd}", POS_CMD,
                OVERLAY_FONT, OVERLAY_SCALE_GESTURE, COLOR_WHITE, OVERLAY_THICKNESS)

    # Speed line
    cv2.putText(img, f"speed_y:{speed_y} speed_z:{speed_z}", POS_SPEED,
                OVERLAY_FONT, OVERLAY_SCALE_INFO, COLOR_WHITE, OVERLAY_THICKNESS_THIN)

    # Mode line - show sub_mode if provided
    mode_text = mode if sub_mode is None else f"{mode} ({sub_mode})"
    cv2.putText(img, f"[{mode_text} MODE]", POS_MODE,
                OVERLAY_FONT, OVERLAY_SCALE_INFO, COLOR_YELLOW, OVERLAY_THICKNESS_THIN)

    return img


def send_rc(me, y, z):
    """Send RC only if in drone mode and armed."""
    if DRONE_MODE and me:
        me.send_rc_control(0, int(y), int(z), 0)


def get_frame(me, cap):
    """Get next frame from drone stream or webcam."""
    if DRONE_MODE and me:
        return me.get_frame_read().frame
    elif cap is not None:
        ret, frame = cap.read()
        if ret and frame is not None:
            return frame
    return None


# ============= STATE =============

DRONE_MODE = False
CURRENT_MOTION_MODE = MOTION_MODE_DEPLOY  # start in deploy mode (default)


# ============= INIT =============

def init_device():
    """Initialize either Tello drone or webcam. Returns (me_or_none, cap_or_none)."""
    me = None
    cap = None

    if DRONE_MODE:
        me = tello.Tello()
        me.connect()
        print(f"Battery: {me.get_battery()}%")
        me.streamon()
        print("Taking off...")
        # me.takeoff()
        print("Drone ready \u2014 start gesturing!")
    else:
        print("[webcam debug mode \u2014 no drone commands]")
        cap = cv2.VideoCapture(DEFAULT_WEBCAM_IDX)
        if not cap.isOpened():
            print(f"Cannot open webcam index {DEFAULT_WEBCAM_IDX}")
            sys.exit(1)

        for i in range(1, 5):  # test 4 frames to be sure
            ret, frame = cap.read()
            if ret and frame is not None and frame.size > 0:
                print(f"  Frame {i}: OK")
            else:
                print(f"  Frame {i}: failed")
                cap.release()
                sys.exit(1)
        print("Webcam OK \u2014 starting flight loop...")

    return me, cap


me, cap = init_device()
prev_error_y, prev_error_z = ZERO_PID
print(f"Started in MOTION_MODE={CURRENT_MOTION_MODE}")


# ============= MAIN LOOP =============

try:
    while True:
        img = get_frame(me, cap)
        if img is None:
            print("no image received")
            continue

        img = cv2.resize(img, (RES_W, RES_H))
        img, info = findPalm(img, DRONE_MODE)
        cx, cy, area, landmarks = info

        error_y = error_z = 0
        gesture = Gesture.UNKNOWN
        speed_y = speed_z = 0
        mode = "WEBCAM"
        if DRONE_MODE:
            mode = "DRONE"

        sub_mode = None  # Will be set based on motion mode

        if cx == 0 or area == 0 or landmarks is None:
            # No hand detected
            gesture = Gesture.UNKNOWN
            speed_y, speed_z = gesture_to_rc(gesture)
            sub_mode = "no_hand"
        elif CURRENT_MOTION_MODE == MOTION_MODE_DEPLOY:
            # === Deploy Mode: Hand X position controls direction ===
            # Deadzone logic: if hand is near center, hover; otherwise directional
            if cx < TARGET_X - DEPLOY_DEADZONE_PX:
                gesture = Gesture.PALM_LEFT
                speed_y, speed_z = gesture_to_rc(gesture)
                sub_mode = MOTION_MODE_DEADZONE  # Left side
            elif cx > TARGET_X + DEPLOY_DEADZONE_PX:
                gesture = Gesture.PALM_RIGHT
                speed_y, speed_z = gesture_to_rc(gesture)
                sub_mode = MOTION_MODE_DIR  # Right side
            else:
                # Within deadzone: hover
                gesture = Gesture.UNKNOWN
                speed_y, speed_z = 0, 0
                sub_mode = MOTION_MODE_DEADZONE
        elif CURRENT_MOTION_MODE == MOTION_MODE_CLASSIFY:
            # === Classify Mode: Recognize discrete gestures ===
            gesture = classify_and_debug(landmarks, img.shape[1], img.shape[0])
            speed_y, speed_z = gesture_to_rc(gesture)
            sub_mode = MOTION_MODE_CLASSIFY_GESTURE

        img = draw_overlays(img, gesture, speed_y, speed_z, mode, sub_mode=sub_mode)

        if DRONE_MODE:
            send_rc(me, int(speed_y), int(speed_z))

        cv2.imshow(WINDOW_NAME, img)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            print("Landing...")
            break
        elif key == ord('t'):
            print("Toggling mode...")
            if DRONE_MODE:
                if me:
                    me.land()
                    sleep(LAND_WAIT_SEC)
                DRONE_MODE = False
                print("Switched to WEBCAM MODE")
                # Release drone resources
                if me:
                    del me
                    me = None
            else:
                me = tello.Tello()
                me.connect()
                print(f"Battery: {me.get_battery()}%")
                me.streamon()
                # me.takeoff()
                print("Drone ready \u2014 start gesturing!")
                DRONE_MODE = True
                print("Switched to DRONE MODE")
        elif key == ord('m'):
            # Toggle motion mode between deploy and classify
            CURRENT_MOTION_MODE = MOTION_MODE_CLASSIFY if CURRENT_MOTION_MODE == MOTION_MODE_DEPLOY else MOTION_MODE_DEPLOY
            print(f"MOTION_MODE switched to {CURRENT_MOTION_MODE}")

        prev_error_y, prev_error_z = error_y, error_z

except KeyboardInterrupt:
    print("Interrupted by user")
finally:
    if DRONE_MODE and 'me' in locals() and me is not None:
        # me.land()
        sleep(EXIT_WAIT_SEC)
        me.streamoff()
    elif 'cap' in locals() and cap is not None and not cap.empty():
        cap.release()
        cap = None
    cv2.destroyAllWindows()
    print("Exit complete.")

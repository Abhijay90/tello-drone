# Real-time webcam gesture classifier test with confusion matrix tracking
# Tests all 8 gesture types against live camera feed, showing per-class accuracy.
# Uses MediaPipe HandLandmarker directly (same approach as gesture_webcam_test.py)

import sys
sys.path.insert(0, '/home/abhikun/Desktop/drone/tello-drone')

import time
import os
import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from vision.gestures_v2 import classify, Gesture, _MODEL_AVAILABLE
from gesture_rc_mapping import GESTURE_RC_MAP, HOVER_COMMAND

# === DRONE/WEBCAM MODE TOGGLE ===
DRONE_MODE = True  # Set True for real drone, False for webcam debug

if DRONE_MODE:
    from djitellopy import tello
    from time import sleep

HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17), (5, 17),
]

GESTURE_COLORS = {
    Gesture.OPEN_PALM: (0, 255, 0),
    Gesture.CLOSED_FIST: (0, 0, 255),
    Gesture.THUMBS_UP: (255, 0, 0),
    Gesture.THUMBS_DOWN: (255, 255, 0),
    Gesture.PALM_UP: (0, 255, 255),
    Gesture.PALM_DOWN: (255, 0, 255),
    Gesture.PALM_LEFT: (128, 0, 128),
    Gesture.PALM_RIGHT: (0, 128, 128),
    Gesture.UNKNOWN: (128, 128, 128),
}

VALID_GESTURES = [
    Gesture.OPEN_PALM, Gesture.CLOSED_FIST,
    Gesture.THUMBS_UP, Gesture.THUMBS_DOWN,
    Gesture.PALM_UP, Gesture.PALM_DOWN,
    Gesture.PALM_LEFT, Gesture.PALM_RIGHT,
]

gesture_correct = {}
gesture_total = {}
hover_pid = None
prev_centroid = None
centroid_smooth_alpha = 0.3


def reset_stats():
    global gesture_correct, gesture_total
    gesture_correct = {g: 0 for g in VALID_GESTURES}
    gesture_total = {g: 0 for g in VALID_GESTURES}

def update_stats(prediction, actual):
    if actual in gesture_correct:
        gesture_total[actual] += 1
        if prediction == actual:
            gesture_correct[actual] += 1

def draw_stats(frame):
    w = frame.shape[1]
    x, y, ww, hh = w - 290, 10, 280, 240
    cv2.rectangle(frame, (x, y), (x + ww, y + hh), (0, 0, 0), cv2.FILLED)
    cv2.rectangle(frame, (x, y), (x + ww, y + hh), (100, 100, 100), 1)
    cv2.putText(frame, "ACCURACY PER GESTURE:", (x + 10, y + 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    oy = 50
    for gesture in VALID_GESTURES:
        t = gesture_total[gesture]
        c = gesture_correct[gesture]
        acc = (c / t * 100) if t > 0 else 0
        color = (0, 255, 0) if acc >= 90 else (255, 255, 0) if acc >= 60 else (0, 255, 255)
        cv2.putText(frame, f"{gesture.name:<13s}: {acc:5.1f}% ({c}/{t})",
                    (x + 10, y + oy), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
        oy += 22
    total = sum(gesture_total.values())
    ov = sum(gesture_correct.values()) / total * 100 if total > 0 else 0
    cv2.putText(frame, f"OVERALL   : {ov:5.1f}% ({sum(gesture_correct.values())}/{total})",
                (x + 10, y + hh - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

def overlay_prediction(frame, prediction):
    x, y = 10, 10
    color = GESTURE_COLORS.get(prediction, (128, 128, 128))
    cv2.rectangle(frame, (x - 10, y - 10), (x + 180, y + 35), (0, 0, 0), cv2.FILLED)
    cv2.rectangle(frame, (x - 10, y - 10), (x + 180, y + 35), color, 2)
    cv2.putText(frame, f"Gesture: {prediction.name}",
                (x, y + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    cv2.putText(frame, "SVM MODEL" if _MODEL_AVAILABLE else "HEURISTIC MODE",
                (x, y + 45), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

def draw_landmarks(frame, hand_landmarks):
    if not hand_landmarks:
        return
    for i, j in HAND_CONNECTIONS:
        if i < len(hand_landmarks[0]) and j < len(hand_landmarks[0]):
            pt1 = (int(hand_landmarks[0][i].x * frame.shape[1]), int(hand_landmarks[0][i].y * frame.shape[0]))
            pt2 = (int(hand_landmarks[0][j].x * frame.shape[1]), int(hand_landmarks[0][j].y * frame.shape[0]))
            cv2.line(frame, pt1, pt2, (0, 255, 0), 2)
    for i in range(min(21, len(hand_landmarks[0]))):
        pt = (int(hand_landmarks[0][i].x * frame.shape[1]), int(hand_landmarks[0][i].y * frame.shape[0]))
        cv2.circle(frame, pt, 5, (255, 0, 0), -1)

def get_hand_centroid(landmarks):
    """Return (cx, cy) normalized [0,1] from 21 hand landmarks."""
    xs = [landmarks[i].x for i in range(21)]
    ys = [landmarks[i].y for i in range(21)]
    return (sum(xs) / len(xs), sum(ys) / len(ys))


def compute_ground_truth(hand_landmarks, w_px, h_px):
    if not hand_landmarks:
        return Gesture.UNKNOWN
    lm = hand_landmarks[0]
    pts = [(lm[i].x * w_px, lm[i].y * h_px) for i in range(21)]
    thumb_ext = pts[4][1] < pts[2][1]
    fingers_ext = pts[8][1] < pts[6][1] and pts[12][1] < pts[10][1]
    if thumb_ext and not fingers_ext:
        thumb_offset = pts[4][0] - pts[1][0]
        return Gesture.THUMBS_DOWN if thumb_offset < -20 else Gesture.THUMBS_UP
    elif thumb_ext and fingers_ext:
        return Gesture.OPEN_PALM
    elif not thumb_ext and not fingers_ext:
        return Gesture.CLOSED_FIST
    else:
        return Gesture.OPEN_PALM

def main():
    print("\n" + "=" * 60)
    print("GESTURE CLASSIFIER - REAL-TIME TEST")
    print("=" * 60)
    if DRONE_MODE:
        print("Mode: DRONE FPV")
        me = tello.Tello()
        me.connect()
        print(f"Battery: {me.get_battery()}%")
        me.streamon()
        me.takeoff()

        print("Drone ready — start gesturing!\n")

        # Gesture-to-RC mapping variables
        last_rc_command = (0, 0, 0, 0)
        gesture_stable_gesture = None
        debounce_counter = 0
        DEBOUNCE_FRAMES = 3


        # Phase 3.1: hover stabilization globals
        global hover_pid, prev_centroid
        hover_pid = type('HoverPID', (), {
            'kp': 15.0, 'kd': 8.0, 'ki': 0.0,
            'drift_threshold': 0.02, 'max_output': 20.0,
            'prev_err_x': 0.0, 'prev_err_y': 0.0, 'prev_time': time.time(),
        })()
        prev_centroid = None

        # Hover/stabilization variables
        gesture_log_file = open('/home/abhikun/Desktop/drone/tello-drone/pid_hover_log.txt', 'a')
        gesture_log_file.write(f"{'Timestamp':<22} {'Altitude (cm)':<16} {'VGX':<8} {'VGY':<8} {'Gesture':<16} {'RC Command':<16}\n")
        gesture_log_file.flush()
        last_log_time = time.time()
        LOG_INTERVAL = 2  # seconds

    else:
        print("Mode: WEBCAM")
        print("Key Bindings:  q=Quit  h=Help  c=Clear stats\n")
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Error: Could not open webcam")
            sys.exit(1)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 960)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        webcam_cap = cap

    model_path = '/home/abhikun/Desktop/drone/tello-drone/models/hand_landmarker.task'
    base_options = python.BaseOptions(model_asset_path=model_path)
    detector = vision.HandLandmarker.create_from_options(
        vision.HandLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.IMAGE,
            num_hands=2,
        )
    )


    reset_stats()
    frame_count = 0

    try:
        while True:
            h_px, w_px = 0, 0
            frame = None
            
            if DRONE_MODE:
                rgb_frame = me.get_frame_read().frame
                if rgb_frame is None:
                    continue
                frame = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2BGR)
                frame=cv2.flip(frame,1)
                h_px, w_px = frame.shape[:2]
            else:
                ret, frame = webcam_cap.read()
                if not ret or frame is None:
                    continue
                w_px = frame.shape[1]
                h_px = frame.shape[0]
                frame = cv2.flip(frame, 1)

            frame_count += 1
            
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            detection_results = detector.detect(img)

            draw_landmarks(frame, detection_results.hand_landmarks)

            if detection_results.hand_landmarks and len(detection_results.hand_landmarks) > 0:
                prediction = classify(
                    detection_results.hand_landmarks[0], w_px, h_px
                )
                actual = compute_ground_truth(detection_results.hand_landmarks, w_px, h_px)
                update_stats(prediction, actual)
            else:
                prediction = Gesture.UNKNOWN
                actual = Gesture.UNKNOWN


            # Gesture stabilization with debounce (runs every frame)
            if DRONE_MODE:
                if detection_results.hand_landmarks and prediction != Gesture.UNKNOWN:
                    if gesture_stable_gesture is None or gesture_stable_gesture != prediction:
                        gesture_stable_gesture = prediction
                        debounce_counter = 0
                        print(f"GESTURE CHANGED: {prediction.name} (debouncing...)")
                    else:
                        debounce_counter += 1
                        if debounce_counter == DEBOUNCE_FRAMES:
                            print(f"Gesture STABLE: {gesture_stable_gesture.name} -> ready for RC commands")
                else:
                    gesture_stable_gesture = None
                    debounce_counter = 0

            # Gesture-to-RC command mapping with Phase 3.1 hover stabilization
            if DRONE_MODE:
                if gesture_stable_gesture is not None and debounce_counter >= DEBOUNCE_FRAMES:
                    print("sending command on basis of gesutre")
                    base_cmd = GESTURE_RC_MAP.get(gesture_stable_gesture.name, HOVER_COMMAND)
                    hover_sent = False

                    if gesture_stable_gesture == Gesture.CLOSED_FIST and detection_results.hand_landmarks:
                        lm = detection_results.hand_landmarks[0]
                        raw_cx, raw_cy = get_hand_centroid(lm)

                        if prev_centroid is not None:
                            raw_cx = centroid_smooth_alpha * raw_cx + (1 - centroid_smooth_alpha) * prev_centroid[0]
                            raw_cy = centroid_smooth_alpha * raw_cy + (1 - centroid_smooth_alpha) * prev_centroid[1]
                        prev_centroid = (raw_cx, raw_cy)

                        now = time.time()
                        dt = max(now - hover_pid.prev_time, 0.001)
                        hover_pid.prev_time = now

                        err_x = raw_cx - 0.5
                        err_y = raw_cy - 0.5
                        d_err_x = (err_x - hover_pid.prev_err_x) / dt
                        d_err_y = (err_y - hover_pid.prev_err_y) / dt
                        hover_pid.prev_err_x = err_x
                        hover_pid.prev_err_y = err_y

                        corr_roll = hover_pid.kp * err_x + hover_pid.kd * d_err_x
                        corr_yaw = 0

                        if abs(err_x) >= hover_pid.drift_threshold or abs(err_y) >= hover_pid.drift_threshold:
                            corr_roll = int(np.clip(corr_roll, -20, 20))
                            corr_yaw = int(np.clip(corr_yaw, -20, 20))
                            if corr_roll != 0 or corr_yaw != 0:
                                print(f"Hover PID: centroid=({raw_cx:.3f},{raw_cy:.3f}) -> corr=({corr_roll},{corr_yaw})")
                                hover_sent = True
                        else:
                            hover_pid.prev_err_x = hover_pid.prev_err_y = 0.0
                            corr_roll = 0
                            corr_yaw = 0

                        base_cmd = (corr_roll, base_cmd[1], base_cmd[2], corr_yaw)

                    rc_cmd = base_cmd
                    if rc_cmd != last_rc_command:
                        print(f"Gesture was {gesture_stable_gesture.name}")
                        print(f"sending rc command {rc_cmd}" )
                        me.send_rc_control(*rc_cmd)
                        last_rc_command = rc_cmd
                        if not hover_sent:
                            print(f"Gesture {gesture_stable_gesture.name} -> RC {rc_cmd}")
                else:
                    # No stable gesture detected — force hover
                    if last_rc_command != HOVER_COMMAND:
                        me.send_rc_control(*HOVER_COMMAND)
                        print(f"Gesture lost, sending hover command (last was {last_rc_command})")
                        last_rc_command = HOVER_COMMAND

            # Periodic height + gesture logging (every LOG_INTERVAL seconds)
            if DRONE_MODE:
                now = time.time()
                if now - last_log_time >= LOG_INTERVAL:
                    last_log_time = now
                    try:
                        state = me.get_current_state()
                        altitude = state.get('h', -1) if state else -1
                        vgx = str(state.get('vgx', -1)) if state else '-1'
                        vgy = str(state.get('vgy', -1)) if state else '-1'
                    except Exception:
                        altitude = -1
                        vgx = '-1'
                        vgy = '-1'
                    current_gesture = gesture_stable_gesture.name if gesture_stable_gesture else "NONE"
                    log_line = f"{now:<16.3f} {altitude:<10} {vgx:<8} {vgy:<8} {current_gesture:<16} {last_rc_command}\n"
                    if gesture_log_file:
                        gesture_log_file.write(log_line)
                        gesture_log_file.flush()
                    
                    # Print state to console
                    print(f"[STATE] Time={now:.3f} Alt={altitude}cm Vgx={vgx} Vgy={vgy} G={current_gesture}")

            overlay_prediction(frame, prediction)
            if frame_count % 30 == 0:
                draw_stats(frame)

            cv2.imshow('Gesture Detector', frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('c') and not DRONE_MODE:
                reset_stats()
                print("Statistics cleared.")
            elif key == ord('h'):
                print("\nKey Bindings: q=Quit  h=Help  c=Clear(only webcam)")
            elif key == ord('t') and not DRONE_MODE:
                print("Toggle DRONE_MODE: edit file line 13")

    finally:


        if DRONE_MODE:
            me.land()
            sleep(2)
            me.streamoff()
            me.end()
        elif not DRONE_MODE:
            webcam_cap.release()
        cv2.destroyAllWindows()

        print("\n" + "=" * 60)
        print("FINAL ACCURACY SUMMARY")
        print("=" * 60)
        total = sum(gesture_total.values())
        if total > 0:
            overall = sum(gesture_correct.values()) / total * 100
            print(f"\nOverall Accuracy: {overall:.1f}% ({sum(gesture_correct.values())}/{total} samples)")
            print("\nPer-Gesture Accuracy:")
            for g in VALID_GESTURES:
                acc = (gesture_correct[g] / gesture_total[g] * 100) if gesture_total[g] > 0 else 0
                mark = "OK" if acc >= 80 else ("WARN" if acc >= 50 else "LOW")
                clr = "\033[92m" if acc >= 80 else "\033[93m" if acc >= 50 else "\033[91m"
                print(f"  {g.name:<15s}: {acc:5.1f}% ({gesture_correct[g]}/{gesture_total[g]}) [{clr}{mark}\033[0m]")
        else:
            print("No samples collected. No hand detected in camera.")
        print(f"\nSVM Model: {'LOADED' if _MODEL_AVAILABLE else 'NOT LOADED'}")
        if not _MODEL_AVAILABLE:
            print("  Run: cd training && python train_and_benchmark.py")
        print(f"Total samples: {total}")
        print("Exit complete.")

if __name__ == '__main__':
    main()

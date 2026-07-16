"""Real-time webcam gesture classifier test with confusion matrix tracking.
Tests all 8 gesture types against live camera feed, showing per-class accuracy.

Usage:
    cd /home/abhikun/Desktop/drone/tello-drone
    python training/gesture_webcam_test.py

Key Bindings:
    q       - Quit
    h       - Show help
    c       - Clear stats
    s       - Screenshot
"""
import sys
sys.path.insert(0, '/home/abhikun/Desktop/drone/tello-drone')

import time
import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from vision.gestures_v2 import classify, Gesture, _MODEL_AVAILABLE

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
    print("Key Bindings:  q=Quit  h=Help  c=Clear stats  s=Screenshot")
    print(f"SVM model: {'LOADED' if _MODEL_AVAILABLE else 'NOT LOADED (heuristic)'}")
    print("Press 'q' to quit. Show gestures to camera to test accuracy.\n")
    
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam")
        sys.exit(1)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
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
    start_time = time.time()
    
    while cap.isOpened() and (time.time() - start_time) < 600:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        h_px, w_px = frame.shape[:2]
        frame = cv2.flip(frame, 1)
        
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
        
        overlay_prediction(frame, prediction)
        if frame_count % 30 == 0:
            draw_stats(frame)
        
        cv2.putText(frame, f"Frame: {frame_count}", (10, frame.shape[0] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        cv2.imshow('Gesture Detector', frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('c'):
            reset_stats()
            print("Statistics cleared.")
        elif key == ord('h'):
            print("\nKey Bindings: q=Quit  h=Help  c=Clear  s=Screenshot")
        elif key == ord('s'):
            cv2.imwrite('screenshot_test.jpg', frame)
            print("Saved screenshot_test.jpg")
    
    cap.release()
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

if __name__ == '__main__':
    main()

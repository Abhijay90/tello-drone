# palmtracker.py

import cv2
import numpy as np
import os
import mediapipe as mp
from mediapipe.tasks.python.vision import HandLandmarker
from mediapipe.tasks.python.core import base_options
from mediapipe.tasks.python.vision.core import vision_task_running_mode
from mediapipe.tasks.python.vision.hand_landmarker import HandLandmarkerOptions, HandLandmarksConnections

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(os.path.dirname(SCRIPT_DIR), "models", "hand_landmarker.task")
HAND_CONNECTIONS = HandLandmarksConnections.HAND_CONNECTIONS

mp_hands = HandLandmarker.create_from_options(
    HandLandmarkerOptions(
        base_options=base_options.BaseOptions(model_asset_path=MODEL_PATH),
        running_mode=vision_task_running_mode.VisionTaskRunningMode.IMAGE,
        num_hands=1,
    )
)


def _draw_landmarks(img, landmarks, img_w, img_h):
    for lm in landmarks:
        x, y = int(lm.x * img_w), int(lm.y * img_h)
        cv2.circle(img, (x, y), 5, (0, 255, 0), -1)


def _draw_connections(img, landmarks, img_w, img_h):
    for conn in HAND_CONNECTIONS:
        p1 = (int(landmarks[conn.start].x * img_w), int(landmarks[conn.start].y * img_h))
        p2 = (int(landmarks[conn.end].x * img_w), int(landmarks[conn.end].y * img_h))
        cv2.line(img, p1, p2, (0, 255, 0), 2)


def findPalm(img):
    # img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) #this line reduced accuracy looks like data from drone reduces info when converted. possbily
    img_rgb = img
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)
    detection_result = mp_hands.detect(mp_image)

    cx, cy, area = 0, 0, 0
    bx, by, bw, bh = 0, 0, 0, 0

    if detection_result.hand_landmarks and len(detection_result.hand_landmarks) > 0:
        landmarks = detection_result.hand_landmarks[0]
        xs = [lm.x for lm in landmarks]
        ys = [lm.y for lm in landmarks]
        cx = int(landmarks[0].x * img.shape[1])
        cy = int(landmarks[0].y * img.shape[0])
        area = int((max(xs) - min(xs)) * (max(ys) - min(ys)) * img.shape[1] * img.shape[0])
        bx = int(min(xs) * img.shape[1])
        by = int(min(ys) * img.shape[0])
        bw = int((max(xs) - min(xs)) * img.shape[1])
        bh = int((max(ys) - min(ys)) * img.shape[0])
        _draw_connections(img, landmarks, img.shape[1], img.shape[0])
        _draw_landmarks(img, landmarks, img.shape[1], img.shape[0])
        cv2.rectangle(img, (bx, by), (bx + bw, by + bh), (0, 255, 0), 2)
        cv2.circle(img, (cx, cy), 12, (0, 255, 0), cv2.FILLED)
    else:
        cv2.putText(img, "NO HAND", (img.shape[1] // 2 - 60, 60),
                      cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    return img, [cx, cy, area]

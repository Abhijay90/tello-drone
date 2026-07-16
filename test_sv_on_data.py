"""Classify all data images with 2 gesture classifiers and report statistics."""
import sys
import os
import glob
import json
import csv
import numpy as np
from collections import Counter
from dataclasses import dataclass

# Add project root to path
sys.path.insert(0, '/home/abhikun/Desktop/drone')
sys.path.insert(0, '/home/abhikun/Desktop/drone/tello-drone')

from training.data_collector import GESTURE_KEYS
from vision.gestures_v2 import classify, Gesture

# ── Configuration ──
DATA_DIR    = '/home/abhikun/Desktop/drone/tello-drone/data/'
MODEL_DIR   = '/home/abhikun/Desktop/drone/tello-drone/models/'
CSV_OUTPUT  = '/home/abhikun/Desktop/drone/tello-drone/evaluation_results.csv'

# ── Load Classifiers ──
import joblib

svm_clf = joblib.load(os.path.join(MODEL_DIR, 'gesture_classifier.pkl'))
scaler = joblib.load(os.path.join(MODEL_DIR, 'scaler.pkl'))
label_encoder = joblib.load(os.path.join(MODEL_DIR, 'label_enc.pkl'))

print("[OK] All classifiers loaded. SVM model type:", type(svm_clf).__name__)

# ── Feature Extraction (must match training) ──
FEATURE_NAMES = [
    'thumb_extend_ratio', 'palm_orientation', 'palm_center_x',
    'palm_center_y', 'thumb_tips_spread', 'wrist_angle',
    'thumb_palm_distance', 'palm_size', 'extended_fingers',
    'wrist_palm_distance', 'palm_aspect_ratio', 'frame_angle',
]

def extract_features(sample):
    features = np.zeros(len(FEATURE_NAMES), dtype=np.float64)
    img_w = sample['width']
    img_h = sample['height']
    raw = sample['landmarks']
    
    # Handle both dict and tuple landmark formats
    if isinstance(raw[0], dict):
        xs = np.array([p['x'] for p in raw])
        ys = np.array([p['y'] for p in raw])
        zs = np.array([p['z'] for p in raw])
    else:
        xs = np.array([p[0] for p in raw])
        ys = np.array([p[1] for p in raw])
        zs = np.array([p[2] for p in raw])
    
    normalized = np.column_stack([xs, ys, zs]) * np.array([img_w, img_h, img_w])
    
    wrist = normalized[0]
    thumb_tip = normalized[4]
    pinky_tip = normalized[20]
    
    # 1. Thumb extend ratio
    ip = normalized[2]
    thumb_extend = np.linalg.norm(thumb_tip[:2] - ip[:2])
    box_size = max(np.ptp(xs) * img_w, np.ptp(ys) * img_h, 1.0)
    features[0] = thumb_extend / box_size
    
    # 2. Palm orientation
    middle = normalized[12]
    pinky = normalized[17]
    vec_middle = middle[:2] - wrist[:2]
    vec_pinky = pinky[:2] - wrist[:2]
    cross_z = vec_middle[0] * vec_pinky[1] - vec_middle[1] * vec_pinky[0]
    features[1] = cross_z / (box_size ** 2)
    
    # 3-4. Palm center
    features[2] = np.mean(xs)
    features[3] = np.mean(ys)
    
    # 5. Thumb-tips spread
    features[4] = np.linalg.norm(thumb_tip[:2] - pinky_tip[:2]) / box_size
    
    # 6. Wrist angle
    wrist_angle = np.arctan2(
        (normalized[12][1] + normalized[17][1]) / 2 - wrist[1],
        (normalized[12][0] + normalized[17][0]) / 2 - wrist[0]
    ) * 180 / np.pi
    features[5] = wrist_angle
    
    # 7. Thumb-palm distance
    palm_center = np.array([features[2], features[3]]) * np.array([img_w, img_h])
    features[6] = np.linalg.norm(thumb_tip[:2] - palm_center) / box_size
    
    # 8. Palm size
    features[7] = (np.ptp(xs) * np.ptp(ys)) / (img_w * img_h)
    
    # 9. Extended fingers
    extended = 0
    for tip_id, mcp_id in [(8, 5), (12, 9), (16, 13), (20, 17)]:
        tip_pt = normalized[tip_id][:2]
        mcp_pt = normalized[mcp_id][:2]
        dist = np.linalg.norm(tip_pt - mcp_pt)
        if dist / box_size > 0.3:
            extended += 1
    features[8] = extended
    
    # 10. Wrist-palm distance
    bbox_center = np.array([
        (np.ptp(xs) + np.min(xs)) / 2 * img_w,
        (np.ptp(ys) + np.min(ys)) / 2 * img_h
    ])
    features[9] = np.linalg.norm(wrist[:2] - bbox_center) / box_size
    
    # 11. Palm aspect ratio
    bbox_w = np.ptp(xs) * img_w
    bbox_h = np.ptp(ys) * img_h
    features[10] = bbox_w / max(bbox_h, 1.0)
    
    # 12. Frame angle
    features[11] = np.arctan2(bbox_h - bbox_w, bbox_h + bbox_w) * 180 / np.pi
    
    return features


# ── Collect Data Files ──
all_classes = sorted(GESTURE_KEYS.values())
file_to_gt = {}

print("\n[1] Scanning dataset...")
for cls in all_classes:
    dirpath = os.path.join(DATA_DIR, cls)
    if not os.path.isdir(dirpath):
        continue
    imgs = sorted(glob.glob(os.path.join(dirpath, 'sample_*.json')))
    if imgs:
        for p in imgs:
            file_to_gt[p] = cls
        print(f"   {cls:15s}: {len(imgs):4d} samples")

all_files = sorted(file_to_gt.keys())
total = len(all_files)
print(f"\n   Total: {total} samples across {len(all_classes)} categories")

if total == 0:
    print("\n[ERROR] No samples found. Check your data directory.")
    sys.exit(1)

# ── Lightweight MediaPipe-like landmark objects ──
@dataclass
class Landmark:
    x: float
    y: float
    z: float

def to_landmarks(lm_raw):
    if isinstance(lm_raw[0], dict):
        return [Landmark(float(p['x']), float(p['y']), float(p['z'])) for p in lm_raw]
    return [Landmark(float(p[0]), float(p[1]), float(p[2])) for p in lm_raw]

# ── Classify All Samples ──
print(f"\n[2] Classifying {total} samples with 2 methods...")
results = []

for i, sample_path in enumerate(all_files):
    i += 1
    if i % 50 == 0:
        print(f"   {i}/{total}...")
    
    # Load JSON (no need to read image for classification)
    with open(sample_path, 'r') as f:
        sample = json.load(f)
    
    img_w = sample.get('width', 640)
    img_h = sample.get('height', 480)
    
    # ── SVM Predict (extract 12 features) ──
    svm_pred = "UNKNOWN"
    if 'landmarks' in sample and sample['landmarks']:
        features = extract_features(sample)
        features = features.reshape(1, -1)
        features_scaled = scaler.transform(features)
        raw_pred = svm_clf.predict(features_scaled)[0]
        svm_pred = label_encoder.inverse_transform([raw_pred])[0]
    
    # ── V2 Distance Classify ──
    v2_pred = "NO_HAND"
    if 'landmarks' in sample and sample['landmarks']:
        mediapipe_landmarks = to_landmarks(sample['landmarks'])
        v2_gesture = classify(mediapipe_landmarks, img_w, img_h)
        v2_pred = v2_gesture.name if v2_gesture else "NO_HAND"
    
    # ── Store result ──
    gt_label = file_to_gt[sample_path]
    svm_correct = 'Yes' if svm_pred == gt_label else 'No'
    v2_correct = 'Yes' if v2_pred == gt_label else 'No'
    
    results.append({
        'File': os.path.basename(sample_path),
        'GT_Class': gt_label,
        'SVM_Pred': svm_pred,
        'V2_1D_Pred': v2_pred,
        'IsCorrect_SVM': svm_correct,
        'IsCorrect_V2': v2_correct,
    })

# ── Print Statistics ──
print(f"\n[3] Classification complete! Generating statistics...")

# ── Overall Accuracy ──
print("\n" + "="*75)
print(f"{'CLASSIFIER':<15} {'Total':>8} {'Correct':>8} {'Accuracy':>8} {'Missed as:':>40}")
print("="*75)

for col in ['SVM_Pred', 'V2_1D_Pred']:
    correct = sum(1 for r in results if r['GT_Class'] == r[col])
    total_n = len(results)
    acc = (correct / total_n * 100) if total_n else 0
    
    missed_count = Counter()
    for r in results:
        if r['GT_Class'] != r[col]:
            missed_count[r[col]] += 1
    top_mis = [f"{k}:{v}" for k, v in missed_count.most_common(4)]
    miss_str = ", ".join(top_mis) if top_mis else "None"
    
    print(f"{col:<15} {total_n:>8} {correct:>8} {acc:>7.1f}%  {miss_str:>40}")

# ── Per-Class Breakdown ──
print("\n" + "-"*75)
print("PER-CLASS ACCURACY BREAKDOWN:")
print("-"*75)
for cls in all_classes:
    cls_samples = [r for r in results if r['GT_Class'] == cls]
    if not cls_samples:
        continue
    total_n = len(cls_samples)
    correct_svm = sum(1 for r in cls_samples if r['GT_Class'] == r['SVM_Pred'])
    correct_v2 = sum(1 for r in cls_samples if r['GT_Class'] == r['V2_1D_Pred'])
    
    print(f"  {cls}:")
    print(f"    Samples={total_n} | SVM: {correct_svm}/{total_n} | V2: {correct_v2}/{total_n}")

# ── Save CSV ──
with open(CSV_OUTPUT, 'w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=results[0].keys())
    writer.writeheader()
    writer.writerows(results)
print(f"\n[OK] Full per-image results saved to: {CSV_OUTPUT}")
print("="*75)

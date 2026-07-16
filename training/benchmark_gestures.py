#!/usr/bin/env python3
"""
benchmark_gestures.py - Compare real gesture data against classifier thresholds.

Tests the gestures_v2 classify engine against actual labeled samples,
identifying confusion patterns and threshold misalignments.

Usage:
  python benchmark_gestures.py                    # run benchmark
  python benchmark_gestures.py --dir /path/to/data  # custom data directory
  python benchmark_gestures.py --summary-only       # skip details
  python benchmark_gestures.py --export-csv results.csv
"""

import json
import os
import sys
import argparse
import glob
from collections import defaultdict

import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
DEFAULT_DATA_DIR = os.path.join(PROJECT_ROOT, "data")

# Import the gesture engine
sys.path.insert(0, PROJECT_ROOT)
from vision.gestures_v2 import (
    classify, Gesture,
    THUMB_TIP_SEPARATION_BEYOND_IP,
    FINGER_TIP_SEPARATION_OPEN_RATIO,
    PALM_NORMAL_Z_THRESHOLD,
)

GESTURES = [
    'open_palm', 'closed_fist', 'thumbs_up', 'thumbs_down',
    'palm_up', 'palm_down', 'palm_left', 'palm_right'
]

# Map gesture labels to Gesture enum values
GESTURE_TO_ENUM = {
    'open_palm': Gesture.OPEN_PALM,
    'closed_fist': Gesture.CLOSED_FIST,
    'thumbs_up': Gesture.THUMBS_UP,
    'thumbs_down': Gesture.THUMBS_DOWN,
    'palm_up': Gesture.PALM_UP,
    'palm_down': Gesture.PALM_DOWN,
    'palm_left': Gesture.PALM_LEFT,
    'palm_right': Gesture.PALM_RIGHT,
}

GESTURE_NAMES = {g: g.replace('_', ' ').title() for g in GESTURES + ['unknown']}
GESTURE_NAMES['UNKNOWN'] = 'Unknown'


def load_data(data_dir):
    """Load all samples and convert landmarks to dicts for MediaPipe compatibility."""
    samples = []
    for gesture in GESTURES:
        pattern = os.path.join(data_dir, gesture, 'sample_*.json')
        fpv_pattern = os.path.join(data_dir, gesture, 'fpv_sample_*.json')
        for f in glob.glob(pattern) + glob.glob(fpv_pattern):
            data = json.load(open(f))
            data['_meta']['gesture'] = gesture
            samples.append(data)
    return samples


def landmarks_to_dict(landmarks_list):
    """Convert MediaPipe landmarks to a list of dicts compatible with classify()."""
    result = []
    for lm in landmarks_list:
        if isinstance(lm, dict):
            result.append(lm)
        else:
            result.append({
                'x': lm.x if hasattr(lm, 'x') else lm[0],
                'y': lm.y if hasattr(lm, 'y') else lm[1],
                'z': lm.z if hasattr(lm, 'z') else (lm[2] if len(lm) > 2 else 0),
            })
    return result


def benchmark(samples_data, summary_only=False):
    """Run classify() on all samples and compute confusion matrix."""
    confusion = defaultdict(lambda: defaultdict(int))
    total = 0

    # Per-gesture accuracy tracking
    per_gesture_correct = defaultdict(int)
    per_gesture_total = defaultdict(int)

    # Metric tracking per sample
    sample_metrics = []

    for sample in samples_data:
        gesture = sample['_meta']['gesture']
        landmarks = sample['landmarks']
        img_w = sample['width']
        img_h = sample['height']

        predicted = classify(landmarks, img_w, img_h)
        predicted_name = predicted.name.lower().replace('unknown_', '')
        if predicted == Gesture.UNKNOWN:
            predicted_name = 'unknown'

        true_name = gesture
        confusion[true_name][predicted_name] += 1
        total += 1
        per_gesture_total[gesture] += 1

        if predicted == GESTURE_TO_ENUM[gesture]:
            per_gesture_correct[gesture] += 1
        else:
            print(f"  MISMATCH: {true_name} -> classified as {predicted_name}")

        # Store metrics for analysis
        metrics = collect_metrics(landmarks, img_w, img_h)
        sample_metrics.append({
            'gesture': true_name,
            'predicted': predicted_name,
            'correct': predicted == GESTURE_TO_ENUM[true_name],
            'metrics': metrics,
        })

    # Global accuracy
    global_correct = sum(per_gesture_correct.values())
    accuracy = global_correct / total if total > 0 else 0

    return confusion, per_gesture_correct, per_gesture_total, accuracy, sample_metrics


def collect_metrics(landmarks, img_w, img_h):
    """Collect raw metrics for threshold analysis."""
    m = {}

    raw = landmarks_to_dict(landmarks)
    pts = [{**d, 'x': d['x'], 'y': d['y'], 'z': d['z']} for d in raw]

    xs = [p['x'] for p in pts]
    ys = [p['y'] for p in pts]
    box_w = max(xs) - min(xs)
    box_h = max(ys) - min(ys)
    box_w = max(box_w, 1.0)
    box_h = max(box_h, 1.0)

    # Thumb extend
    tip_x = raw[4]['x'] * img_w
    tip_y = raw[4]['y'] * img_h
    ip_x = raw[2]['x'] * img_w
    ip_y = raw[2]['y'] * img_h
    tip_to_ip = ((tip_x-ip_x)**2 + (tip_y-ip_y)**2)**0.5
    ref = max(box_w, box_h)
    m['thumb_extend'] = tip_to_ip / ref if ref > 0 else 0

    # Palm orientation
    wrist_x = raw[0]['x'] * img_w
    wrist_y = raw[0]['y'] * img_h
    idx_x = raw[5]['x'] * img_w
    idx_y = raw[5]['y'] * img_h
    pinky_x = raw[17]['x'] * img_w
    pinky_y = raw[17]['y'] * img_h
    cross = (idx_x-wrist_x)*(pinky_y-wrist_y) - (idx_y-wrist_y)*(pinky_x-wrist_x)
    m['palm_orient'] = cross / max(box_w * box_h, 1.0)

    # Palm center
    cx = sum(p['x'] for p in pts) / len(pts)
    cy = sum(p['y'] for p in pts) / len(pts)
    m['palm_cx'] = cx
    m['palm_cy'] = cy

    # Extended fingers
    extended = 0
    for tip_id, mcp_id in [(8,5),(12,9),(16,13),(20,17)]:
        tx = raw[tip_id]['x'] * img_w
        ty = raw[tip_id]['y'] * img_h
        mx = raw[mcp_id]['x'] * img_w
        my = raw[mcp_id]['y'] * img_h
        dist = ((tx-mx)**2 + (ty-my)**2)**0.5
        if dist / max(box_w, box_h) > 0.45:
            extended += 1
    m['extended_fingers'] = extended

    return m


def print_confusion_matrix(scripts_confusion, total_samples):
    """Print formatted confusion matrix."""
    print("\n" + "=" * 70)
    print("  CONFUSION MATRIX")
    print("  (rows = true label, columns = predicted label)")
    print("=" * 70 + "\n")

    # Header row
    max_width = max(len(GESTURE_NAMES.get(g, g)) for g in GESTURES)
    max_width = max(max_width, len('predicted'))
    header = "  " + " " * max_width
    for pred in GESTURES + ['unknown']:
        header += f" {pred:>^{max_width}}"
    header += f" {'Total':>{max_width}}"
    print(header)
    print("  " + " " * (max_width * 2) + "+" + "─" * (len(header) - max_width * 2 - 1))

    # Data rows
    row_width = max_width * 2 + 2
    for gesture in GESTURES:
        row = f"  {GESTURE_NAMES.get(gesture, gesture):>{max_width}}"
        total_for = 0
        for pred in GESTURES + ['unknown']:
            count = scripts_confusion[gesture][pred]
            total_for += count
            row += f" {count:>{max_width}}"
        row += f" {total_for:>{max_width}}"
        print(row)

    # Total row
    col_totals = {}
    for pred in GESTURES + ['unknown']:
        col_totals[pred] = sum(scripts_confusion[g][pred] for g in GESTURES)
    row = "  " + " " * max_width
    for pred in GESTURES + ['unknown']:
        row += f" {col_totals[pred]:>{max_width}}"
    row += f" {total_samples:>{max_width}}"
    print(row + "\n")


def analyze_thresholds(sample_metrics):
    """Look closely at threshold overlaps causing misclassifications."""
    print("\n" + "=" * 70)
    print("  THRESHOLD OVERLAP ANALYSIS")
    print("  (showing metrics where classifications may be ambiguous)")
    print("=" * 70)

    print(f"\n  Current thresholds:")
    print(f"    thumb_extend:    > {THUMB_TIP_SEPARATION_BEYOND_IP} (thumb extend)")
    print(f"    palm_orient:     > {PALM_NORMAL_Z_THRESHOLD} = UP, < -{PALM_NORMAL_Z_THRESHOLD} = DOWN")
    print(f"    palm_cx:         < 0.4 = LEFT, > 0.6 = RIGHT")
    print(f"    ext_fingers:     > 0.45 = extended (x{FINGER_TIP_SEPARATION_OPEN_RATIO} ratio)")

    # Group misclassifications by type
    errors = [m for m in sample_metrics if not m['correct']]
    if not errors:
        print("\n  [OK] No misclassifications found! All thresholds aligned. \n")
        return

    print(f"\n  Total errors: {len(errors)} / {len(sample_metrics)} ({len(errors)/len(sample_metrics)*100:.1f}%)\n")

    # Show the most confusing gesture pairs
    error_pairs = defaultdict(int)
    for e in errors:
        true_g = e['gesture']
        pred_g = e['predicted']
        error_pairs[(true_g, pred_g)] += 1

    print("  Most confusing pairs:")
    for (true_g, pred_g), count in sorted(error_pairs.items(), key=lambda x: -x[1]):
        count_true = sum(1 for m in sample_metrics if m['gesture'] == true_g)
        pct = count / count_true * 100 if count_true else 0
        print(f"    {GESTURE_NAMES.get(true_g):15s} -> {GESTURE_NAMES.get(pred_g):15s} ({count} times, {pct:.0f}% of true)")

    # Show metric distributions for top confusion pairs
    print("\n  Metric values for confused samples:")
    for (true_g, pred_g), _ in list(error_pairs.items())[:5]:
        print(f"\n    {GESTURE_NAMES.get(true_g):15s} samples misclassified as {GESTURE_NAMES.get(pred_g):15s}:")
        confused = [m for m in errors if m['gesture'] == true_g and m['predicted'] == pred_g]
        if confused:
            thumb_vals = [m['metrics']['thumb_extend'] for m in confused]
            palm_vals = [m['metrics']['palm_orient'] for m in confused]
            cx_vals = [m['metrics']['palm_cx'] for m in confused]
            print(f"      thumb_extend:    {np.mean(thumb_vals):.4f} ± {np.std(thumb_vals):.4f} (thresh={THUMB_TIP_SEPARATION_BEYOND_IP})")
            print(f"      palm_orient:     {np.mean(palm_vals):.6f} ± {np.std(palm_vals):.6f} (thresh={PALM_NORMAL_Z_THRESHOLD})")
            print(f"      palm_cx:         {np.mean(cx_vals):.4f} ± {np.std(cx_vals):.4f} (LEFT<0.4, RIGHT>0.6)")

        # Compare with correct samples of the same true gesture
        correct_same = [m for m in sample_metrics if m['gesture'] == true_g and m['correct']]
        if correct_same:
            thumb_correct = [m['metrics']['thumb_extend'] for m in correct_same]
            palm_correct = [m['metrics']['palm_orient'] for m in correct_same]
            cx_correct = [m['metrics']['palm_cx'] for m in correct_same if 'palm_cx' in m['metrics']]
            print(f"\n    {GESTURE_NAMES.get(true_g):15s} CORRECT samples:")
            print(f"      thumb_extend:    {np.mean(thumb_correct):.4f} ± {np.std(thumb_correct):.4f}")
            print(f"      palm_orient:     {np.mean(palm_correct):.6f} ± {np.std(palm_correct):.6f}")
            if cx_correct:
                print(f"      palm_cx:         {np.mean(cx_correct):.4f} ± {np.std(cx_correct):.4f}")


def per_gesture_accuracy(per_gesture_correct, per_gesture_total, total_samples):
    """Print per-gesture accuracy."""
    print("\n" + "=" * 70)
    print("  PER-GESTURE ACCURACY")
    print("=" * 70)
    print(f"\n  Overall accuracy: {sum(per_gesture_correct.values())}/{total_samples} = {sum(per_gesture_correct.values())/total_samples*100:.1f}%\n")
    print(f"  {'Gesture':<20} {'Correct':>8} {'Total':>6} {'':>4} {'Accuracy':>10}")
    print(f"  {'─' * 20} {'─' * 8} {'─' * 6} {'':>4} {'─' * 10}")

    for g in GESTURES:
        correct = per_gesture_correct.get(g, 0)
        total = per_gesture_total.get(g, 0)
        acc = correct / total * 100 if total > 0 else 0
        bar = '█' * int(acc/5) + '░' * (20 - int(acc/5))
        print(f"  {GESTURE_NAMES.get(g, g):<20} {correct:>8} {total:>6} {'':>4} [{bar}] {acc:>6.1f}%")


def export_csv(scripts_metrics, filepath):
    """Export sample metrics to CSV."""
    import csv
    headers = ['gesture', 'predicted', 'correct', 'thumb_extend', 'palm_orient', 'palm_cx', 'palm_cy', 'extended_fingers']
    with open(filepath, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        for m in scripts_metrics:
            writer.writerow([
                m['gesture'], m['predicted'], m['correct'],
                m['metrics']['thumb_extend'], m['metrics']['palm_orient'],
                m['metrics']['palm_cx'], m['metrics']['palm_cy'],
                m['metrics']['extended_fingers']
            ])
    print(f"\n  [INFO] Metrics exported to: {filepath}")


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark gestures_v2 classifier against real data")
    parser.add_argument('--dir', type=str, default=None,
                        help="Data directory")
    parser.add_argument('--summary-only', action='store_true',
                        help="Only show overall accuracy, skip details")
    parser.add_argument('--export-csv', type=str, default=None,
                        help="Export metrics to CSV file")
    args = parser.parse_args()
    print()

    data_dir = args.dir if args.dir else DEFAULT_DATA_DIR
    if not os.path.isdir(data_dir):
        print(f"[ERR] Data directory not found: {data_dir}")
        sys.exit(1)

    samples_data = load_data(data_dir)
    total = len(samples_data)
    print(f"Benchmarking {total} samples from {data_dir}")
    print(f"Thresholds: thumb_extend>{THUMB_TIP_SEPARATION_BEYOND_IP}, "
          f"palm_orient>{PALM_NORMAL_Z_THRESHOLD}, "
          f"palm_cx: (<0.4 LEFT, >0.6 RIGHT), "
          f"ext_ratio>{FINGER_TIP_SEPARATION_OPEN_RATIO}\n")

    confusion, per_gesture_correct, per_gesture_total, accuracy, metrics = benchmark(samples_data, args.summary_only)

    print_confusion_matrix(confusion, total)
    per_gesture_accuracy(per_gesture_correct, per_gesture_total, total)

    if not args.summary_only:
        analyze_thresholds(metrics)

    if args.export_csv:
        export_csv(metrics, args.export_csv)

    print(f"\n[INFO] Benchmark complete. Overall accuracy: {accuracy*100:.1f}%")


if __name__ == '__main__':
    main()

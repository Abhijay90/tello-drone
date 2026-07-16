#!/usr/bin/env python3
"""
analyze_data.py - Analyze gesture dataset and visualize metric distributions.

Loads all JSON samples and computes gesture-engine metrics for visualization.

Usage:
  python analyze_data.py                          # interactive mode
  python analyze_data.py --export-png             # saves all plots as PNG
  python analyze_data.py --dir /path/to/data      # custom data directory
  python analyze_data.py --no-interactive         # skip input, run headless
"""

import json
import os
import sys
import argparse
import glob
import struct
import matplotlib
matplotlib.use('Agg')
from dataclasses import dataclass
from pathlib import Path

import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
DEFAULT_DATA_DIR = os.path.join(PROJECT_ROOT, "data")

GESTURES = [
    'open_palm', 'closed_fist', 'thumbs_up', 'thumbs_down',
    'palm_up', 'palm_down', 'palm_left', 'palm_right'
]

# MediaPipe landmark indices
WRIST = 0
THUMB_CMC = 1
THUMB_IP = 2
THUMB_TIP = 4
INDEX_MCP = 5
INDEX_PIP = 6
INDEX_DIP = 7
INDEX_TIP = 8
MIDDLE_MCP = 9
MIDDLE_PIP = 10
MIDDLE_DIP = 11
MIDDLE_TIP = 12
RING_MCP = 13
RING_PIP = 14
RING_DIP = 15
RING_TIP = 16
PINKY_MCP = 17
PINKY_PIP = 18
PINKY_DIP = 19
PINKY_TIP = 20


def decode_frame(hex_str):
    """Decode hex BGR frame string to numpy array [H, W, 3]."""
    raw = bytes.fromhex(hex_str)
    arr = np.frombuffer(raw, dtype=np.uint8)
    return arr.reshape(-1, 480, 3)


def compute_metrics(landmarks, img_w=640, img_h=480):
    """Compute gesture-engine metrics from a raw landmark list."""
    m = {}

    # Normalize landmarks to [0,1]
    pts = []
    for lm in landmarks:
        pts.append((lm['x'], lm['y'], lm['z']))
    normalized = pts

    # Box dims
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    box_w = max(xs) - min(xs)
    box_h = max(ys) - min(ys)
    box_w = max(box_w, 1.0)
    box_h = max(box_h, 1.0)

    # Thumb extend ratio (thumb tip-IP / box max dimension)
    tip_x = landmarks[THUMB_TIP]['x'] * img_w
    tip_y = landmarks[THUMB_TIP]['y'] * img_h
    ip_x = landmarks[THUMB_IP]['x'] * img_w
    ip_y = landmarks[THUMB_IP]['y'] * img_h
    tip_to_ip = ((tip_x - ip_x)**2 + (tip_y - ip_y)**2)**0.5
    ref = max(box_w, box_h)
    m['thumb_extend_ratio'] = tip_to_ip / ref if ref > 0 else 0

    # Palm orientation (cross product of wrist->index and wrist->pinky)
    wrist_x = landmarks[0]['x'] * img_w
    wrist_y = landmarks[0]['y'] * img_h
    idx_x = landmarks[INDEX_MCP]['x'] * img_w
    idx_y = landmarks[INDEX_MCP]['y'] * img_h
    pinky_x = landmarks[PINKY_MCP]['x'] * img_w
    pinky_y = landmarks[PINKY_MCP]['y'] * img_h
    vec1_x = idx_x - wrist_x
    vec1_y = idx_y - wrist_y
    vec2_x = pinky_x - wrist_x
    vec2_y = pinky_y - wrist_y
    scale = max(box_w * box_h, 1.0)
    m['palm_orientation'] = (vec1_x * vec2_y - vec1_y * vec2_x) / scale

    # Palm center x (normalized)
    cx = sum(p[0] for p in pts) / len(pts)
    cy = sum(p[1] for p in pts) / len(pts)
    m['palm_center_x'] = cx
    m['palm_center_y'] = cy

    # Thumb center in image coords
    cx_img = cx * img_w
    cy_img = cy * img_h
    m['thumb_center'] = (cx_img, cy_img)

    # Finger extension counts
    m['extended_fingers'] = compute_extended_fingers(landmarks, img_w, img_h, (box_w, box_h))

    # Thumb tip vs IP in pixel coords
    m['thumb_tip_y'] = tip_y
    m['thumb_ip_y'] = ip_y
    m['thumb_points_down'] = tip_y > ip_y + 80

    # Thumb side
    min_x = min(p[0] for p in pts)
    max_x = max(p[0] for p in pts)
    thumb_x = landmarks[THUMB_TIP]['x']
    m['thumb_side'] = 'right' if thumb_x > max_x * 0.7 else ('left' if thumb_x < min_x * 1.3 else 'middle')

    # Finger spread (avg distance between fingertip and MCP)
    finger_spreads = []
    for tip_id, mcp_id in [(8,5),(12,9),(16,13),(20,17)]:
        tx = landmarks[tip_id]['x'] * img_w
        ty = landmarks[tip_id]['y'] * img_h
        mx = landmarks[mcp_id]['x'] * img_w
        my = landmarks[mcp_id]['y'] * img_h
        spread = ((tx-mx)**2 + (ty-my)**2)**0.5
        finger_spreads.append(spread)
    m['finger_spread_x'] = np.mean(finger_spreads[:2])  # index + middle

    # Box area for normalization reference
    m['box_area'] = box_w * box_h

    return m


def compute_extended_fingers(landmarks, img_w, img_h, box_dims):
    """Count extended fingers (tip-MCP distance > 45% of box dimension)."""
    THRESH = 0.45
    extended = 0
    for tip_id, mcp_id in [(8,5),(12,9),(16,13),(20,17)]:
        tx = landmarks[tip_id]['x'] * img_w
        ty = landmarks[tip_id]['y'] * img_h
        mx = landmarks[mcp_id]['x'] * img_w
        my = landmarks[mcp_id]['y'] * img_h
        dist = ((tx-mx)**2 + (ty-my)**2)**0.5
        ref = max(box_dims[0], box_dims[1])
        if dist / ref > THRESH:
            extended += 1
    return extended


def load_data(data_dir):
    """Load all samples from the data directory."""
    all_samples = []
    for gesture in GESTURES:
        pattern = os.path.join(data_dir, gesture, 'sample_*.json')
        fpv_pattern = os.path.join(data_dir, gesture, 'fpv_sample_*.json')
        files = glob.glob(pattern) + glob.glob(fpv_pattern)
        for f in files:
            data = json.load(open(f))
            if '_meta' not in data:
                data['_meta'] = {}
            data['_meta']['gesture'] = gesture
            data['_meta']['file'] = os.path.basename(f)
            data['_meta']['path'] = f
            all_samples.append(data)
    print(f"Loaded {len(all_samples)} samples from {data_dir}")
    return all_samples


def print_statistics(scripts_data):
    """Print summary statistics for each gesture."""
    print("\n" + "=" * 70)
    print("  GESTURE DATASET STATISTICS")
    print("=" * 70)

    counts = {}
    for gesture in GESTURES:
        samples = [d for d in scripts_data if d['_meta']['gesture'] == gesture]
        counts[gesture] = len(samples)
        if not samples:
            print(f"\n  {gesture}: 0 samples (SKIPPED)")
            continue

        metrics = [compute_metrics(s['landmarks'], s['width'], s['height']) for s in samples]

        thumb_ratios = [m['thumb_extend_ratio'] for m in metrics]
        palm_orient = [m['palm_orientation'] for m in metrics]
        palm_cxs = [m['palm_center_x'] for m in metrics]
        ext_fingers = [m['extended_fingers'] for m in metrics]

        print(f"\n  {gesture.upper()} ({len(samples)} samples)")
        print(f"  {'─' * 50}")
        print(f"  Thumb extend ratio: {np.mean(thumb_ratios):.4f} ± {np.std(thumb_ratios):.4f}  [min={np.min(thumb_ratios):.4f}, max={np.max(thumb_ratios):.4f}]")
        print(f"  Palm orientation:   {np.mean(palm_orient):.6f} ± {np.std(palm_orient):.6f}  [min={np.min(palm_orient):.6f}, max={np.max(palm_orient):.6f}]")
        print(f"  Palm center X:      {np.mean(palm_cxs):.4f} ± {np.std(palm_cxs):.4f}  [min={np.min(palm_cxs):.4f}, max={np.max(palm_cxs):.4f}]")
        print(f"  Expanded fingers:   {np.mean(ext_fingers):.1f} ± {np.std(ext_fingers):.1f}  [range={np.min(ext_fingers)}-{np.max(ext_fingers)} of 4]")

        # Thumb direction
        dir_counts = {}
        for m in metrics:
            d = m['thumb_side']
            dir_counts[d] = dir_counts.get(d, 0) + 1
        thumb_dist = ', '.join(f'{k}:{v}' for k,v in sorted(dir_counts.items()))
        print(f"  Thumb side:         {thumb_dist}")

        # Thumb points down
        down_pct = sum(1 for m in metrics if m['thumb_points_down']) / len(metrics)
        print(f"  Thumb points down:  {down_pct*100:.0f}% of samples")

        # Per-sample thumb extend ratio histogram
        if len(samples) <= 200:
            print(f"\n  Per-sample thumb extend ratios:")
            for i, (s, m) in enumerate(zip(samples, metrics)):
                thumb_ratio_str = f"{m['thumb_extend_ratio']:.4f}"
                print(f"    {i+1:2d}. {thumb_ratio_str}")


def create_visualizations(scripts_data, export_png=False):
    """Create matplotlib visualizations of the data."""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import matplotlib.gridspec as gridspec
    except ImportError:
        print("[WARN] matplotlib not available for visualizations")
        return

    fig = plt.figure(figsize=(20, 16))
    gs = gridspec.GridSpec(4, 2, hspace=0.35, wspace=0.3)

    colors = {
        'open_palm': '#FF6B6B',
        'closed_fist': '#4A47A3',
        'thumbs_up': '#7BC67E',
        'thumbs_down': '#FFA15A',
        'palm_up': '#9BC53D',
        'palm_down': '#4ECDC4',
        'palm_left': '#A855F7',
        'palm_right': '#EC4899',
    }

    gesture_indices = {g: i for i, g in enumerate(GESTURES)}

    # Plot 1: Thumb extend ratio by gesture
    ax1 = fig.add_subplot(gs[0, :])
    x_pos = np.arange(len(GESTURES))
    for gesture in GESTURES:
        samples = [d for d in scripts_data if d['_meta']['gesture'] == gesture]
        metrics_list = [compute_metrics(s['landmarks'], s['width'], s['height']) for s in samples]
        ratios = [m['thumb_extend_ratio'] for m in metrics_list]
        if ratios:
            ax1.plot([gesture_indices[gesture]]*len(ratios), ratios,
                     'o', color=colors[gesture], markersize=5, alpha=0.6, label=gesture)
    ax1.axhline(y=0.35, color='red', linestyle='--', linewidth=1, alpha=0.7, label='THUMB threshold=0.35')
    ax1.set_xlabel('Gesture')
    ax1.set_ylabel('Thumb Extend Ratio')
    ax1.set_title('Thumb Extend Ratio by Gesture')
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels([g.replace('_',' ').title() for g in GESTURES], rotation=45, ha='right')
    ax1.legend(fontsize=7, ncol=2, loc='upper right')
    ax1.grid(True, alpha=0.3)

    # Plot 2: Palm orientation by gesture
    ax2 = fig.add_subplot(gs[1, :])
    hue_list = []
    orient_list = []
    for gesture in GESTURES:
        samples = [d for d in scripts_data if d['_meta']['gesture'] == gesture]
        metrics_list = [compute_metrics(s['landmarks'], s['width'], s['height']) for s in samples]
        orientations = [m['palm_orientation'] for m in metrics_list]
        hue_list.extend([gesture] * len(orientations))
        orient_list.extend(orientations)

    for gesture in GESTURES:
        samples = [d for d in scripts_data if d['_meta']['gesture'] == gesture]
        metrics_list = [compute_metrics(s['landmarks'], s['width'], s['height']) for s in samples]
        orientations = [m['palm_orientation'] for m in metrics_list]
        ax2.plot([gesture_indices[gesture]]*len(orientations), orientations,
                 'o', color=colors[gesture], markersize=5, alpha=0.6, label=gesture)
    ax2.axhline(y=0.02, color='green', linestyle='--', linewidth=1, alpha=0.5, label='Palm UP threshold=+0.02')
    ax2.axhline(y=-0.02, color='orange', linestyle='--', linewidth=1, alpha=0.5, label='Palm DOWN threshold=-0.02')
    ax2.set_xlabel('Gesture')
    ax2.set_ylabel('Palm Orientation (cross-product scale)')
    ax2.set_title('Palm Orientation by Gesture')
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels([g.replace('_',' ').title() for g in GESTURES], rotation=45, ha='right')
    ax2.legend(fontsize=7, ncol=2, loc='upper right')
    ax2.grid(True, alpha=0.3)

    # Plot 3: Palm center X by gesture
    ax3 = fig.add_subplot(gs[2, :])
    for i, gesture in enumerate(GESTURES):
        samples = [d for d in scripts_data if d['_meta']['gesture'] == gesture]
        metrics_list = [compute_metrics(s['landmarks'], s['width'], s['height']) for s in samples]
        cx = [m['palm_center_x'] for m in metrics_list]
        if cx:
            ax3.bar([i]*len(cx), cx, alpha=0.6, color=colors[gesture], label=gesture)
    ax3.axvline(x=0.4, color='red', linestyle='--', linewidth=1, alpha=0.5, label='PALM LEFT threshold=0.4')
    ax3.axvline(x=0.6, color='red', linestyle='--', linewidth=1, alpha=0.5, label='PALM RIGHT threshold=0.6')
    ax3.set_xlabel('Gesture')
    ax3.set_ylabel('Palm Center X (normalized)')
    ax3.set_title('Palm Center X by Gesture')
    ax3.set_xticks(range(len(GESTURES)))
    ax3.set_xticklabels([g.replace('_',' ').title() for g in GESTURES], rotation=45, ha='right')
    ax3.legend(fontsize=7, ncol=2, loc='upper right')
    ax3.grid(True, alpha=0.3)

    # Plot 4: Extended fingers distribution
    ax4 = fig.add_subplot(gs[3, 0])
    for i, gesture in enumerate(GESTURES):
        samples = [d for d in scripts_data if d['_meta']['gesture'] == gesture]
        metrics_list = [compute_metrics(s['landmarks'], s['width'], s['height']) for s in samples]
        ext = [m['extended_fingers'] for m in metrics_list]
        counts = [sum(1 for e in ext if e == k) for k in range(5)]
        ax4.bar(range(5), counts, color=colors[gesture], alpha=0.5, label=gesture[:3])
    ax4.set_xlabel('Extended Fingers')
    ax4.set_ylabel('Count')
    ax4.set_title('Extended Finger Count Distribution')
    ax4.set_xticks(range(5))
    ax4.legend(fontsize=6, ncol=4)

    # Plot 5: Gesture count bar chart
    ax5 = fig.add_subplot(gs[3, 1])
    counts = []
    for gesture in GESTURES:
        samples = [d for d in scripts_data if d['_meta']['gesture'] == gesture]
        counts.append(len(samples))
    ax5.bar([g.replace('_',' ').title() for g in GESTURES], counts, color=[colors[g] for g in GESTURES])
    ax5.set_title('Samples per Gesture')
    ax5.set_ylabel('Count')

    # Save or show
    if export_png or os.environ.get('MPLBACKEND') == 'Agg':
        out_dir = os.path.join(os.path.dirname(__file__), 'output')
        os.makedirs(out_dir, exist_ok=True)
        save_path = os.path.join(out_dir, 'gesture_metrics_analysis.png')
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"\n[INFO] Visualization saved to: {save_path}")
        plt.close('all')
    else:
        plt.show()

    return save_path if export_png else None


def interactive_mode(scripts_data):
    """Allow user to explore individual samples."""
    print("\n" + "=" * 70)
    print("  INTERACTIVE SAMPLE EXPLORER")
    print("=" * 70)
    print("  'n' = next sample, 'p' = previous sample, 'q' = quit")
    # Use simple line-based exploration
    idx = 0
    while True:
        if idx < 0 or idx >= len(scripts_data):
            print("  [INFO] End of dataset.")
            break
        sample = scripts_data[idx]
        gesture = sample['_meta']['gesture']
        metrics = compute_metrics(sample['landmarks'], sample['width'], sample['height'])
        print(f"\n  Sample {idx+1}/{len(scripts_data)}: {gesture}")
        print(f"    Thumb extend ratio:   {metrics['thumb_extend_ratio']:.4f}")
        print(f"    Palm orientation:     {metrics['palm_orientation']:.6f}")
        print(f"    Palm center X:        {metrics['palm_center_x']:.4f}")
        print(f"    Palm center Y:        {metrics['palm_center_y']:.4f}")
        print(f"    Extended fingers:     {metrics['extended_fingers']} / 4")
        print(f"    Thumb side:           {metrics['thumb_side']}")
        print(f"    Thumb points down:    {metrics['thumb_points_down']}")
        inp = input("  [n]ext / [p]rev / [q]uit > ").strip().lower()
        if inp == 'q':
            break
        elif inp == 'n':
            idx += 1
        elif inp == 'p':
            idx -= 1


def main():
    parser = argparse.ArgumentParser(
        description="Analyze gesture dataset: statistics, distributions, and metrics")
    parser.add_argument('--dir', type=str, default=None,
                        help="Data directory (default: data/ in project root)")
    parser.add_argument('--export-png', action='store_true',
                        help="Save visualizations as PNG (no interactive window)")
    parser.add_argument('--no-interactive', action='store_true',
                        help="Skip interactive explorer")
    args = parser.parse_args()
    print()

    data_dir = args.dir if args.dir else DEFAULT_DATA_DIR
    if not os.path.isdir(data_dir):
        print(f"[ERR] Data directory not found: {data_dir}")
        sys.exit(1)

    # Load data
    scripts_data = load_data(data_dir)
    export_png = args.export_png or args.no_interactive or os.environ.get('MPLBACKEND') == 'Agg'

    # Statistics
    print_statistics(scripts_data)

    # Visualizations
    create_visualizations(scripts_data, export_png=export_png)

    # Interactive explorer
    if not args.no_interactive and not args.export_png and not os.environ.get('MPLBACKEND'):
        try:
            interactive_mode(scripts_data)
        except (KeyboardInterrupt, EOFError):
            pass

    print("\n[INFO] Done.")


if __name__ == '__main__':
    main()

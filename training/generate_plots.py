"""Generate visualization charts for Tello drone gesture dataset."""

import os
import json
import sys
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

import numpy as np

# Must be set before importing pyplot
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
DATA_DIR = os.path.join(PROJECT_ROOT, "data")

GESTURES = [
    'open_palm', 'closed_fist', 'thumbs_up', 'thumbs_down',
    'palm_up', 'palm_down', 'palm_left', 'palm_right',
]
GESTURE_COLORS = {
    'open_palm': '#FF6B6B',
    'closed_fist': '#4A47A3',
    'thumbs_up': '#7BC67E',
    'thumbs_down': '#FFD166',
    'palm_up': '#06D6A0',
    'palm_down': '#118AB2',
    'palm_left': '#EF476F',
    'palm_right': '#073B4C',
}
FEATURE_NAMES = [
    'thumb_extend_ratio', 'palm_orientation', 'palm_center_x',
    'palm_center_y', 'thumb_tips_spread', 'wrist_angle',
    'thumb_palm_distance', 'palm_size',
]


def load_all_samples():
    """Load all gesture samples from nested data directories."""
    samples = []
    for gesture_dir in sorted(Path(DATA_DIR).iterdir()):
        if gesture_dir.is_dir():
            for json_file in sorted(gesture_dir.glob("*.json")):
                with open(json_file) as fp:
                    data = json.load(fp)
                if isinstance(data, list):
                    samples.extend(data)
                elif isinstance(data, dict) and 'samples' in data:
                    samples.extend(data['samples'])
                else:
                    samples.append(data)
    print(f"Loaded {len(samples)} samples from {DATA_DIR}")
    return samples


def extract_features(samples):
    """Return all feature dicts and labels."""
    features = []
    labels = []
    for s in samples:
        meta = s.get('_meta', {})
        feat = s.get('features', {})
        for k in FEATURE_NAMES:
            if k not in feat:
                feat[k] = meta.get(k, 0.0)
        features.append(feat)
        labels.append(meta.get('gesture', 'unknown'))
    return features, labels


def compute_correlation_matrix(samples):
    """Compute Pearson correlation between features."""
    features, _ = extract_features(samples)
    n = len(FEATURE_NAMES)
    mat = np.zeros((n, n))

    for i, f1 in enumerate(FEATURE_NAMES):
        for j, f2 in enumerate(FEATURE_NAMES):
            v1 = np.array([s[f1] for s in features])
            v2 = np.array([s[f2] for s in features])
            denom = np.std(v1) * np.std(v2)
            if denom and np.isfinite(denom):
                mat[i, j] = np.corrcoef(v1, v2)[0, 1]
            else:
                mat[i, j] = 0.0
    return mat


def plot_correlation_heatmap(corr, output_path):
    """Plot feature correlation heatmap."""
    fig, ax = plt.subplots(figsize=(12, 9))
    cmap = LinearSegmentedColormap.from_list('corr', ['#2b2d42', '#ffffff', '#8d9f87'], N=256)
    ax.imshow(corr, cmap=cmap, vmin=-1, vmax=1)
    ax.set_xticks(range(len(FEATURE_NAMES)))
    ax.set_yticks(range(len(FEATURE_NAMES)))
    ax.set_xticklabels([f[:14].replace('_', '\n') for f in FEATURE_NAMES], rotation=45, ha='right')
    ax.set_yticklabels(FEATURE_NAMES)
    for i in range(len(FEATURE_NAMES)):
        for j in range(len(FEATURE_NAMES)):
            ax.text(j, i, f'{corr[i, j]:.2f}', ha='center', va='center',
                    fontsize=9, color='white' if abs(corr[i, j]) > 0.5 else '#2b2d42')
    ax.set_title('Feature Correlation Matrix', fontsize=14, fontweight='bold', pad=20)
    plt.colorbar(ax.images[0], ax=ax, label='Pearson correlation', shrink=0.8)
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"[OK] Correlation heatmap saved")


def plot_distribution_boxplots(samples, output_path):
    """Plot boxplots of key features per gesture."""
    features, labels = extract_features(samples)
    fig = plt.figure(figsize=(18, 12))
    gs = gridspec.GridSpec(2, 3, hspace=0.35, wspace=0.3)

    key_features = [
        'thumb_extend_ratio', 'palm_orientation', 'palm_center_x',
        'thumb_tips_spread', 'wrist_angle', 'palm_size'
    ]
    feature_labels = [
        'Thumb Extend Ratio', 'Palm Orientation', 'Palm Center X',
        'Thumb Tips Spread', 'Wrist Angle', 'Palm Size',
    ]

    active = [g for g in GESTURES if labels.count(g) > 0]

    for idx, (feat, label) in enumerate(zip(key_features, feature_labels)):
        ax = fig.add_subplot(gs[idx])
        data = []
        names = []
        for g in active:
            vals = [features[j][feat] for j, l in enumerate(labels) if l == g]
            if vals:
                data.append(vals)
                names.append(g.replace('_', ' ').title())
        if not data:
            continue
        bp = ax.boxplot(data, patch_artist=True, widths=0.5)
        for patch, col in zip(bp['boxes'], [GESTURE_COLORS[g] for g in active]):
            patch.set_facecolor(col)
            patch.set_alpha(0.6)
        ax.set_xticklabels(names, rotation=30, ha='right', fontsize=7)
        ax.set_ylabel(label, fontsize=8)
        ax.set_title(f'{label} by Gesture', fontsize=10, fontweight='bold')
        ax.grid(True, alpha=0.2, axis='y')

    fig.suptitle('Feature Distributions by Gesture', fontsize=16, fontweight='bold', y=1.02)
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"[OK] Distribution boxplots saved")


def plot_decision_boundaries(samples, output_path):
    """Plot 2D feature pair scatter plots."""
    features, labels = extract_features(samples)
    fig = plt.figure(figsize=(18, 10))
    gs = gridspec.GridSpec(2, 3, hspace=0.35, wspace=0.3)

    pairs = [
        ('thumb_extend_ratio', 'palm_orientation', 'T.E. Ratio', 'Palm Orientation'),
        ('palm_center_x', 'palm_center_y', 'Palm X', 'Palm Y'),
        ('thumb_extend_ratio', 'thumb_tips_spread', 'T.E. Ratio', 'Tips Spread'),
        ('palm_orientation', 'palm_size', 'Palm Orientation', 'Palm Size'),
        ('wrist_angle', 'thumb_palm_distance', 'Wrist Angle', 'Thumb-Palm Dist'),
        ('thumb_tips_spread', 'palm_size', 'Tips Spread', 'Palm Size'),
    ]

    active = [g for g in GESTURES if labels.count(g) > 0]

    for idx, (xf, yf, xl, yl) in enumerate(pairs):
        ax = fig.add_subplot(gs[idx])
        for g in active:
            xv = [features[j][xf] for j, l in enumerate(labels) if l == g]
            yv = [features[j][yf] for j, l in enumerate(labels) if l == g]
            if xv and yv:
                ax.scatter(xv, yv, alpha=0.5, s=30, color=GESTURE_COLORS[g],
                           label=g.replace('_', ' ').title())
        ax.set_xlabel(xl); ax.set_ylabel(yl)
        ax.set_title(f'{xl} vs {yl}', fontsize=11, fontweight='bold')
        ax.legend(fontsize=7, loc='best', ncol=2)
        ax.grid(True, alpha=0.2)

    fig.suptitle('2D Feature Projections by Gesture', fontsize=16, fontweight='bold', y=1.02)
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"[OK] Decision boundary plots saved")


def plot_feature_importance(samples, output_path):
    """Plot feature importance by class separability."""
    features, labels = extract_features(samples)
    importances = []
    classes = [g for g in GESTURES if labels.count(g) > 0]
    for feat in FEATURE_NAMES:
        vals = np.array([features[j][feat] for j, l in enumerate(labels)])
        g_std = np.std(vals)
        c_stds = [np.std(vals[np.array(labels) == c]) for c in classes]
        intra = np.mean(c_stds) ** 2 if c_stds else 1
        imp = g_std / (intra ** 0.5) if intra > 0 and np.isfinite(intra) else 0
        importances.append(max(0, imp))
    total = sum(importances) or 1
    pcts = [i / total * 100 for i in importances]

    fig, ax = plt.subplots(figsize=(12, 8))
    bars = ax.barh(FEATURE_NAMES, pcts, color='#3498DB', edgecolor='#333', linewidth=0.5)
    ax.set_xlabel('Relative Importance (%)', fontsize=12)
    ax.set_ylabel('Feature', fontsize=12)
    ax.set_title('Feature Importance by Class Separability', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.2, axis='x')
    for bar, v in zip(bars, pcts):
        ax.text(bar.get_width() + 0.1, bar.get_y() + bar.get_height() / 2,
                f'{v:.1f}%', va='center', fontsize=10)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"[OK] Feature importance chart saved")


def plot_class_counts(samples, output_path):
    """Plot sample counts per gesture class."""
    features, labels = extract_features(samples)
    counts = {g: labels.count(g) for g in GESTURES}
    fig, ax = plt.subplots(figsize=(14, 7))
    x = range(len(GESTURES))
    active = [g for g in GESTURES if counts[g] > 0]
    bars = ax.bar(x, [counts[g] for g in GESTURES],
                  color=[GESTURE_COLORS[g] for g in GESTURES],
                  edgecolor='#333', linewidth=0.5, alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels([g.replace('_', '\n').title() for g in GESTURES], rotation=30, ha='right')
    ax.set_ylabel('Sample Count', fontsize=12)
    ax.set_title('Gesture Dataset Distribution', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.2, axis='y')
    for bar, cnt in zip(bars, [counts[g] for g in GESTURES]):
        if cnt:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                    str(cnt), ha='center', va='bottom', fontsize=11, fontweight='bold')
    total = sum(counts.values())
    ax.text(0.5, 1.02, f'Total: {total} samples', transform=ax.transAxes,
            ha='center', fontsize=12, fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"[OK] Class count chart saved")


def plot_confusion_matrix(true_labels, pred_labels, class_indices, output_path):
    """Plot confusion matrix from true vs predicted labels."""
    cm = np.zeros((len(GESTURES), len(GESTURES)), dtype=int)
    for t, p in zip(true_labels, pred_labels):
        cm[class_indices[t]][class_indices[p]] += 1
    fig, ax = plt.subplots(figsize=(12, 10))
    im = ax.imshow(cm, interpolation='nearest', cmap='Blues')
    tick_marks = np.arange(len(GESTURES))
    ax.set_xticks(tick_marks)
    ax.set_yticks(tick_marks)
    ax.set_xticklabels([g.title().replace('_', ' ') for g in GESTURES], rotation=45, ha='right')
    ax.set_yticklabels([g.title().replace('_', ' ') for g in GESTURES])
    thresh = cm.max() / 2.0
    for i in range(len(GESTURES)):
        for j in range(len(GESTURES)):
            ax.text(j, i, format(cm[i, j], 'd'), ha='center', va='center',
                    color='white' if cm[i, j] > thresh else 'black')
    ax.set_xlabel('Predicted', fontsize=12); ax.set_ylabel('True', fontsize=12)
    ax.set_title('Confusion Matrix', fontsize=14, fontweight='bold')
    plt.colorbar(im, ax=ax, shrink=0.8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"[OK] Confusion matrix saved")


def plot_per_sample_timeline(samples, output_path):
    """Plot temporal trend of features for a sample."""
    if not samples:
        return
    frames = samples[0].get('frames', [])
    if not frames:
        return
    n = len(frames)
    step = max(1, n // 200)
    tr = [f.get('thumb_extend_ratio', 0) for f in frames[::step]]
    or_ = [f.get('palm_orientation', 0) for f in frames[::step]]
    cx = [f.get('palm_center_x', 0) for f in frames[::step]]
    cy = [f.get('palm_center_y', 0) for f in frames[::step]]
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.plot(tr, 'o-', color='#7BC67E', label='Thumb Extend', linewidth=1.5, alpha=0.7)
    ax.plot(or_, 's-', color='#06D6A0', label='Palm Orientation', linewidth=1.5, alpha=0.7)
    ax.plot(cx, '^-', color='#118AB2', label='Palm Center X', linewidth=1.5, alpha=0.7)
    ax.plot(cy, 'd-', color='#EF476F', label='Palm Center Y', linewidth=1.5, alpha=0.7)
    ax.set_xlabel('Frame'); ax.set_ylabel('Value')
    ax.set_title(f"Temporal Evolution — {samples[0].get('filename', '?')}", fontweight='bold')
    ax.legend(fontsize=8, ncol=2)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"[OK] Timeline saved")


def main():
    """Generate all visualization charts."""
    print("=" * 60)
    print("  GENERATING Gesture Dataset Visualization Charts")
    print("=" * 60)

    samples = load_all_samples()
    if not samples:
        print("[ERROR] No samples found. Exiting."); sys.exit(1)

    out_dir = os.path.join(SCRIPT_DIR, "output")
    os.makedirs(out_dir, exist_ok=True)

    print("\n[1/5] Correlation heatmap...")
    plot_correlation_heatmap(compute_correlation_matrix(samples),
                             os.path.join(out_dir, "feature_correlation_heatmap.png"))

    print("[2/5] Distribution boxplots...")
    plot_distribution_boxplots(samples,
                               os.path.join(out_dir, "feature_distribution_boxplots.png"))

    print("[3/5] Decision boundaries...")
    plot_decision_boundaries(samples, os.path.join(out_dir, "decision_boundaries.png"))

    print("[4/5] Feature importance...")
    plot_feature_importance(samples, os.path.join(out_dir, "feature_importance.png"))

    print("[5/5] Class distribution...")
    plot_class_counts(samples, os.path.join(out_dir, "class_distribution.png"))

    print("[BONUS] Timeline...")
    plot_per_sample_timeline(samples, os.path.join(out_dir, "per_sample_timeline.png"))

    res_path = os.path.join(SCRIPT_DIR, "classification_results.json")
    if os.path.exists(res_path):
        with open(res_path) as fp:
            res = json.load(fp)
        if 'classifications' in res:
            plot_confusion_matrix(
                [c.get('true_label', 'unknown') for c in res['classifications']],
                [c.get('predicted_label', 'unknown') for c in res['classifications']],
                {g: i for i, g in enumerate(GESTURES)},
                os.path.join(out_dir, "confusion_matrix.png"),
            )

    print("\n" + "=" * 60)
    print(f"  All charts saved to: {out_dir}/")
    print("=" * 60)


if __name__ == '__main__':
    main()

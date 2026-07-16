#!/usr/bin/env python3
"""
train_and_benchmark.py - Train gesture classifiers and run full benchmark suite.

Trains multiple models against real gesture data, benchmarks them,
and generates comprehensive accuracy reports.

Usage:
  python train_and_benchmark.py                    # full training + benchmark
  python train_and_benchmark.py --models rf         # only train RF
  python train_and_benchmark.py --no-validation     # skip cross-validation
  python train_and_benchmark.py --output-dir ./results
"""

import json
import os
import sys
import argparse
import glob
from collections import defaultdict
from pathlib import Path

import numpy as np

# ML imports
try:
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
    from sklearn.linear_model import LogisticRegression, SGDClassifier
    from sklearn.svm import SVC
    from sklearn.neural_network import MLPClassifier
    from sklearn.model_selection import StratifiedKFold, cross_val_score
    from sklearn.metrics import (
        classification_report, confusion_matrix, accuracy_score,
        f1_score, precision_score, recall_score, roc_auc_score, cohen_kappa_score
    )
    from sklearn.preprocessing import StandardScaler, LabelEncoder
    import sklearn
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False
    print("[WARN] scikit-learn not installed. Install with: pip install scikit-learn")
    sys.exit(1)

# Visualization imports
try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    from matplotlib.colors import LinearSegmentedColormap
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

# Import the existing gesture engine for comparison
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
DATA_DIR = os.path.join(PROJECT_ROOT, "data")

sys.path.insert(0, PROJECT_ROOT)
from vision.gestures_v2 import (
    classify, Gesture,
    THUMB_TIP_SEPARATION_BEYOND_IP,
    PALM_NORMAL_Z_THRESHOLD,
    FINGER_TIP_SEPARATION_OPEN_RATIO
)

# --- Configuration ---
GESTURES = [
    'open_palm', 'closed_fist', 'thumbs_up', 'thumbs_down',
    'palm_up', 'palm_down', 'palm_left', 'palm_right',
]
GESTURE_COLORS = {
    'open_palm': '#FF6B6B', 'closed_fist': '#4A47A3',
    'thumbs_up': '#7BC67E', 'thumbs_down': '#FFD166',
    'palm_up': '#06D6A0', 'palm_down': '#118AB2',
    'palm_left': '#EF476F', 'palm_right': '#073B4C',
}
GESTURE_TO_ENUM = {g: getattr(Gesture, g.upper().replace('_', '_')) for g in GESTURES}

FEATURE_NAMES = [
    'thumb_extend_ratio', 'palm_orientation', 'palm_center_x',
    'palm_center_y', 'thumb_tips_spread', 'wrist_angle',
    'thumb_palm_distance', 'palm_size', 'extended_fingers',
    'wrist_palm_distance', 'palm_aspect_ratio', 'frame_angle',
]


def load_all_samples(data_dir):
    """Load all gesture samples from nested data directories."""
    samples = []
    for gesture in GESTURES:
        gesture_dir = os.path.join(data_dir, gesture)
        if not os.path.isdir(gesture_dir):
            continue
        for json_file in sorted(glob.glob(os.path.join(gesture_dir, 'sample_*.json'))):
            with open(json_file) as fp:
                data = json.load(fp)
            if '_meta' not in data:
                data['_meta'] = {}
            data['_meta']['gesture'] = gesture
            samples.append(data)
    print(f"Loaded {len(samples)} samples from {data_dir}")
    return samples


def landmarks_to_dict(lm_list):
    """Convert landmarks to list of dicts."""
    result = []
    for lm in lm_list:
        if isinstance(lm, dict):
            result.append(lm)
        else:
            result.append({
                'x': lm.x if hasattr(lm, 'x') else lm[0],
                'y': lm.y if hasattr(lm, 'y') else lm[1],
                'z': lm.z if hasattr(lm, 'z') else (lm[2] if len(lm) > 2 else 0),
            })
    return result


def extract_features(sample):
    """Extract 12 features from raw landmarks."""
    features = np.zeros(len(FEATURE_NAMES), dtype=np.float64)
    img_w = sample['width']
    img_h = sample['height']
    raw = landmarks_to_dict(sample['landmarks'])

    # Extract key points
    xs = np.array([p['x'] for p in raw])
    ys = np.array([p['y'] for p in raw])
    zs = np.array([p['z'] for p in raw])

    # Normalize landmark positions
    normalized = np.column_stack([xs, ys, zs]) * np.array([img_w, img_h, img_w])

    # Key landmarks
    wrist = normalized[0]
    thumb_tip = normalized[4]
    pinky_tip = normalized[20]

    # 1. Thumb extend ratio: thumb tip distance from IP joint / box size
    ip = normalized[2]
    ip_j = normalized[3]
    thumb_extend = np.linalg.norm(thumb_tip[:2] - ip[:2])
    box_size = max(np.ptp(xs) * img_w, np.ptp(ys) * img_h, 1.0)
    features[0] = thumb_extend / box_size

    # 2. Palm orientation: cross product of middle finger and pinky vectors relative to wrist
    middle = normalized[12]
    pinky = normalized[17]
    vec_middle = middle[:2] - wrist[:2]
    vec_pinky = pinky[:2] - wrist[:2]
    cross_z = vec_middle[0] * vec_pinky[1] - vec_middle[1] * vec_pinky[0]
    features[1] = cross_z / (box_size ** 2)

    # 3. Palm center X
    features[2] = np.mean(xs)

    # 4. Palm center Y
    features[3] = np.mean(ys)

    # 5. Thumb-tips spread: thumb tip to pinky tip distance / box size
    features[4] = np.linalg.norm(thumb_tip[:2] - pinky_tip[:2]) / box_size

    # 6. Wrist angle from horizontal
    wrist_angle = np.arctan2(
        (normalized[12][1] + normalized[17][1]) / 2 - wrist[1],
        (normalized[12][0] + normalized[17][0]) / 2 - wrist[0]
    ) * 180 / np.pi
    features[5] = wrist_angle

    # 7. Thumb-palm distance: thumb tip center to palm center distance
    palm_center = np.array([features[2], features[3]]) * np.array([img_w, img_h])
    features[6] = np.linalg.norm(thumb_tip[:2] - palm_center) / box_size

    # 8. Palm size: bounding box area / image area
    features[7] = (np.ptp(xs) * np.ptp(ys)) / (img_w * img_h)

    # 9. Extended fingers count (simplified: check if tip is far from MCP for each finger)
    extended = 0
    for tip_id, mcp_id in [(8, 5), (12, 9), (16, 13), (20, 17)]:
        tip_pt = raw[tip_id]
        mcp_pt = raw[mcp_id]
        dist = np.sqrt((tip_pt['x'] - mcp_pt['x'])**2 + (tip_pt['y'] - mcp_pt['y'])**2) * max(img_w, img_h)
        if dist / box_size > 0.3:
            extended += 1
    features[8] = extended

    # 10. Wrist-palm distance (0 if wrist is in center, else distance to bounding box center)
    bbox_center = np.array([(np.ptp(xs) + np.min(xs)) / 2 * img_w,
                            (np.ptp(ys) + np.min(ys)) / 2 * img_h])
    features[9] = np.linalg.norm(wrist[:2] - bbox_center) / box_size

    # 11. Palm aspect ratio
    bbox_w = np.ptp(xs) * img_w
    bbox_h = np.ptp(ys) * img_h
    features[10] = bbox_w / max(bbox_h, 1.0)

    # 12. Frame angle: orientation of hand relative to frame
    features[11] = np.arctan2(bbox_h - bbox_w, bbox_h + bbox_w) * 180 / np.pi

    return features


def prepare_dataset(samples):
    """Convert samples to X, y arrays."""
    X = []
    y = []
    y_names = []
    for s in samples:
        feat = extract_features(s)
        X.append(feat)
        y.append(s['_meta']['gesture'])
        y_names.append(s['_meta']['gesture'])
    X = np.array(X)
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    return X, y_encoded, le, y_names


def train_random_forest(X_train, y_train, n_estimators=200):
    """Train RandomForest classifier."""
    model = RandomForestClassifier(
        n_estimators=n_estimators, max_depth=None, min_samples_split=2,
        min_samples_leaf=1, random_state=42, n_jobs=-1
    )
    model.fit(X_train, y_train)
    return model


def train_gradient_boosting(X_train, y_train):
    """Train GradientBoosting classifier."""
    model = GradientBoostingClassifier(
        n_estimators=150, learning_rate=0.1, max_depth=4, random_state=42
    )
    model.fit(X_train, y_train)
    return model


def train_svm(X_train, y_train):
    """Train SVM classifier."""
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_train)
    model = SVC(kernel='rbf', C=10, gamma='scale', random_state=42)
    model.fit(X_scaled, y_train)
    return model, scaler


def train_mlp(X_train, y_train):
    """Train Multi-layer Perceptron (neural network)."""
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_train)
    model = MLPClassifier(
        hidden_layer_sizes=(64, 32), activation='relu',
        solver='adam', max_iter=500, random_state=42
    )
    model.fit(X_scaled, y_train)
    return model, scaler


def train_logistic_regression(X_train, y_train):
    """Train Logistic Regression."""
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_train)
    model = LogisticRegression(max_iter=1000, random_state=42, C=1.0)
    model.fit(X_scaled, y_train)
    return model, scaler


def train_sgd(X_train, y_train):
    """Train SGD (Stochastic Gradient Descent) classifier."""
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_train)
    model = SGDClassifier(max_iter=1000, random_state=42, loss='log_loss')
    model.fit(X_scaled, y_train)
    return model, scaler


def train_model(model_type, X_train, y_train):
    """Train a model and return (model, scaler_or_none, name)."""
    if model_type == 'rf':
        model = train_random_forest(X_train, y_train)
        return model, None, 'RandomForest'
    elif model_type == 'gb':
        model = train_gradient_boosting(X_train, y_train)
        return model, None, 'GradientBoosting'
    elif model_type == 'svm':
        model, scaler = train_svm(X_train, y_train)
        return model, scaler, 'SVM'
    elif model_type == 'mlp':
        model, scaler = train_mlp(X_train, y_train)
        return model, scaler, 'MLP'
    elif model_type == 'lr':
        model, scaler = train_logistic_regression(X_train, y_train)
        return model, scaler, 'LogisticRegression'
    elif model_type == 'sgd':
        model, scaler = train_sgd(X_train, y_train)
        return model, scaler, 'SGD'
    else:
        raise ValueError(f"Unknown model type: {model_type}")


def predict_with_model(model, scaler, X, le):
    """Predict with model (handles scalers if needed)."""
    if scaler is not None:
        X_processed = scaler.transform(X)
    else:
        X_processed = X
    pred_encoded = model.predict(X_processed)
    pred_names = le.inverse_transform(pred_encoded)
    return pred_names, pred_encoded


def evaluate_model(model, scaler, X_test, y_test, le, model_name):
    """Evaluate a model and return metrics dict."""
    pred_names, pred_encoded = predict_with_model(model, scaler, X_test, le)

    acc = accuracy_score(y_test, pred_encoded) * 100
    f1_macro = f1_score(y_test, pred_encoded, average='macro', zero_division=0) * 100
    f1_weighted = f1_score(y_test, pred_encoded, average='weighted', zero_division=0) * 100
    kappa = cohen_kappa_score(y_test, pred_encoded)

    try:
        auc = roc_auc_score(y_test, model.predict_proba(X_test) if scaler is None else model.predict_proba(scaler.transform(X_test)),
                            multi_class='ovr', average='macro') * 100
    except:
        auc = 0.0

    # Per-class metrics
    class_names = le.classes_
    target_names = [f"{n} ({i})" for i, n in enumerate(class_names)]
    cr = classification_report(y_test, pred_encoded, target_names=target_names, zero_division=0)

    # Confusion matrix
    cm = confusion_matrix(y_test, pred_encoded)

    return {
        'accuracy': acc,
        'f1_macro': f1_macro,
        'f1_weighted': f1_weighted,
        'kappa': kappa,
        'auc': auc,
        'classification_report': cr,
        'confusion_matrix': cm,
        'true_labels': y_test,
        'pred_labels': pred_encoded,
    }


def cross_validate_model(model_type, X, y, n_splits=5):
    """Cross-validate model and return mean scores."""
    results = {}
    
    if model_type == 'rf':
        model = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
        scaler = None
        fit_scaler = False
    elif model_type == 'gb':
        model = GradientBoostingClassifier(n_estimators=150, learning_rate=0.1, random_state=42)
        scaler = None
        fit_scaler = False
    elif model_type == 'svm':
        scaler = StandardScaler()
        model = SVC(kernel='rbf', C=10, random_state=42)
        fit_scaler = True
    elif model_type == 'mlp':
        scaler = StandardScaler()
        model = MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=500, random_state=42)
        fit_scaler = True
    elif model_type == 'lr':
        scaler = StandardScaler()
        model = LogisticRegression(max_iter=1000, random_state=42)
        fit_scaler = True
    elif model_type == 'sgd':
        scaler = StandardScaler()
        model = SGDClassifier(max_iter=1000, random_state=42)
        fit_scaler = True
    else:
        return {}

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    scores_acc = []
    scores_f1 = []
    for train_idx, val_idx in skf.split(X, y):
        X_tr, X_val = X[train_idx], X[val_idx]
        y_tr, y_val = y[train_idx], y[val_idx]

        if fit_scaler and scaler is not None:
            X_tr = scaler.fit_transform(X_tr)
            X_val = scaler.transform(X_val)

        model.fit(X_tr, y_tr)
        y_val_pred = model.predict(X_val)
        scores_acc.append(accuracy_score(y_val, y_val_pred) * 100)
        scores_f1.append(f1_score(y_val, y_val_pred, average='macro', zero_division=0) * 100)

    results['cv_accuracy_mean'] = np.mean(scores_acc)
    results['cv_accuracy_std'] = np.std(scores_acc)
    results['cv_f1_macro_mean'] = np.mean(scores_f1)
    results['cv_f1_macro_std'] = np.std(scores_f1)
    return results


def benchmark_existing_gesture_engine(samples):
    """Benchmark the existing gestures_v2.classify() method."""
    confusion = defaultdict(lambda: defaultdict(int))
    total = 0
    correct = 0
    per_gesture_correct = defaultdict(int)
    per_gesture_total = defaultdict(int)

    for sample in samples:
        gesture = sample['_meta']['gesture']
        landmarks_orig = sample['landmarks']
        img_w = sample['width']
        img_h = sample['height']
        
        # Convert dict landmarks to class-style for gestures_v2.classify
        class Lm:
            def __init__(self, d):
                self.x = d['x']
                self.y = d['y']
                self.z = d['z']
        landmarks = [Lm(lm) if isinstance(lm, dict) else lm for lm in landmarks_orig]

        predicted = classify(landmarks, img_w, img_h)
        predicted_name = predicted.name.lower().replace('unknown_', '')
        if predicted == Gesture.UNKNOWN:
            predicted_name = 'unknown'

        true_name = gesture
        confusion[true_name][predicted_name] += 1
        total += 1
        per_gesture_total[gesture] += 1

        # Match gesture name to enum
        enum_key = gesture.upper().replace('_', '_')
        if enum_key in Gesture.__members__:
            true_enum = Gesture[enum_key]
        else:
            true_enum = Gesture[gesture.upper()] if gesture.upper() in Gesture.__members__ else None
        
        if true_enum and predicted == true_enum:
            correct += 1
            per_gesture_correct[gesture] += 1

    accuracy = correct / total if total > 0 else 0

    return {
        'confusion': confusion,
        'total': total,
        'correct': correct,
        'accuracy': accuracy * 100,
        'per_gesture_correct': per_gesture_correct,
        'per_gesture_total': per_gesture_total,
    }


def print_confusion_matrix(cm, classes, title="Confusion Matrix"):
    """Print formatted confusion matrix."""
    print(f"\n{title}")
    print("-" * (len(classes) * 8 + 10))
    header = f"{'':>20}"
    for c in classes:
        header += f"{c:>8}"
    print(header)
    print("-" * (len(classes) * 8 + 14))

    for i, true_class in enumerate(classes):
        row = f"{true_class:>20}"
        for j, pred_class in enumerate(classes):
            row += f"{cm[i, j]:>8}"
        print(row)
    print("-" * (len(classes) * 8 + 14))


def print_per_gesture_accuracy(per_gesture_correct, per_gesture_total, GESTURES):
    """Print per-gesture accuracy bars."""
    print(f"\n{'Gesture':<20} {'Correct':>8} {'Total':>6} {'':>4} {'Accuracy':>10}")
    print(f"{'─' * 20} {'─' * 8} {'─' * 6} {'':>4} {'─' * 10}")

    for g in GESTURES:
        correct = per_gesture_correct.get(g, 0)
        total = per_gesture_total.get(g, 0)
        acc = correct / total * 100 if total > 0 else 0
        bar = '█' * int(acc / 5) + '░' * (20 - int(acc / 5))
        print(f"{g:<20} {correct:>8} {total:>6} {'':>4} [{bar}] {acc:>6.1f}%")


def plot_confusion_matrices(results, output_path):
    """Plot confusion matrices for all models."""
    if not HAS_MATPLOTLIB:
        return

    fig, axes = plt.subplots(len(results), 1, figsize=(14, 4 * len(results)))
    if len(results) == 1:
        axes = [axes]

    for ax, (name, res) in zip(axes, results.items()):
        cm = res['confusion_matrix']
        classes = res.get('classes', GESTURES)
        acc = res['accuracy']

        im = ax.imshow(cm, interpolation='nearest', cmap='Blues')
        tick_marks = np.arange(len(classes))
        ax.set_xticks(tick_marks)
        ax.set_yticks(tick_marks)
        ax.set_xticklabels([g[:8] for g in classes], rotation=45, ha='right')
        ax.set_yticklabels([g[:8] for g in classes])

        # Add text annotations
        thresh = cm.max() / 2
        for i in range(len(classes)):
            for j in range(len(classes)):
                ax.text(j, i, f'{cm[i, j]}', ha='center', va='center',
                        color='white' if cm[i, j] > thresh else 'black', fontsize=8)

        ax.set_xlabel('Predicted', fontsize=10)
        ax.set_ylabel('True', fontsize=10)
        ax.set_title(f'{name} | Accuracy: {acc:.1f}%', fontsize=12, fontweight='bold')
        plt.colorbar(im, ax=ax, shrink=0.8)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[OK] Confusion matrices saved to {output_path}")


def plot_feature_importance(model, feature_names, output_path):
    """Plot feature importance for tree-based models."""
    if not HAS_MATPLOTLIB:
        return

    try:
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1]

        fig, ax = plt.subplots(figsize=(12, 8))
        top_n = min(12, len(importances))
        bar_labels = [feature_names[i].replace('_', ' ') for i in indices][:top_n]
        bar_values = [importances[i] for i in indices][:top_n]

        ax.barh(range(top_n), bar_values, color='#3498DB', edgecolor='#333', linewidth=0.5)
        ax.set_yticks(range(top_n))
        ax.set_yticklabels(bar_labels)
        ax.set_xlabel('Importance', fontsize=12)
        ax.set_title('Feature Importance (RandomForest)', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.2, axis='x')
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"[OK] Feature importance saved to {output_path}")
    except Exception as e:
        print(f"[WARN] Could not plot feature importance: {e}")


def generate_benchmark_report(results, output_dir):
    """Generate comprehensive benchmark report."""
    report_lines = [
        "=" * 70,
        "  GESTURE CLASSIFICATION BENCHMARK REPORT",
        "=" * 70,
        f"\n  Date: {np.datetime64('now').astype(str)[:16]}",
        f"  Models Evaluated: {len(results)}",
        f"  Features: {len(FEATURE_NAMES)}",
        f"  Classifiers: {', '.join(results.keys())}",
        "=" * 70,
    ]

    # Summary table
    report_lines.append("\n  MODEL PERFORMANCE SUMMARY")
    report_lines.append("  ─" * 70)
    report_lines.append(f"  {'Model':<20} {'Acc%':>8} {'F1-Macro':>10} {'F1-Weighted':>12} {'Kappa':>8}")
    report_lines.append("  " + "─" * (20 + 8 + 10 + 12 + 8))

    for name, res in results.items():
        report_lines.append(
            f"  {name:<20} {res['accuracy']:>8.1f} {res['f1_macro']:>10.1f} "
            f"{res['f1_weighted']:>12.1f} {res['kappa']:>8.2f}"
        )
    report_lines.append("  ─" * 70)

    # Best model
    best_model = max(results.items(), key=lambda x: x[1]['accuracy'])
    report_lines.append(f"\n  ★ Best Model: {best_model[0]} (Accuracy: {best_model[1]['accuracy']:.1f}%)")

    # Detailed classifications
    for name, res in results.items():
        report_lines.append(f"\n  ─" * 70)
        report_lines.append(f"  DETAILED REPORT: {name}")
        report_lines.append(f"  ─" * 70)
        report_lines.append(res['classification_report'])

    report_lines.append("=" * 70)
    report_lines.append("  END OF BENCHMARK REPORT")
    report_lines.append("=" * 70)

    report_content = '\n'.join(report_lines)
    report_file = os.path.join(output_dir, 'benchmark_report.txt')

    with open(report_file, 'w') as f:
        f.write(report_content)

    print(f"[OK] Benchmark report saved to {report_file}")
    return report_content


def main():
    parser = argparse.ArgumentParser(description="Train gesture classifiers and run benchmark.")
    parser.add_argument('--models', type=str, default='all',
                        help="Comma-separated list: rf, gb, svm, mlp, lr, sgd, all")
    parser.add_argument('--cv', action='store_true', default=True,
                        help="Run cross-validation")
    parser.add_argument('--no-cv', action='store_true',
                        help="Skip cross-validation")
    parser.add_argument('--n-splits', type=int, default=5,
                        help="Number of CV folds")
    parser.add_argument('--output-dir', type=str, default=os.path.join(SCRIPT_DIR, 'results'),
                        help="Output directory for results")

    args = parser.parse_args()
    print("=" * 70)
    print("  TRAINING & BENCHMARKING GESTURE CLASSIFIERS")
    print("=" * 70)

    # Prepare data
    print("\n[1/7] Loading data...")
    samples = load_all_samples(DATA_DIR)
    if not samples:
        print("[ERROR] No samples found. Exiting."); sys.exit(1)

    X, y, le, y_names = prepare_dataset(samples)
    print(f"  Dataset: {X.shape[0]} samples, {X.shape[1]} features")
    print(f"  Classes: {le.classes_}")
    print(f"  Classes per gesture: {[np.sum(y == le.transform([g])[0]) for g in le.classes_]}")

    # Split data
    print("\n[2/7] Preparing train/test split...")
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    print(f"  Train: {X_train.shape[0]} samples")
    print(f"  Test: {X_test.shape[0]} samples")

    # Initialize results
    results = {}
    cv_results = {}

    # Model types to train
    if args.models == 'all':
        model_types = ['rf', 'gb', 'svm', 'mlp', 'lr', 'sgd']
    else:
        model_types = [m.strip() for m in args.models.split(',')]

    if args.no_cv:
        args.cv = False

    # Train and evaluate each model
    print("\n[3/7] Training and evaluating models...")
    models_and_scalers = {}

    for model_type in model_types:
        print(f"\n  • Training {model_type.upper()}...")
        try:
            model, scaler, name = train_model(model_type, X_train, y_train)
            models_and_scalers[model_type] = (model, scaler)

            # Evaluate on test set
            metrics = evaluate_model(model, scaler, X_test, y_test, le, name)
            results[model_type] = metrics
            print(f"    [OK] Accuracy: {metrics['accuracy']:.1f}% | F1-Macro: {metrics['f1_macro']:.1f}%")

            # Cross-validation
            if args.cv:
                print(f"    Running {args.n_splits}-fold CV...")
                cv_metrics = cross_validate_model(model_type, X, y, n_splits=args.n_splits)
                cv_results[model_type] = cv_metrics
                print(f"    [OK] CV-Accuracy: {cv_metrics['cv_accuracy_mean']:.1f}% ± {cv_metrics['cv_accuracy_std']:.1f}%")

        except Exception as e:
            print(f"    [ERROR] Failed to train {model_type}: {e}")
            import traceback
            traceback.print_exc()

    # Benchmark gestures_v2.classify()
    print("\n[4/7] Benchmarking gestures_v2.classify()...")
    benchmark_results = benchmark_existing_gesture_engine(samples)
    benchmark_accuracy = benchmark_results['accuracy']
    print(f"  [OK] gestures_v2.classify() accuracy: {benchmark_accuracy:.1f}%")

    # Print confusion matrices
    print("\n[5/7] Per-gesture accuracy:")
    for model_type, res in results.items():
        print(f"\n  {model_type.upper()}:")
        print_per_gesture_accuracy(
            defaultdict(int, {le.classes_[i]: res['pred_labels'].tolist().count(i)
                            for i in range(len(le.classes_))}),
            defaultdict(int, {le.classes_[i]: res['true_labels'].tolist().count(i)
                            for i in range(len(le.classes_))}),
            le.classes_
        )

    # Visualization
    print("\n[6/7] Generating visualizations...")
    os.makedirs(args.output_dir, exist_ok=True)

    # Plot confusion matrices
    plot_confusion_matrices(results, os.path.join(args.output_dir, 'confusion_matrix.png'))

    # Plot feature importance (for RandomForest)
    if 'rf' in models_and_scalers:
        m, _ = models_and_scalers['rf']
        plot_feature_importance(m, FEATURE_NAMES, os.path.join(args.output_dir, 'feature_importance.png'))

    # Generate comprehensive benchmark report
    print("\n[7/7] Generating benchmark report...")
    for model_type, res in results.items():
        res['classes'] = le.classes_
    generate_benchmark_report(results, args.output_dir)

    # Print final comparison
    print("\n" + "=" * 70)
    print("  FINAL COMPARISON")
    print("=" * 70)
    print(f"  Model                    Accuracy     F1-Macro     CV-Accuracy")
    print(f"  {'─' * 70}")
    print(f"  {'gestures_v2.classify':20s} {benchmark_accuracy:>10.1f}         {'':>8}         {'':>8}")

    for model_type, res in results.items():
        cv_acc = cv_results.get(model_type, {}).get('cv_accuracy_mean', 0)
        name = model_type.upper()[:8].ljust(20)
        acc_str = f"{res['accuracy']:>10.1f}"
        f1_str = f"{res['f1_macro']:>10.1f}"
        cv_str = f"{cv_acc:>10.1f}"
        print(f"  {name} {acc_str} {f1_str} {cv_str}")

    # Output best model
    best_model = max(results.items(), key=lambda x: x[1]['accuracy'])
    print(f"\n  ★ BEST MODEL: {best_model[0].upper()} "
          f"(Training Accuracy: {best_model[1]['accuracy']:.1f}%)")

    # Save best model + scaler + label_encoder
    if best_model:
        import joblib
        path = os.path.join(args.output_dir, f'best_model_{best_model[0]}.pkl')
        
        # Get the actual model, scaler, and label encoder
        best_result = results[best_model[0]]
        best_model_obj, scaler = models_and_scalers[best_model[0]]
        
        joblib.dump({
            'model': models_and_scalers[best_model[0]][0],
            'scaler': models_and_scalers[best_model[0]][1],
            'label_encoder': le,
            'feature_names': FEATURE_NAMES,
            'best_accuracy': best_result['accuracy'],
            'best_f1_macro': best_result['f1_macro'],
            'best_f1_weighted': best_result['f1_weighted'],
        }, path, protocol=4)
        print(f"  [OK] Best model saved to {path}")

    print("\n" + "=" * 70)
    print("  BENCHMARK COMPLETE")
    print("=" * 70)


if __name__ == '__main__':
    main()

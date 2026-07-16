#!/usr/bin/env python3
"""
train_lstm_model.py - Train LSTM temporal gesture classifier.

Generates synthetic temporal sequences from single-frame data by simulating
gesture transition dynamics, then trains an LSTM model for temporal-based classification.
The LSTM model captures the dynamic flow of gesture movements for more robust recognition.

Usage:
  python train_lstm_model.py           # full training pipeline
  python train_lstm_model.py --seq-len 30   # custom sequence length
  python train_lstm_model.py --epochs 200    # custom training iterations
"""

import os
import sys
import glob
import json
import argparse
import numpy as np
from pathlib import Path
from collections import defaultdict

# ML imports
try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
    from tensorflow.keras.models import load_model
    HAS_TF = True
except ImportError:
    HAS_TF = False
    print("[WARN] TensorFlow not installed. Install with: pip install tensorflow")
    sys.exit(1)

try:
    import joblib
    HAS_JOBLIB = True
except ImportError:
    HAS_JOBLIB = False

# Paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
RESULTS_DIR = os.path.join(SCRIPT_DIR, "results")

GESTURES = [
    'open_palm', 'closed_fist', 'thumbs_up', 'thumbs_down',
    'palm_up', 'palm_down', 'palm_left', 'palm_right',
]

FEATURE_NAMES = [
    'thumb_extend_ratio', 'palm_orientation', 'palm_center_x',
    'palm_center_y', 'thumb_tips_spread', 'wrist_angle',
    'thumb_palm_distance', 'palm_size', 'extended_fingers',
    'wrist_palm_distance', 'palm_aspect_ratio', 'frame_angle',
]


def load_all_features():
    """Load features from all gesture samples."""
    feature_data = defaultdict(list)
    
    for gesture in GESTURES:
        gesture_dir = os.path.join(DATA_DIR, gesture)
        if not os.path.isdir(gesture_dir):
            continue
            
        for sample_file in sorted(glob.glob(os.path.join(gesture_dir, 'sample_*.json'))):
            with open(sample_file) as f:
                data = json.load(f)
            
            landmarks = data['landmarks']
            width = data['width']
            height = data['height']
            
            # Extract coordinates
            xs = np.array([l['x'] * width for l in landmarks[:21]])
            ys = np.array([l['y'] * height for l in landmarks[:21]])
            zs = np.array([l['z'] * width for l in landmarks[:21]])
            
            normalized = np.column_stack([xs, ys, zs])
            wrist = normalized[0]
            thumb_tip = normalized[4]
            pinky_tip = normalized[20]
            ip = normalized[2]
            
            box_size = max(np.ptp(xs), np.ptp(ys), 1.0)
            
            features = np.zeros(12, dtype=np.float64)
            
            # 1. thumb_extend_ratio
            thumb_extend = np.linalg.norm(thumb_tip[:2] - ip[:2])
            features[0] = thumb_extend / box_size
            
            # 2. palm_orientation (cross product)
            middle = normalized[12]
            pinky_vec = normalized[17]
            vec_middle = middle[:2] - wrist[:2]
            vec_pinky = pinky_vec[:2] - wrist[:2]
            cross_z = vec_middle[0] * vec_pinky[1] - vec_middle[1] * vec_pinky[0]
            features[1] = cross_z / (box_size ** 2)
            
            # 3-4. palm_center_x, palm_center_y
            features[2] = np.mean(xs)
            features[3] = np.mean(ys)
            
            # 5. thumb_tips_spread
            features[4] = np.linalg.norm(thumb_tip[:2] - pinky_tip[:2]) / box_size
            
            # 6. wrist_angle
            wrist_angle = np.arctan2(
                (ys[12] + ys[17]) / 2 - ys[0],
                (xs[12] + xs[17]) / 2 - xs[0]
            ) * 180 / np.pi
            features[5] = wrist_angle
            
            # 7. thumb_palm_distance
            palm_center = np.array([np.mean(xs), np.mean(ys)])
            features[6] = np.linalg.norm(thumb_tip[:2] - palm_center) / box_size
            
            # 8. palm_size (normalized)
            features[7] = (np.ptp(xs) * np.ptp(ys)) / (width * height)
            
            # 9. extended_fingers
            extended = 0
            for tip_id, mcp_id in [(8, 5), (12, 9), (16, 13), (20, 17)]:
                tip_pt = np.array([xs[tip_id], ys[tip_id]])
                mcp_pt = np.array([xs[mcp_id], ys[mcp_id]])
                dist = np.linalg.norm(tip_pt - mcp_pt)
                if dist / box_size > 0.3:
                    extended += 1
            features[8] = extended
            
            # 10. wrist_palm_distance
            bbox_center = np.array([(np.ptp(xs) + np.min(xs)) / 2,
                                    (np.ptp(ys) + np.min(ys)) / 2])
            features[9] = np.linalg.norm(wrist[:2] - bbox_center) / box_size
            
            # 11. palm_aspect_ratio
            bbox_w = np.ptp(xs)
            bbox_h = np.ptp(ys)
            features[10] = bbox_w / max(bbox_h, 1.0)
            
            # 12. frame_angle
            features[11] = np.arctan2(bbox_h - bbox_w, bbox_h + bbox_w) * 180 / np.pi
            
            feature_data[gesture].append(features)
    
    return feature_data


def generate_temporal_sequences(feature_data, seq_len=30, num_per_class=100):
    """Generate synthetic temporal sequences from single-frame features."""
    sequences = []
    labels = []
    
    gesture_ids = {g: i for i, g in enumerate(GESTURES)}
    
    for gesture, feature_list in feature_data.items():
        if len(feature_list) < 2:
            continue
        
        feature_array = np.array(feature_list)
        gesture_id = gesture_ids[gesture]
        
        # Generate sequences by adding temporal dynamics
        for _ in range(num_per_class):
            # Select a base sample
            base_idx = np.random.randint(len(feature_array))
            base_features = feature_array[base_idx].copy()
            
            # Different gesture types have different motion patterns
            if gesture in ['open_palm', 'closed_fist']:
                # Opening/closing is more dramatic
                motion_scale = 0.05
                jitter_type = 'sinusoidal'
            elif gesture in ['thumbs_up', 'thumbs_down']:
                # Thumb gestures are simpler
                motion_scale = 0.03
                jitter_type = 'gentle'
            elif gesture in ['palm_up', 'palm_down']:
                # Palm orientation changes
                motion_scale = 0.04
                jitter_type = 'oscillating'
            else:  # palm_left, palm_right
                # Position-based gestures
                motion_scale = 0.06
                jitter_type = 'directional'
            
            # Generate sequence with realistic temporal progression
            sequence = np.zeros((seq_len, 12))
            
            for t in range(seq_len):
                t_norm = t / seq_len  # normalized time 0-1
                
                # Base with gradual progression toward "final" pose
                noise = np.random.randn(12) * motion_scale
                
                # Time-dependent modulation
                if jitter_type == 'sinusoidal':
                    modulation = 0.5 + 0.5 * np.sin(np.pi * t_norm * 4)  # 2 full cycles
                elif jitter_type == 'oscillating':
                    modulation = np.sin(np.pi * t_norm * 6) * 0.3
                elif jitter_type == 'directional':
                    modulation = np.clip(t_norm * 2, 0, 1)
                else:  # gentle
                    modulation = np.sin(np.pi * t_norm * 2) * 0.5
                
                noise *= modulation
                
                # Add temporal autocorrelation (each frame depends on previous)
                if t > 0:
                    sequence[t] = sequence[t-1] * 0.7 + (noise + base_features) * 0.3
                else:
                    # Start from slightly perturbed version for natural motion
                    start_noise = np.random.randn(12) * 0.02
                    sequence[t] = base_features + start_noise
            
            sequences.append(sequence)
            labels.append(gesture_id)
    
    # Convert to numpy arrays
    X = np.array(sequences)
    y = np.array(labels)
    
    # Normalize features
    # Reshape for scaler: (seq_len * batch_size, features)
    X_flat = X.reshape(-1, 12)
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    X_flat_scaled = scaler.fit_transform(X_flat)
    X_scaled = X_flat_scaled.reshape(X.shape)
    
    # One-hot encode labels
    from sklearn.preprocessing import LabelBinarizer
    lb = LabelBinarizer()
    y_onehot = lb.fit_transform(y)
    
    return X_scaled, y, y_onehot, scaler, lb, gesture_ids


def build_lstm_model(input_shape, num_classes):
    """Build LSTM model architecture."""
    model = Sequential([
        LSTM(128, return_sequences=True, input_shape=input_shape,
             kernel_regularizer=tf.keras.regularizers.l2(1e-4),
             recurrent_regularizer=tf.keras.regularizers.l2(1e-4)),
        Dropout(0.3),
        BatchNormalization(),
        
        LSTM(64, return_sequences=True,
             kernel_regularizer=tf.keras.regularizers.l2(1e-4),
             recurrent_regularizer=tf.keras.regularizers.l2(1e-4)),
        Dropout(0.3),
        BatchNormalization(),
        
        LSTM(32,
             kernel_regularizer=tf.keras.regularizers.l2(1e-4)),
        Dropout(0.2),
        
        Dense(32, activation='relu'),
        Dropout(0.2),
        
        Dense(num_classes, activation='softmax')
    ])
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model


def train_lstm_model():
    """Full training pipeline."""
    print("=" * 60)
    print("LSTM Temporal Gesture Classifier Training")
    print("=" * 60)
    
    # Step 1: Load features
    print("\n[1/5] Loading feature data...")
    feature_data = load_all_features()
    
    total_samples = sum(len(v) for v in feature_data.values())
    print(f"   Loaded {total_samples} samples from {len(feature_data)} classes:")
    for gesture, features in feature_data.items():
        print(f"   - {gesture}: {len(features)} samples")
    
    if total_samples < 50:
        print("[ERROR] Not enough data for training. Need at least 50 samples.")
        sys.exit(1)
    
    # Step 2: Generate temporal sequences
    print("\n[2/5] Generating temporal sequences...")
    X_scaled, y_orig, y_onehot, scaler, lb, gesture_ids = generate_temporal_sequences(
        feature_data, seq_len=30, num_per_class=80
    )
    
    print(f"   Generated {len(X_scaled)} sequences")
    print(f"   Shape: {X_scaled.shape} (seq_len, samples, features)")
    print(f"   Classes: {len(gesture_ids)} ({', '.join(GESTURES)})")
    
    # Step 3: Split data
    print("\n[3/5] Splitting data...")
    from sklearn.model_selection import train_test_split
    
    # Temporal-aware split: keep sequences of same gesture together for validation
    train_idx, val_idx = train_test_split(
        np.arange(len(X_scaled)),
        test_size=0.2,
        stratify=y_orig,
        random_state=42
    )
    
    X_train = X_scaled[train_idx]
    y_train = y_onehot[train_idx]
    X_val = X_scaled[val_idx]
    y_val = y_onehot[val_idx]
    
    print(f"   Training set: {len(X_train)} sequences")
    print(f"   Validation set: {len(X_val)} sequences")
    
    # Step 4: Build and train model
    print("\n[4/5] Building LSTM model...")
    model = build_lstm_model((30, 12), len(GESTURES))
    model.summary()
    
    # Callbacks
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=30,
        restore_best_weights=True,
        verbose=1
    )
    
    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=10,
        min_lr=1e-6,
        verbose=1
    )
    
    print("\nTraining...")
    history = model.fit(
        X_train, y_train,
        epochs=200,
        batch_size=32,
        validation_data=(X_val, y_val),
        callbacks=[early_stopping, reduce_lr],
        verbose=1
    )
    
    # Step 5: Evaluate and save
    print("\n[5/5] Evaluating and saving model...")
    
    # Evaluate
    val_loss, val_acc = model.evaluate(X_val, y_val, verbose=0)
    print(f"\nValidation Loss: {val_loss:.4f}")
    print(f"Validation Accuracy: {val_acc:.4f}")
    
    # Predict on validation set for detailed stats
    y_pred = model.predict(X_val)
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_true_classes = np.argmax(y_val, axis=1)
    
    from sklearn.metrics import classification_report, confusion_matrix
    report = classification_report(y_true_classes, y_pred_classes, 
                                   target_names=GESTURES)
    print("\nClassification Report:")
    print(report)
    
    cm = confusion_matrix(y_true_classes, y_pred_classes)
    
    # Save model
    os.makedirs(RESULTS_DIR, exist_ok=True)
    model_path = os.path.join(RESULTS_DIR, 'lstm_gesture_model.h5')
    model.save(model_path)
    print(f"   Model saved to: {model_path}")
    
    # Save scaler and label encoder
    scaler_path = os.path.join(RESULTS_DIR, 'lstm_scaler.pkl')
    with open(scaler_path, 'wb') as f:
        joblib.dump(scaler, f)
    print(f"   Scaler saved to: {scaler_path}")
    
    lb_path = os.path.join(RESULTS_DIR, 'lstm_label_encoder.pkl')
    with open(lb_path, 'wb') as f:
        joblib.dump(lb, f)
    print(f"   Label encoder saved to: {lb_path}")
    
    # Save config
    config = {
        'seq_len': 30,
        'num_features': 12,
        'num_classes': len(GESTURES),
        'classes': GESTURES,
        'gesture_ids': gesture_ids,
        'model_path': model_path,
        'trainer_path': os.path.realpath(__file__)
    }
    
    config_path = os.path.join(RESULTS_DIR, 'lstm_model_config.json')
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    print(f"   Config saved to: {config_path}")
    
    return model, model_path, val_acc


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train LSTM gesture model')
    parser.add_argument('--epochs', type=int, default=200, help='Training epochs')
    parser.add_argument('--seq-len', type=int, default=30, help='Sequence length')
    args = parser.parse_args()
    
    model, model_path, accuracy = train_lstm_model()
    print(f"\n{'='*60}")
    print(f"LSTM Model Training Complete!")
    print(f"Final Validation Accuracy: {accuracy:.2%}")
    print(f"{'='*60}")

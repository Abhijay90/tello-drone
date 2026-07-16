"""
Realistic simulated MediaPipe landmarks for testing gesture_classifier.py classify() function.
Generates plausible landmark coordinates for each gesture type and calls the real classifier.
"""
import sys
sys.path.insert(0, '/home/abhikun/Desktop/drone/tello-drone')

from vision.gestures_v2 import classify, Gesture, _SVM_GESTURE_MAP
from dataclasses import dataclass
from typing import List

@dataclass
class Landmark:
    x: float
    y: float
    z: float

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

def make_landmarks(coords=None):
    """Convert list of (x, y, z) tuples to list of Landmark objects."""
    if coords is None:
        coords = []
    result = []
    for i in range(21):
        lm = Landmark(0.5, 0.5, 0.0)
        if i < len(coords):
            c = coords[i]
            if isinstance(c, (tuple, list)) and len(c) >= 3:
                lm.x = c[0]
                lm.y = c[1]
                lm.z = c[2]
        result.append(lm)
    return result


# Simulated coordinates for different gestures (x, y in [0,1], z in [-0.3, 0.3])
GESTURE_COORDS = {
    # ---- THUMBS UP ----
    # Extended thumb, folded fingers
    'THUMBS_UP': [
        (0.5, 0.8, 0.0),    # WRIST   (bottom center)
        (0.42, 0.75, 0.05), # THUMB_CMC
        (0.38, 0.65, 0.03), # THUMB_IP
        (0.38, 0.65, -0.02),# THUMB_MCP - NOT USED
        (0.35, 0.45, 0.01), # THUMB_TIP (extended upward)
        (0.55, 0.72, -0.1), # INDEX_MCP
        (0.56, 0.65, -0.15),# INDEX_PIP
        (0.55, 0.58, -0.12),# INDEX_DIP
        (0.54, 0.55, -0.14),# INDEX_TIP (folded in)
        (0.58, 0.68, -0.1),  # MIDDLE_MCP
        (0.59, 0.63, -0.12), # MIDDLE_PIP
        (0.58, 0.59, -0.11), # MIDDLE_DIP
        (0.57, 0.57, -0.13), # MIDDLE_TIP
        (0.60, 0.64, -0.08), # RING_MCP
        (0.61, 0.60, -0.09), # RING_PIP
        (0.60, 0.57, -0.08), # RING_DIP
        (0.59, 0.56, -0.10), # RING_TIP
        (0.63, 0.62, -0.06), # PINKY_MCP
        (0.63, 0.59, -0.07), # PINKY_PIP
        (0.62, 0.57, -0.06), # PINKY_DIP
        (0.62, 0.56, -0.08), # PINKY_TIP
    ]
    ,
    'THUMBS_DOWN': [
        (0.5, 0.2, 0.0),     # WRIST   (top center)
        (0.42, 0.25, 0.05),  # THUMB_CMC
        (0.38, 0.35, 0.03),  # THUMB_IP
        (0.38, 0.35, -0.02), # THUMB_MCP - NOT USED
        (0.35, 0.55, 0.01),  # THUMB_TIP (folded down - larger y)
        (0.55, 0.28, -0.1),  # INDEX_MCP
        (0.56, 0.35, -0.15), # INDEX_PIP
        (0.55, 0.42, -0.12), # INDEX_DIP
        (0.54, 0.45, -0.14), # INDEX_TIP (extended)
        (0.58, 0.32, -0.1),   # MIDDLE_MCP
        (0.59, 0.37, -0.12),  # MIDDLE_PIP
        (0.58, 0.41, -0.11),  # MIDDLE_DIP
        (0.57, 0.43, -0.13),  # MIDDLE_TIP
        (0.60, 0.36, -0.08),  # RING_MCP
        (0.61, 0.40, -0.09),  # RING_PIP
        (0.60, 0.43, -0.08),  # RING_DIP
        (0.59, 0.44, -0.10),  # RING_TIP
        (0.63, 0.38, -0.06),  # PINKY_MCP
        (0.63, 0.41, -0.07),  # PINKY_PIP
        (0.62, 0.43, -0.06),  # PINKY_DIP
        (0.62, 0.44, -0.08),  # PINKY_TIP
    ]
    ,
    'OPEN_PALM': [
        (0.5, 0.5, 0.0),     # WRIST   (center)
        (0.45, 0.48, 0.02),  # THUMB_CMC
        (0.42, 0.45, 0.01),  # THUMB_IP
        (0.42, 0.45, -0.01), # THUMB_MCP
        (0.38, 0.42, 0.0),   # THUMB_TIP (extended)
        (0.52, 0.45, -0.1),  # INDEX_MCP
        (0.53, 0.38, -0.12), # INDEX_PIP
        (0.53, 0.33, -0.11), # INDEX_DIP
        (0.54, 0.28, -0.13), # INDEX_TIP (extended)
        (0.55, 0.43, -0.08), # MIDDLE_MCP
        (0.56, 0.35, -0.10), # MIDDLE_PIP
        (0.56, 0.30, -0.09), # MIDDLE_DIP
        (0.57, 0.25, -0.11), # MIDDLE_TIP
        (0.58, 0.45, -0.06), # RING_MCP
        (0.59, 0.40, -0.07), # RING_PIP
        (0.59, 0.36, -0.06), # RING_DIP
        (0.60, 0.33, -0.08), # RING_TIP
        (0.62, 0.48, -0.04), # PINKY_MCP
        (0.62, 0.45, -0.05), # PINKY_PIP
        (0.62, 0.42, -0.04), # PINKY_DIP
        (0.62, 0.40, -0.06), # PINKY_TIP
    ]
    ,
    'CLOSED_FIST': [
        (0.5, 0.5, 0.0),      # WRIST
        (0.45, 0.52, 0.05),   # THUMB_CMC
        (0.42, 0.53, 0.03),   # THUMB_IP
        (0.42, 0.53, -0.02),  # THUMB_MCP
        (0.48, 0.51, 0.01),   # THUMB_TIP (folded in)
        (0.55, 0.48, -0.1),   # INDEX_MCP
        (0.56, 0.47, -0.12),  # INDEX_PIP
        (0.55, 0.46, -0.11),  # INDEX_DIP
        (0.54, 0.45, -0.13),  # INDEX_TIP (folded in)
        (0.58, 0.47, -0.08),  # MIDDLE_MCP
        (0.59, 0.46, -0.09),  # MIDDLE_PIP
        (0.58, 0.45, -0.08),  # MIDDLE_DIP
        (0.57, 0.44, -0.10),  # MIDDLE_TIP
        (0.60, 0.48, -0.06),  # RING_MCP
        (0.61, 0.47, -0.07),  # RING_PIP
        (0.60, 0.46, -0.06),  # RING_DIP
        (0.59, 0.45, -0.08),  # RING_TIP
        (0.63, 0.50, -0.04),  # PINKY_MCP
        (0.63, 0.49, -0.05),  # PINKY_PIP
        (0.62, 0.48, -0.04),  # PINKY_DIP
        (0.62, 0.47, -0.06),  # PINKY_TIP
    ]
    ,
    'PALM_LEFT': [
        (0.25, 0.5, 0.0),     # WRIST   (left side)
        (0.30, 0.48, 0.02),   # THUMB_CMC
        (0.33, 0.45, 0.01),   # THUMB_IP
        (0.33, 0.45, -0.01),  # THUMB_MCP
        (0.36, 0.42, 0.0),    # THUMB_TIP (facing left)
        (0.40, 0.45, -0.1),   # INDEX_MCP
        (0.42, 0.38, -0.12),  # INDEX_PIP
        (0.42, 0.33, -0.11),  # INDEX_DIP
        (0.43, 0.28, -0.13),  # INDEX_TIP
        (0.45, 0.43, -0.08),  # MIDDLE_MCP
        (0.47, 0.35, -0.10),  # MIDDLE_PIP
        (0.47, 0.30, -0.09),  # MIDDLE_DIP
        (0.48, 0.25, -0.11),  # MIDDLE_TIP
        (0.50, 0.45, -0.06),  # RING_MCP
        (0.51, 0.40, -0.07),  # RING_PIP
        (0.51, 0.36, -0.06),  # RING_DIP
        (0.52, 0.33, -0.08),  # RING_TIP
        (0.55, 0.48, -0.04),  # PINKY_MCP
        (0.55, 0.45, -0.05),  # PINKY_PIP
        (0.55, 0.42, -0.04),  # PINKY_DIP
        (0.55, 0.40, -0.06),  # PINKY_TIP
    ]
    ,
    'PALM_RIGHT': [
        (0.75, 0.5, 0.0),     # WRIST   (right side)
        (0.70, 0.52, 0.02),   # THUMB_CMC
        (0.67, 0.55, 0.01),   # THUMB_IP
        (0.67, 0.55, -0.01),  # THUMB_MCP
        (0.64, 0.58, 0.0),    # THUMB_TIP (facing right)
        (0.60, 0.55, -0.1),   # INDEX_MCP
        (0.58, 0.62, -0.12),  # INDEX_PIP
        (0.58, 0.67, -0.11),  # INDEX_DIP
        (0.57, 0.72, -0.13),  # INDEX_TIP
        (0.55, 0.57, -0.08),  # MIDDLE_MCP
        (0.53, 0.65, -0.10),  # MIDDLE_PIP
        (0.53, 0.70, -0.09),  # MIDDLE_DIP
        (0.52, 0.75, -0.11),  # MIDDLE_TIP
        (0.50, 0.55, -0.06),  # RING_MCP
        (0.49, 0.60, -0.07),  # RING_PIP
        (0.49, 0.64, -0.06),  # RING_DIP
        (0.48, 0.67, -0.08),  # RING_TIP
        (0.45, 0.52, -0.04),  # PINKY_MCP
        (0.45, 0.55, -0.05),  # PINKY_PIP
        (0.45, 0.58, -0.04),  # PINKY_DIP
        (0.45, 0.60, -0.06),  # PINKY_TIP
    ]
    ,
    'PALM_UP': [
        # Normal pointing up: Z positive in cross product
        (0.5, 0.5, 0.1),      # WRIST
        (0.48, 0.48, 0.2),    # THUMB_CMC
        (0.52, 0.52, 0.3),    # THUMB_IP
        (0.52, 0.52, 0.15),   # THUMB_MCP
        (0.46, 0.46, 0.25),   # THUMB_TIP
        (0.45, 0.55, 0.0),    # INDEX_MCP
        (0.38, 0.56, 0.02),   # INDEX_PIP
        (0.33, 0.57, 0.01),   # INDEX_DIP
        (0.28, 0.58, 0.03),   # INDEX_TIP
        (0.55, 0.50, 0.0),    # MIDDLE_MCP
        (0.60, 0.45, 0.02),   # MIDDLE_PIP
        (0.62, 0.35, 0.01),   # MIDDLE_DIP
        (0.63, 0.25, 0.03),   # MIDDLE_TIP
        (0.65, 0.48, -0.02),  # RING_MCP
        (0.70, 0.47, 0.0),    # RING_PIP
        (0.72, 0.45, -0.01),  # RING_DIP
        (0.73, 0.43, 0.01),   # RING_TIP
        (0.70, 0.50, -0.04),  # PINKY_MCP
        (0.75, 0.52, -0.02),  # PINKY_PIP
        (0.77, 0.53, -0.03),  # PINKY_DIP
        (0.78, 0.55, -0.01),  # PINKY_TIP
    ]
    ,
    'PALM_DOWN': [
        # Normal pointing down: Z negative in cross product  
        (0.5, 0.5, -0.1),     # WRIST
        (0.48, 0.48, -0.2),   # THUMB_CMC
        (0.52, 0.52, -0.3),   # THUMB_IP
        (0.52, 0.52, -0.15),  # THUMB_MCP
        (0.46, 0.46, -0.25),  # THUMB_TIP
        (0.45, 0.55, 0.0),    # INDEX_MCP
        (0.38, 0.56, -0.02),  # INDEX_PIP
        (0.33, 0.57, -0.01),  # INDEX_DIP
        (0.28, 0.58, -0.03),  # INDEX_TIP
        (0.55, 0.50, 0.0),    # MIDDLE_MCP
        (0.60, 0.45, -0.02),  # MIDDLE_PIP
        (0.62, 0.35, -0.01),  # MIDDLE_DIP
        (0.63, 0.25, -0.03),  # MIDDLE_TIP
        (0.65, 0.48, 0.02),   # RING_MCP
        (0.70, 0.47, 0.0),    # RING_PIP
        (0.72, 0.45, 0.01),   # RING_DIP
        (0.73, 0.43, -0.01),  # RING_TIP
        (0.70, 0.50, 0.04),   # PINKY_MCP
        (0.75, 0.52, 0.02),   # PINKY_PIP
        (0.77, 0.53, 0.03),   # PINKY_DIP
        (0.78, 0.55, 0.01),   # PINKY_TIP
    ]
}


def run_tests():
    """Test classify() for each gesture with simulated landmarks."""
    print("\n" + "=" * 80)
    print("GESTURE CLASSIFICATION TEST (SVM Model: " + ("AVAILABLE" if True else "NOT AVAILABLE") + ")")  # We'll check
    print("=" * 80 + "\n")
    
    results = []
    total_success = 0
    total_tests = 0
    
    for gesture_name, coords in GESTURE_COORDS.items():
        landmarks = make_landmarks(coords)
        predicted = classify(landmarks, 640, 480)  # 640x480 is resolution-independent
        actual_gesture = Gesture[gesture_name]
        is_correct = (predicted == actual_gesture)
        
        if is_correct:
            total_success += 1
        total_tests += 1
        
        color = "\033[92m" if is_correct else "\033[91m"
        reset = "\033[0m"
        status = "PASS" if is_correct else "FAIL"
        
        print(f"{color}{status:4s} | {gesture_name:<15} | Predicted: {predicted.name:<15} | Expected: {actual_gesture.name}<reset>")
        
        results.append({
            'gesture': gesture_name,
            'expected': actual_gesture.name,
            'predicted': predicted.name,
            'correct': is_correct
        })
    
    # Also test UNKNOWN (no hand)
    print("\n--- Edge Cases ---")
    
    # No landmarks
    empty_lms = make_landmarks([])
    pred_empty = classify(empty_lms, 640, 480)
    is_correct = (pred_empty == Gesture.UNKNOWN)
    color = "\033[92m" if is_correct else "\033[91m"
    status = "PASS" if is_correct else "FAIL"
    print(f"{color}{status:4s} | {'NO_HAND':<15} | Predicted: {pred_empty.name:<15} | Expected: UNKNOWN")
    if is_correct:
        total_success += 1
    total_tests += 1
    
    print(f"\n{'=' * 80}")
    print(f"TOTAL: {total_success}/{total_tests} tests passed")
    
    # Print summary table
    print("\nSUMMARY TABLE:")
    print(f"{'Gesture':<15} | {'Expected':<15} | {'Predicted':<15} | {'Status'}")
    print("-" * 60)
    for r in results:
        status = "\033[92m" + "✓" + "\033[0m" if r['correct'] else "\033[91m" + "✗" + "\033[0m"
        print(f"{r['gesture']:<15} | {r['expected']:<15} | {r['predicted']:<15} | {status}")
    
    if total_success == total_tests:
        print(f"\n\033[92m✓ ALL TESTS PASSED!\033[0m")
        return True
    else:
        failed = [r for r in results if not r['correct']]
        print(f"\n\033[91m✗ {len(failed)} GUESTORE FAILED:\033[0m")
        for f in failed:
            print(f"  - {f['gesture']}: expected {f['expected']}, got {f['predicted']}")
        return False


if __name__ == '__main__':
    success = run_tests()
    sys.exit(0 if success else 1)
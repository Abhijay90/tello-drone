"""Standalone test of gesture_classifier.py with synthetic landmark data."""
import sys, math
sys.path.insert(0, '../src')
from gesture_classifier import classify, Gesture

class LM:
    __slots__ = ['x','y','z']
    def __init__(self, x, y, z): self.x = x; self.y = y; self.z = z

def l(x, y, z): return LM(x, y, z)

def landmarks(
    wr=(0.5,0.95,0.3),
    thumb_cm=(0.5,0.90,0.35), thumb_ip=(0.5,0.85,0.35), thumb_mp=(0.55,0.88,0.35), thumb_tp=(0.45,0.78,0.30),
    idx_mcp=(0.46,0.65,0.3), idx_pip=(0.46,0.45,0.3), idx_m3=(0.46,0.30,0.3), idx_tip=(0.46,0.10,0.3),
    mid_mcp=(0.50,0.65,0.3), mid_pip=(0.50,0.45,0.3), mid_m3=(0.50,0.30,0.3), mid_m4=(0.50,0.15,0.3), mid_t=(0.50,0.05,0.3),
    ring_mcp=(0.50,0.55,0.3), ring_pip=(0.50,0.40,0.3), ring_m3=(0.50,0.25,0.3), ring_m4=(0.50,0.15,0.3), ring_t=(0.50,0.05,0.3),
    pink_mcp=(0.54,0.65,0.3), pink_pip=(0.54,0.50,0.3), pink_m3=(0.54,0.35,0.3), pink_t=(0.54,0.20,0.3)
):
    # 0=WRIST, 1=THUMB_CMC, 2=THUMB_IP, 3=THUMB_MCP, 4=THUMB_TIP
    # 5=INDEX_MCP, 6=INDEX_PIP, 7=INDEX_MCP3, 8=INDEX_TIP
    # 9=MIDDLE_MCP, 10=MIDDLE_PIP, 11=MIDDLE_MCP3, 12=MIDDLE_MCP4, 13=MIDDLE_TIP? No...
    # MediaPipe indices: 0=WRIST, 1=THUMB_CMC, 2=THUMB_IP, 3=THUMB_MCP, 4=THUMB_TIP
    # 5=INDEX_MCP, 6=INDEX_PIP, 7=INDEX_MCP3, 8=INDEX_TIP
    # 9=MIDDLE_MCP, 10=MIDDLE_PIP, 11=MIDDLE_MCP3, 12=MIDDLE_MCP4, 13=RING_MCP
    # 14=RING_PIP, 15=RING_MCP3, 16=RING_TIP, 17=PINKY_MCP
    # 18=PINKY_PIP, 19=PINKY_MCP3, 20=PINKY_MCP4
    return [
        l(*wr),             # 0
        l(*thumb_cm),       # 1
        l(*thumb_ip),       # 2
        l(*thumb_mp),       # 3
        l(*thumb_tp),       # 4
        l(*idx_mcp),        # 5
        l(*idx_pip),        # 6
        l(*idx_m3),         # 7
        l(*idx_tip),        # 8
        l(*mid_mcp),        # 9
        l(*mid_pip),        # 10
        l(*mid_m3),         # 11
        l(*mid_m4),         # 12
        l(*ring_mcp),       # 13
        l(*ring_pip),       # 14
        l(*ring_m3),        # 15
        l(*ring_t),         # 16
        l(*pink_mcp),       # 17
        l(*pink_pip),       # 18
        l(*pink_m3),        # 19
        l(*pink_t),         # 20
    ]

def print_result(gesture_name, detected):
    status = "✅" if gest_names.get(gesture_name) == detected else "❌"
    print(f"  {status} {gesture_name:<15} → {detected:<15} (expected: {gest_names.get(gesture_name, '???')})")

gest_names = {
    'OPEN_PALM': Gesture.OPEN_PALM,
    'CLOSED_FIST': Gesture.CLOSE_PALM,
    'THUMB_UP': Gesture.THUMB_UP,
    'THUMB_DOWN': Gesture.THUMB_DOWN,
    'PALM_LEFT': Gesture.PALM_LEFT,
    'PALM_RIGHT': Gesture.PALM_RIGHT,
    'PALM_UP': Gesture.PALM_DOWN,  # palm facing down
    'PALM_DOWN': Gesture.PALM_UP,  # palm facing up (camera-facing)
    'UNKNOWN': Gesture.POINT,
    'PINCH': Gesture.PINCH,
    'POINT': Gesture.POINT,
}

print("=" * 70)
print("GESTURE CLASSIFIER STANDALONE TEST")
print("=" * 70)

tests = []

# 1. OPEN_PALM - all fingers extended, palm facing camera
def test_open_palm():
    pts = landmarks(
        wr=(0.5,0.95,0.3),
        thumb_tp=(0.48, 0.80, 0.30),  # thumb folded slightly
        idx_tip=(0.46, 0.10, 0.3), mid_t=(0.50, 0.00, 0.3),
        ring_t=(0.50, -0.10, 0.3), pink_t=(0.54, 0.10, 0.3),
    )
    # Make all fingers extended by checking if tips are farther from wrist than PIPs
    # Actually the classifier uses _is_finger_extended which checks wrist-to-tip vs wrist-to-PIP distance
    # Let me verify each finger's tip is farther from wrist than its PIP
    # Default setup should have extended fingers
    return pts

# Simpler approach: build landmarks that the classifier clearly accepts/rejects

# MediaPipe coordinate system: y increases downward (0=top, 1=bottom), z=depth, x left-right
# But in the classifier code, _is_finger_extended compares wrist-to-tip distance vs wrist-to-PIP
# So for a finger to be "extended", tip must be farther from wrist than PIP

# For OPEN_PALM: all 4 fingers extended, palm facing camera
def test_open_palm():
    pts = landmarks(
        wr=(0.5, 0.90, 0.5),
        thumb_cm=(0.5, 0.85, 0.5), thumb_ip=(0.48, 0.80, 0.5), thumb_mp=(0.52, 0.82, 0.5), thumb_tp=(0.45, 0.70, 0.45),
        idx_mcp=(0.45, 0.70, 0.5), idx_pip=(0.45, 0.50, 0.5), idx_m3=(0.45, 0.35, 0.5), idx_tip=(0.45, 0.15, 0.5),
        mid_mcp=(0.50, 0.70, 0.5), mid_pip=(0.50, 0.50, 0.5), mid_m3=(0.50, 0.30, 0.5), mid_m4=(0.50, 0.15, 0.5), mid_t=(0.50, -0.05, 0.5),
        ring_mcp=(0.55, 0.70, 0.5), ring_pip=(0.55, 0.50, 0.5), ring_m3=(0.55, 0.35, 0.5), ring_m4=(0.55, 0.20, 0.5), ring_t=(0.55, 0.00, 0.5),
        pink_mcp=(0.60, 0.70, 0.5), pink_pip=(0.60, 0.55, 0.5), pink_m3=(0.60, 0.40, 0.5), pink_t=(0.60, 0.25, 0.5),
    )
    return pts

# CLOSED_FIST - all fingers folded
def test_closed_fist():
    pts = landmarks(
        wr=(0.5, 0.90, 0.5),
        thumb_cm=(0.5, 0.85, 0.5), thumb_ip=(0.48, 0.82, 0.6), thumb_mp=(0.52, 0.85, 0.7), thumb_tp=(0.48, 0.75, 0.65),
        idx_mcp=(0.45, 0.65, 0.5), idx_pip=(0.43, 0.55, 0.6), idx_m3=(0.42, 0.48, 0.65), idx_tip=(0.44, 0.58, 0.7),
        mid_mcp=(0.50, 0.65, 0.5), mid_pip=(0.48, 0.55, 0.6), mid_m3=(0.47, 0.48, 0.65), mid_m4=(0.46, 0.45, 0.7), mid_t=(0.48, 0.55, 0.75),
        ring_mcp=(0.55, 0.65, 0.5), ring_pip=(0.53, 0.55, 0.6), ring_m3=(0.52, 0.48, 0.65), ring_m4=(0.51, 0.45, 0.7), ring_t=(0.53, 0.55, 0.75),
        pink_mcp=(0.60, 0.65, 0.5), pink_pip=(0.58, 0.55, 0.6), pink_m3=(0.57, 0.48, 0.65), pink_t=(0.59, 0.55, 0.75),
    )
    return pts

# THUMB_UP - thumb pointing up
def test_thumb_up():
    pts = landmarks(
        wr=(0.5, 0.90, 0.5),
        thumb_cm=(0.5, 0.85, 0.5), thumb_ip=(0.48, 0.82, 0.5), thumb_mp=(0.52, 0.85, 0.5), thumb_tp=(0.45, 1.00, 0.5),
        idx_mcp=(0.45, 0.70, 0.5), idx_pip=(0.45, 0.50, 0.5), idx_m3=(0.45, 0.35, 0.5), idx_tip=(0.45, 0.15, 0.5),
        mid_mcp=(0.50, 0.70, 0.5), mid_pip=(0.50, 0.50, 0.5), mid_m3=(0.50, 0.30, 0.5), mid_m4=(0.50, 0.15, 0.5), mid_t=(0.50, -0.05, 0.5),
        ring_mcp=(0.55, 0.70, 0.5), ring_pip=(0.55, 0.50, 0.5), ring_m3=(0.55, 0.35, 0.5), ring_m4=(0.55, 0.20, 0.5), ring_t=(0.55, 0.00, 0.5),
        pink_mcp=(0.60, 0.70, 0.5), pink_pip=(0.60, 0.55, 0.5), pink_m3=(0.60, 0.40, 0.5), pink_t=(0.60, 0.25, 0.5),
    )
    return pts

# THUMB_DOWN - thumb pointing down
def test_thumb_down():
    pts = landmarks(
        wr=(0.5, 0.10, 0.5),  # wrist at top
        thumb_cm=(0.5, 0.15, 0.5), thumb_ip=(0.48, 0.18, 0.5), thumb_mp=(0.52, 0.15, 0.5), thumb_tp=(0.45, 0.00, 0.5),
        idx_mcp=(0.45, 0.30, 0.5), idx_pip=(0.45, 0.50, 0.5), idx_m3=(0.45, 0.65, 0.5), idx_tip=(0.45, 0.85, 0.5),
        mid_mcp=(0.50, 0.30, 0.5), mid_pip=(0.50, 0.50, 0.5), mid_m3=(0.50, 0.70, 0.5), mid_m4=(0.50, 0.85, 0.5), mid_t=(0.50, 1.05, 0.5),
        ring_mcp=(0.55, 0.30, 0.5), ring_pip=(0.55, 0.50, 0.5), ring_m3=(0.55, 0.65, 0.5), ring_m4=(0.55, 0.80, 0.5), ring_t=(0.55, 1.00, 0.5),
        pink_mcp=(0.60, 0.30, 0.5), pink_pip=(0.60, 0.45, 0.5), pink_m3=(0.60, 0.60, 0.5), pink_t=(0.60, 0.75, 0.5),
    )
    return pts

print("\n1. Testing OPEN_PALM...")
detected = classify(test_open_palm())
print_result("OPEN_PALM", detected)

print("\n2. Testing CLOSED_FIST...")
detected = classify(test_closed_fist())
print_result("CLOSED_FIST", detected)

print("\n3. Testing THUMB_UP...")
detected = classify(test_thumb_up())
print_result("THUMB_UP", detected)

print("\n4. Testing THUMB_DOWN...")
detected = classify(test_thumb_down())
print_result("THUMB_DOWN", detected)

print("\n" + "=" * 70)
print("TEST COMPLETE")
print("=" * 70)

"""Test gestures_v2.py SVM classifier with synthetic landmarks for all 8 classes."""
import sys, math, types
sys.path.insert(0, '/home/abhikun/Desktop/drone/tello-drone')

class Lm:
    def __init__(self, x=0, y=0, z=0): self.x = x; self.y = y; self.z = z

try:
    from vision.gestures_v2 import classify, Gesture, _MODEL_AVAILABLE
except Exception as e:
    print(f"FAIL: cannot import: {e}")
    sys.exit(1)

print(f"SVM model available: {_MODEL_AVAILABLE}")
print()

results = []

def check(name, expected, actual):
    ok = expected == actual
    status = "PASS" if ok else "FAIL"
    results.append(ok)
    print(f"  [{status}] {name}: expected={expected}, got={actual}")

# 1. THUMBS_UP — thumb extended, 4 fingers folded
#    thumb tip far from palm, MCPs close together (folded)
lm = [Lm()] * 21
lm[0] = Lm(0.50, 0.80); lm[4] = Lm(0.50, 0.10)  # thumb extended up (z=0)
lm[1] = Lm(0.50, 0.75); lm[2] = Lm(0.50, 0.45)
lm[5] = Lm(0.45, 0.55); lm[6] = Lm(0.45, 0.55)  # index folded (MCP~PIP)
lm[9] = Lm(0.52, 0.55); lm[10] = Lm(0.52, 0.55) # middle folded
lm[13] = Lm(0.60, 0.57); lm[14] = Lm(0.60, 0.57) # ring folded
lm[17] = Lm(0.66, 0.58); lm[18] = Lm(0.66, 0.58) # pinky folded
lm[8] = Lm(0.45, 0.35); lm[12] = Lm(0.50, 0.35)
lm[16] = Lm(0.60, 0.38); lm[20] = Lm(0.68, 0.38)
check("THUMBS_UP", Gesture.THUMBS_UP, classify(lm, 320, 240))

# 2. OPEN_PALM — all fingers extended, palm horizontal
lm = [Lm()] * 21
lm[0] = Lm(0.50, 0.60); lm[4] = Lm(0.35, 0.55)  # thumb extended left
lm[5] = Lm(0.70, 0.62); lm[6] = Lm(0.80, 0.58); lm[8] = Lm(0.90, 0.55)  # index open
lm[9] = Lm(0.65, 0.58); lm[10] = Lm(0.72, 0.55); lm[12] = Lm(0.82, 0.52)  # middle open
lm[13] = Lm(0.60, 0.58); lm[14] = Lm(0.66, 0.55); lm[16] = Lm(0.72, 0.52)  # ring open
lm[17] = Lm(0.56, 0.60); lm[18] = Lm(0.60, 0.57); lm[20] = Lm(0.64, 0.54)  # pinky open
check("OPEN_PALM", Gesture.OPEN_PALM, classify(lm, 320, 240))

# 3. CLOSED_FIST — all fingers folded
lm = [Lm()] * 21
lm[0] = Lm(0.50, 0.60)
lm[4] = Lm(0.52, 0.58); lm[2] = Lm(0.51, 0.59)
lm[6] = Lm(0.70, 0.60); lm[5] = Lm(0.70, 0.60)
lm[10] = Lm(0.65, 0.58); lm[9] = Lm(0.65, 0.58)
lm[14] = Lm(0.60, 0.58); lm[13] = Lm(0.60, 0.58)
lm[18] = Lm(0.57, 0.59); lm[17] = Lm(0.57, 0.59)
lm[8] = Lm(0.70, 0.60)
lm[12] = Lm(0.65, 0.58)
lm[16] = Lm(0.60, 0.58)
lm[20] = Lm(0.57, 0.59)
check("CLOSED_FIST", Gesture.CLOSED_FIST, classify(lm, 320, 240))

# 4. THUMBS_DOWN — thumb folded, thumb tip below IP, all 4 fingers extended
lm = [Lm()] * 21
lm[0] = Lm(0.50, 0.70); lm[4] = Lm(0.50, 0.78)  # thumb tip below IP
lm[1] = Lm(0.50, 0.68); lm[2] = Lm(0.50, 0.75)
lm[5] = Lm(0.70, 0.52); lm[6] = Lm(0.80, 0.48); lm[8] = Lm(0.90, 0.45)
lm[9] = Lm(0.65, 0.50); lm[10] = Lm(0.72, 0.46); lm[12] = Lm(0.82, 0.43)
lm[13] = Lm(0.60, 0.50); lm[14] = Lm(0.66, 0.46); lm[16] = Lm(0.72, 0.43)
lm[17] = Lm(0.56, 0.52); lm[18] = Lm(0.60, 0.48); lm[20] = Lm(0.64, 0.45)
check("THUMBS_DOWN", Gesture.THUMBS_DOWN, classify(lm, 320, 240))

# 5. PALM_UP — palm normal positive (facing up)
lm = [Lm()] * 21
lm[0] = Lm(0.50, 0.60); lm[4] = Lm(0.55, 0.58)
lm[5] = Lm(0.70, 0.58); lm[6] = Lm(0.80, 0.55); lm[8] = Lm(0.90, 0.50)  # index up in image
lm[9] = Lm(0.65, 0.56); lm[10] = Lm(0.72, 0.52); lm[12] = Lm(0.82, 0.48)  # middle up
lm[13] = Lm(0.60, 0.56); lm[14] = Lm(0.66, 0.52); lm[16] = Lm(0.72, 0.48)  # ring up
lm[17] = Lm(0.56, 0.55); lm[18] = Lm(0.60, 0.51); lm[20] = Lm(0.64, 0.47)  # pinky up
# Give fingers a z-depth pattern to create palm_up orientation
for i in [4, 6, 8, 10, 12, 14, 16, 18, 20]:
    lm[i].z = 0.1
check("PALM_UP", Gesture.PALM_UP, classify(lm, 320, 240))

# 6. PALM_DOWN — palm normal negative (facing down)
lm = [Lm()] * 21
lm[0] = Lm(0.50, 0.60); lm[4] = Lm(0.55, 0.58)
lm[5] = Lm(0.70, 0.58); lm[6] = Lm(0.80, 0.55); lm[8] = Lm(0.90, 0.50)
lm[9] = Lm(0.65, 0.56); lm[10] = Lm(0.72, 0.52); lm[12] = Lm(0.82, 0.48)
lm[13] = Lm(0.60, 0.56); lm[14] = Lm(0.66, 0.52); lm[16] = Lm(0.72, 0.48)
lm[17] = Lm(0.56, 0.55); lm[18] = Lm(0.60, 0.51); lm[20] = Lm(0.64, 0.47)
# Negative z for palm_down
for i in [4, 6, 8, 10, 12, 14, 16, 18, 20]:
    lm[i].z = -0.1
check("PALM_DOWN", Gesture.PALM_DOWN, classify(lm, 320, 240))

# 7. PALM_LEFT — thumb on left side, horizontal hand
lm = [Lm()] * 21
lm[0] = Lm(0.50, 0.60); lm[4] = Lm(0.30, 0.55)  # thumb far left
lm[5] = Lm(0.70, 0.62); lm[6] = Lm(0.80, 0.58); lm[8] = Lm(0.90, 0.55)
lm[9] = Lm(0.65, 0.58); lm[10] = Lm(0.72, 0.55); lm[12] = Lm(0.82, 0.52)
lm[13] = Lm(0.60, 0.58); lm[14] = Lm(0.66, 0.55); lm[16] = Lm(0.72, 0.52)
lm[17] = Lm(0.56, 0.60); lm[18] = Lm(0.60, 0.57); lm[20] = Lm(0.64, 0.54)
check("PALM_LEFT", Gesture.PALM_LEFT, classify(lm, 320, 240))

# 8. PALM_RIGHT — thumb on right side, horizontal hand
lm = [Lm()] * 21
lm[0] = Lm(0.50, 0.60); lm[4] = Lm(0.70, 0.55)  # thumb far right
lm[5] = Lm(0.30, 0.62); lm[6] = Lm(0.20, 0.58); lm[8] = Lm(0.10, 0.55)
lm[9] = Lm(0.35, 0.58); lm[10] = Lm(0.28, 0.55); lm[12] = Lm(0.18, 0.52)
lm[13] = Lm(0.40, 0.58); lm[14] = Lm(0.34, 0.55); lm[16] = Lm(0.28, 0.52)
lm[17] = Lm(0.44, 0.60); lm[18] = Lm(0.40, 0.57); lm[20] = Lm(0.36, 0.54)
check("PALM_RIGHT", Gesture.PALM_RIGHT, classify(lm, 320, 240))

# Summary
n = len(results)
ps = sum(results)
print(f"\n{'='*50}")
print(f"  TOTAL: {ps}/{n} pass ✓  |  {n-ps}/{n} fail ✗")
print(f"{'='*50}")
if ps == n:
    print("  ALL GESTURES CLASSIFIED CORRECTLY!")
else:
    print("  FAILURES DETECTED — see above")
    sys.exit(1)

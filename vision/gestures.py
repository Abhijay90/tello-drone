# vision/gestures.py — Pure gesture classifier
import cv2

class Gesture:
    """Gesture string constants."""
    OPEN_PALM = "open_palm"
    CLOSED_FIST = "closed_fist"
    THUMBS_UP = "thumbs_up"
    THUMBS_DOWN = "thumbs_down"
    PALM_DOWN = "palm_down"
    PALM_UP = "palm_up"
    UNKNOWN = "unknown"


# Thresholds (tunable)
THUMB_EXTEND_PX = 60
FINGER_EXTEND_PX = 20
PALM_ORIENT_OFFSET = 30


def _euclidean(lm1, lm2, img_w, img_h):
    """Euclidean distance in pixels between two landmarks."""
    x1, y1 = lm1.x * img_w, lm1.y * img_h
    x2, y2 = lm2.x * img_w, lm2.y * img_h
    return ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5


def get_thumb_distance(landmarks, img_w, img_h):
    """Distance from thumb tip (4) to wrist (0) in pixels."""
    return _euclidean(landmarks[4], landmarks[0], img_w, img_h)


def get_finger_count(landmarks, img_w, img_h):
    """Return (extended_count, total_count) for 4 fingers.
    Finger is extended if tip-MCP distance >= FINGER_EXTEND_PX."""
    tips = [8, 12, 16, 20]
    mcps = [5, 9, 13, 17]
    extended = sum(1 for tip, mcp in zip(tips, mcps)
                   if _euclidean(landmarks[tip], landmarks[mcp], img_w, img_h) >= FINGER_EXTEND_PX)
    return (extended, 4)


def classify(landmarks, img_w, img_h):
    """Classify hand gesture from landmarks. Returns a Gesture string."""
    thumb_dist = get_thumb_distance(landmarks, img_w, img_h)
    finger_ext, _ = get_finger_count(landmarks, img_w, img_h)
    thumb_is_extended = thumb_dist >= THUMB_EXTEND_PX
    all_fingers = (finger_ext == 4)
    no_fingers = (finger_ext == 0)

    # Check palm orientation
    palm_wrist = landmarks[0]
    palm_mcp5 = landmarks[5]
    wrist_y = palm_wrist.y * img_h
    mcp5_y = palm_mcp5.y * img_h

    if not thumb_is_extended and all_fingers:
        return Gesture.THUMBS_DOWN
    elif thumb_is_extended and no_fingers:
        return Gesture.THUMBS_UP
    elif thumb_is_extended and all_fingers:
        if wrist_y > mcp5_y + PALM_ORIENT_OFFSET:
            return Gesture.PALM_DOWN
        elif wrist_y < mcp5_y - PALM_ORIENT_OFFSET:
            return Gesture.PALM_UP
        else:
            return Gesture.OPEN_PALM
    elif not thumb_is_extended and no_fingers:
        return Gesture.CLOSED_FIST
    else:
        return Gesture.UNKNOWN


# == Debug utilities ==
_debug_frame_count = 0
_debug_prev_gesture = Gesture.UNKNOWN
_DEBUG_DIR = "/home/abhikun/Desktop/drone/tello-drone/debug_frames"
import os
os.makedirs(_DEBUG_DIR, exist_ok=True)


def print_thresholds(landmarks, img_w, img_h):
    """Debug: print raw threshold values to terminal (no gesture logic)."""
    thumb_dist = get_thumb_distance(landmarks, img_w, img_h)
    finger_ext, _ = get_finger_count(landmarks, img_w, img_h)
    wrist_y = landmarks[0].y * img_h
    mcp5_y = landmarks[5].y * img_h
    palm_orient = abs(wrist_y - mcp5_y)

    print("  --- THRESHOLDS ---")
    print(f"  THUMB_DIST: {thumb_dist:.1f} px  (threshold={THUMB_EXTEND_PX})")
    print(f"  FINGER_EXT: {finger_ext}/4  (threshold={FINGER_EXTEND_PX} px)")
    print(f"  PALM_ORIENT: {palm_orient:.1f} px (threshold={PALM_ORIENT_OFFSET})")
    print(f"  RESULT: {classify(landmarks, img_w, img_h)}")


def classify_and_debug(landmarks, img_w, img_h, debug_img=None):
    """Classify gesture with debug output. Saves proof frames when gesture changes."""
    global _debug_frame_count
    global _debug_prev_gesture
    
    thumb_dist = get_thumb_distance(landmarks, img_w, img_h)
    finger_ext, finger_total = get_finger_count(landmarks, img_w, img_h)
    thumb_is_extended = thumb_dist >= THUMB_EXTEND_PX
    all_fingers = (finger_ext == 4)
    no_fingers = (finger_ext == 0)
    
    result = classify(landmarks, img_w, img_h)
    _debug_frame_count += 1
    
    # Print per-frame detailed debug
    finger_names = ['Index', 'Middle', 'Ring', 'Pinky']
    all_finger_dists = []
    for i, (tip, mcp) in enumerate(zip([8, 12, 16, 20], [5, 9, 13, 17])):
        x1, y1 = landmarks[tip].x * img_w, landmarks[tip].y * img_h
        x2, y2 = landmarks[mcp].x * img_w, landmarks[mcp].y * img_h
        d = ((x2-x1)**2 + (y2-y1)**2) ** 0.5
        all_finger_dists.append(d)
    
    # Print detailed info when gesture changes
    if result != _debug_prev_gesture:
        print(f"\n{'='*60}")
        print(f"FRAME #{_debug_frame_count} — GESTURE CHANGED: {_debug_prev_gesture} → {result}")
        print(f"{'='*60}")
        print(f"  THUMB_DIST:  {thumb_dist:.1f} px (threshold={THUMB_EXTEND_PX}, extended={'YES' if thumb_is_extended else 'NO'})")
        for i, name in enumerate(finger_names):
            d = all_finger_dists[i]
            folded = "FOLDED" if d < 20 else "EXTENDED"
            print(f"  {name:6s} tip→MCP: {d:6.1f} px  [{folded}]")
        print(f"  FINGER_EXT: {finger_ext}/{finger_total} (threshold=30)")
        print(f"  NO_FINGERS: {no_fingers}, ALL_FINGERS: {all_fingers}")
        
        if result == Gesture.CLOSED_FIST:
            print(f"  ✓ CLOSED FIST DETECTED!")
            print(f"  → thumb_extended={thumb_is_extended}, finger_ext={finger_ext}")
            print(f"  → To fix: LOWER FINGER_EXTEND_PX from 30 to ~15-20")
        elif result != Gesture.UNKNOWN:
            print(f"  → Expected gesture check: verify hand position matches gesture")
        
        # Save proof frame if provided
        if debug_img is not None:
            _debug_frame_count += 1
            proof_path = os.path.join(_DEBUG_DIR, f"detection_{result}_{_debug_frame_count}.jpg")
            cv2.imwrite(proof_path, debug_img)
            print(f"  → Proof saved: {proof_path}")
        
        _debug_prev_gesture = result
    
    return result


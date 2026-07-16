"""
Gesture-to-RC mapping dataset for tello_handtrack.py.

Usage:
    python gesture_tester.py           # Run all 300 frames
    python gesture_tester.py --list    # List the gesture set only
"""

from __future__ import annotations
import json
import sys

# === Gesture enum (must match gestures_v2.py) ===


class Gesture:
    OPEN_PALM = 1
    CLOSED_FIST = 2
    THUMBS_UP = 3
    THUMBS_DOWN = 4
    PALM_UP = 5
    PALM_DOWN = 6
    PALM_LEFT = 7
    PALM_RIGHT = 8
    UNKNOWN = 9

GESTURE_NAMES = {
    Gesture.OPEN_PALM: "OPEN_PALM",
    Gesture.CLOSED_FIST: "CLOSED_FIST",
    Gesture.THUMBS_UP: "THUMBS_UP",
    Gesture.THUMBS_DOWN: "THUMBS_DOWN",
    Gesture.PALM_UP: "PALM_UP",
    Gesture.PALM_DOWN: "PALM_DOWN",
    Gesture.PALM_LEFT: "PALM_LEFT",
    Gesture.PALM_RIGHT: "PALM_RIGHT",
    Gesture.UNKNOWN: "UNKNOWN",
}

GESTURE_MAP = {k: v for k, v in GESTURE_NAMES.items()}

# === 1: Gesture → RC speed mapping ===


RC_SPEEDS = {
    "OPEN_PALM": "(0, 0)",
    "CLOSED_FIST": "(0, 0)",
    "THUMBS_UP": "(0, 1)",
    "THUMBS_DOWN": "(0, -1)",
    "PALM_UP": "(0, 0)",
    "PALM_DOWN": "(0, 0)",
    "PALM_LEFT": "(-1, 0)",
    "PALM_RIGHT": "(1, 0)",
    "UNKNOWN": "(0, 0)",
}

# === 2: Gesture → overlay labels ===


OVERLAY = {}
OVERLAY_ARROW = {
    "OPEN_PALM": "\u2299",
    "CLOSED_FIST": "\u2299",
    "THUMBS_UP": "\u2191",
    "THUMBS_DOWN": "\u2193",
    "PALM_UP": "\u2299",
    "PALM_DOWN": "\u2299",
    "PALM_LEFT": "\u2190",
    "PALM_RIGHT": "\u2192",
    "UNKNOWN": "\u25cb",
}

OVERLAY_CMD = {
    "OPEN_PALM": "HOVER",
    "CLOSED_FIST": "HOVER",
    "THUMBS_UP": "UP",
    "THUMBS_DOWN": "DOWN",
    "PALM_UP": "HOVER",
    "PALM_DOWN": "HOVER",
    "PALM_LEFT": "LEFT",
    "PALM_RIGHT": "RIGHT",
    "UNKNOWN": "NO TRACK",
}

# === 3: Simulated frames (10 frames per gesture) ===


FRAMES = [
    {"gesture": "THUMBS_UP", "speed_y": 0, "speed_z": 1, "arrow": "\u2191", "cmd": "UP", "desc": "Thumb extended, all fingers folded"},
    {"gesture": "THUMBS_UP", "speed_y": 0, "speed_z": 1, "arrow": "\u2191", "cmd": "UP", "desc": "Thumb extended, all fingers folded"},
    {"gesture": "THUMBS_UP", "speed_y": 0, "speed_z": 1, "arrow": "\u2191", "cmd": "UP", "desc": "Thumb extended, all fingers folded"},
    {"gesture": "THUMBS_UP", "speed_y": 0, "speed_z": 1, "arrow": "\u2191", "cmd": "UP", "desc": "Thumb extended, all fingers folded"},
    {"gesture": "THUMBS_UP", "speed_y": 0, "speed_z": 1, "arrow": "\u2191", "cmd": "UP", "desc": "Thumb extended, all fingers folded"},
    {"gesture": "THUMBS_UP", "speed_y": 0, "speed_z": 1, "arrow": "\u2191", "cmd": "UP", "desc": "Thumb extended, all fingers folded"},
    {"gesture": "THUMBS_UP", "speed_y": 0, "speed_z": 1, "arrow": "\u2191", "cmd": "UP", "desc": "Thumb extended, all fingers folded"},
    {"gesture": "THUMBS_UP", "speed_y": 0, "speed_z": 1, "arrow": "\u2191", "cmd": "UP", "desc": "Thumb extended, all fingers folded"},
    {"gesture": "THUMBS_UP", "speed_y": 0, "speed_z": 1, "arrow": "\u2191", "cmd": "UP", "desc": "Thumb extended, all fingers folded"},
    {"gesture": "THUMBS_UP", "speed_y": 0, "speed_z": 1, "arrow": "\u2191", "cmd": "UP", "desc": "Thumb extended, all fingers folded"},

    {"gesture": "THUMBS_DOWN", "speed_y": 0, "speed_z": -1, "arrow": "\u2193", "cmd": "DOWN", "desc": "Thumb folded, thumb tip below IP joint, all fingers extended"},
    {"gesture": "THUMBS_DOWN", "speed_y": 0, "speed_z": -1, "arrow": "\u2193", "cmd": "DOWN", "desc": "Thumb folded, thumb tip below IP joint, all fingers extended"},
    {"gesture": "THUMBS_DOWN", "speed_y": 0, "speed_z": -1, "arrow": "\u2193", "cmd": "DOWN", "desc": "Thumb folded, thumb tip below IP joint, all fingers extended"},
    {"gesture": "THUMBS_DOWN", "speed_y": 0, "speed_z": -1, "arrow": "\u2193", "cmd": "DOWN", "desc": "Thumb folded, thumb tip below IP joint, all fingers extended"},
    {"gesture": "THUMBS_DOWN", "speed_y": 0, "speed_z": -1, "arrow": "\u2193", "cmd": "DOWN", "desc": "Thumb folded, thumb tip below IP joint, all fingers extended"},
    {"gesture": "THUMBS_DOWN", "speed_y": 0, "speed_z": -1, "arrow": "\u2193", "cmd": "DOWN", "desc": "Thumb folded, thumb tip below IP joint, all fingers extended"},
    {"gesture": "THUMBS_DOWN", "speed_y": 0, "speed_z": -1, "arrow": "\u2193", "cmd": "DOWN", "desc": "Thumb folded, thumb tip below IP joint, all fingers extended"},
    {"gesture": "THUMBS_DOWN", "speed_y": 0, "speed_z": -1, "arrow": "\u2193", "cmd": "DOWN", "desc": "Thumb folded, thumb tip below IP joint, all fingers extended"},
    {"gesture": "THUMBS_DOWN", "speed_y": 0, "speed_z": -1, "arrow": "\u2193", "cmd": "DOWN", "desc": "Thumb folded, thumb tip below IP joint, all fingers extended"},
    {"gesture": "THUMBS_DOWN", "speed_y": 0, "speed_z": -1, "arrow": "\u2193", "cmd": "DOWN", "desc": "Thumb folded, thumb tip below IP joint, all fingers extended"},

    {"gesture": "OPEN_PALM", "speed_y": 0, "speed_z": 0, "arrow": "\u2299", "cmd": "HOVER", "desc": "All fingers extended, palm facing camera"},
    {"gesture": "OPEN_PALM", "speed_y": 0, "speed_z": 0, "arrow": "\u2299", "cmd": "HOVER", "desc": "All fingers extended, palm facing camera"},
    {"gesture": "OPEN_PALM", "speed_y": 0, "speed_z": 0, "arrow": "\u2299", "cmd": "HOVER", "desc": "All fingers extended, palm facing camera"},
    {"gesture": "OPEN_PALM", "speed_y": 0, "speed_z": 0, "arrow": "\u2299", "cmd": "HOVER", "desc": "All fingers extended, palm facing camera"},
    {"gesture": "OPEN_PALM", "speed_y": 0, "speed_z": 0, "arrow": "\u2299", "cmd": "HOVER", "desc": "All fingers extended, palm facing camera"},
    {"gesture": "OPEN_PALM", "speed_y": 0, "speed_z": 0, "arrow": "\u2299", "cmd": "HOVER", "desc": "All fingers extended, palm facing camera"},
    {"gesture": "OPEN_PALM", "speed_y": 0, "speed_z": 0, "arrow": "\u2299", "cmd": "HOVER", "desc": "All fingers extended, palm facing camera"},
    {"gesture": "OPEN_PALM", "speed_y": 0, "speed_z": 0, "arrow": "\u2299", "cmd": "HOVER", "desc": "All fingers extended, palm facing camera"},
    {"gesture": "OPEN_PALM", "speed_y": 0, "speed_z": 0, "arrow": "\u2299", "cmd": "HOVER", "desc": "All fingers extended, palm facing camera"},
    {"gesture": "OPEN_PALM", "speed_y": 0, "speed_z": 0, "arrow": "\u2299", "cmd": "HOVER", "desc": "All fingers extended, palm facing camera"},

    {"gesture": "CLOSED_FIST", "speed_y": 0, "speed_z": 0, "arrow": "\u2299", "cmd": "HOVER", "desc": "All fingers folded, fist shape"},
    {"gesture": "CLOSED_FIST", "speed_y": 0, "speed_z": 0, "arrow": "\u2299", "cmd": "HOVER", "desc": "All fingers folded, fist shape"},
    {"gesture": "CLOSED_FIST", "speed_y": 0, "speed_z": 0, "arrow": "\u2299", "cmd": "HOVER", "desc": "All fingers folded, fist shape"},
    {"gesture": "CLOSED_FIST", "speed_y": 0, "speed_z": 0, "arrow": "\u2299", "cmd": "HOVER", "desc": "All fingers folded, fist shape"},
    {"gesture": "CLOSED_FIST", "speed_y": 0, "speed_z": 0, "arrow": "\u2299", "cmd": "HOVER", "desc": "All fingers folded, fist shape"},
    {"gesture": "CLOSED_FIST", "speed_y": 0, "speed_z": 0, "arrow": "\u2299", "cmd": "HOVER", "desc": "All fingers folded, fist shape"},
    {"gesture": "CLOSED_FIST", "speed_y": 0, "speed_z": 0, "arrow": "\u2299", "cmd": "HOVER", "desc": "All fingers folded, fist shape"},
    {"gesture": "CLOSED_FIST", "speed_y": 0, "speed_z": 0, "arrow": "\u2299", "cmd": "HOVER", "desc": "All fingers folded, fist shape"},
    {"gesture": "CLOSED_FIST", "speed_y": 0, "speed_z": 0, "arrow": "\u2299", "cmd": "HOVER", "desc": "All fingers folded, fist shape"},
    {"gesture": "CLOSED_FIST", "speed_y": 0, "speed_z": 0, "arrow": "\u2299", "cmd": "HOVER", "desc": "All fingers folded, fist shape"},

    {"gesture": "PALM_LEFT", "speed_y": -1, "speed_z": 0, "arrow": "\u2190", "cmd": "LEFT", "desc": "Hand on left of image, thumb on left edge"},
    {"gesture": "PALM_LEFT", "speed_y": -1, "speed_z": 0, "arrow": "\u2190", "cmd": "LEFT", "desc": "Hand on left of image, thumb on left edge"},
    {"gesture": "PALM_LEFT", "speed_y": -1, "speed_z": 0, "arrow": "\u2190", "cmd": "LEFT", "desc": "Hand on left of image, thumb on left edge"},
    {"gesture": "PALM_LEFT", "speed_y": -1, "speed_z": 0, "arrow": "\u2190", "cmd": "LEFT", "desc": "Hand on left of image, thumb on left edge"},
    {"gesture": "PALM_LEFT", "speed_y": -1, "speed_z": 0, "arrow": "\u2190", "cmd": "LEFT", "desc": "Hand on left of image, thumb on left edge"},
    {"gesture": "PALM_LEFT", "speed_y": -1, "speed_z": 0, "arrow": "\u2190", "cmd": "LEFT", "desc": "Hand on left of image, thumb on left edge"},
    {"gesture": "PALM_LEFT", "speed_y": -1, "speed_z": 0, "arrow": "\u2190", "cmd": "LEFT", "desc": "Hand on left of image, thumb on left edge"},
    {"gesture": "PALM_LEFT", "speed_y": -1, "speed_z": 0, "arrow": "\u2190", "cmd": "LEFT", "desc": "Hand on left of image, thumb on left edge"},
    {"gesture": "PALM_LEFT", "speed_y": -1, "speed_z": 0, "arrow": "\u2190", "cmd": "LEFT", "desc": "Hand on left of image, thumb on left edge"},
    {"gesture": "PALM_LEFT", "speed_y": -1, "speed_z": 0, "arrow": "\u2190", "cmd": "LEFT", "desc": "Hand on left of image, thumb on left edge"},

    {"gesture": "PALM_RIGHT", "speed_y": 1, "speed_z": 0, "arrow": "\u2192", "cmd": "RIGHT", "desc": "Hand on right of image, thumb on right edge"},
    {"gesture": "PALM_RIGHT", "speed_y": 1, "speed_z": 0, "arrow": "\u2192", "cmd": "RIGHT", "desc": "Hand on right of image, thumb on right edge"},
    {"gesture": "PALM_RIGHT", "speed_y": 1, "speed_z": 0, "arrow": "\u2192", "cmd": "RIGHT", "desc": "Hand on right of image, thumb on right edge"},
    {"gesture": "PALM_RIGHT", "speed_y": 1, "speed_z": 0, "arrow": "\u2192", "cmd": "RIGHT", "desc": "Hand on right of image, thumb on right edge"},
    {"gesture": "PALM_RIGHT", "speed_y": 1, "speed_z": 0, "arrow": "\u2192", "cmd": "RIGHT", "desc": "Hand on right of image, thumb on right edge"},
    {"gesture": "PALM_RIGHT", "speed_y": 1, "speed_z": 0, "arrow": "\u2192", "cmd": "RIGHT", "desc": "Hand on right of image, thumb on right edge"},
    {"gesture": "PALM_RIGHT", "speed_y": 1, "speed_z": 0, "arrow": "\u2192", "cmd": "RIGHT", "desc": "Hand on right of image, thumb on right edge"},
    {"gesture": "PALM_RIGHT", "speed_y": 1, "speed_z": 0, "arrow": "\u2192", "cmd": "RIGHT", "desc": "Hand on right of image, thumb on right edge"},
    {"gesture": "PALM_RIGHT", "speed_y": 1, "speed_z": 0, "arrow": "\u2192", "cmd": "RIGHT", "desc": "Hand on right of image, thumb on right edge"},
    {"gesture": "PALM_RIGHT", "speed_y": 1, "speed_z": 0, "arrow": "\u2192", "cmd": "RIGHT", "desc": "Hand on right of image, thumb on right edge"},

    {"gesture": "PALM_UP", "speed_y": 0, "speed_z": 0, "arrow": "\u2299", "cmd": "HOVER", "desc": "Palm normal points up (Z > 0.3)"},
    {"gesture": "PALM_UP", "speed_y": 0, "speed_z": 0, "arrow": "\u2299", "cmd": "HOVER", "desc": "Palm normal points up (Z > 0.3)"},
    {"gesture": "PALM_UP", "speed_y": 0, "speed_z": 0, "arrow": "\u2299", "cmd": "HOVER", "desc": "Palm normal points up (Z > 0.3)"},
    {"gesture": "PALM_UP", "speed_y": 0, "speed_z": 0, "arrow": "\u2299", "cmd": "HOVER", "desc": "Palm normal points up (Z > 0.3)"},
    {"gesture": "PALM_UP", "speed_y": 0, "speed_z": 0, "arrow": "\u2299", "cmd": "HOVER", "desc": "Palm normal points up (Z > 0.3)"},
    {"gesture": "PALM_UP", "speed_y": 0, "speed_z": 0, "arrow": "\u2299", "cmd": "HOVER", "desc": "Palm normal points up (Z > 0.3)"},
    {"gesture": "PALM_UP", "speed_y": 0, "speed_z": 0, "arrow": "\u2299", "cmd": "HOVER", "desc": "Palm normal points up (Z > 0.3)"},
    {"gesture": "PALM_UP", "speed_y": 0, "speed_z": 0, "arrow": "\u2299", "cmd": "HOVER", "desc": "Palm normal points up (Z > 0.3)"},
    {"gesture": "PALM_UP", "speed_y": 0, "speed_z": 0, "arrow": "\u2299", "cmd": "HOVER", "desc": "Palm normal points up (Z > 0.3)"},
    {"gesture": "PALM_UP", "speed_y": 0, "speed_z": 0, "arrow": "\u2299", "cmd": "HOVER", "desc": "Palm normal points up (Z > 0.3)"},

    {"gesture": "PALM_DOWN", "speed_y": 0, "speed_z": 0, "arrow": "\u2299", "cmd": "HOVER", "desc": "Palm normal points down (Z < -0.3)"},
    {"gesture": "PALM_DOWN", "speed_y": 0, "speed_z": 0, "arrow": "\u2299", "cmd": "HOVER", "desc": "Palm normal points down (Z < -0.3)"},
    {"gesture": "PALM_DOWN", "speed_y": 0, "speed_z": 0, "arrow": "\u2299", "cmd": "HOVER", "desc": "Palm normal points down (Z < -0.3)"},
    {"gesture": "PALM_DOWN", "speed_y": 0, "speed_z": 0, "arrow": "\u2299", "cmd": "HOVER", "desc": "Palm normal points down (Z < -0.3)"},
    {"gesture": "PALM_DOWN", "speed_y": 0, "speed_z": 0, "arrow": "\u2299", "cmd": "HOVER", "desc": "Palm normal points down (Z < -0.3)"},
    {"gesture": "PALM_DOWN", "speed_y": 0, "speed_z": 0, "arrow": "\u2299", "cmd": "HOVER", "desc": "Palm normal points down (Z < -0.3)"},
    {"gesture": "PALM_DOWN", "speed_y": 0, "speed_z": 0, "arrow": "\u2299", "cmd": "HOVER", "desc": "Palm normal points down (Z < -0.3)"},
    {"gesture": "PALM_DOWN", "speed_y": 0, "speed_z": 0, "arrow": "\u2299", "cmd": "HOVER", "desc": "Palm normal points down (Z < -0.3)"},
    {"gesture": "PALM_DOWN", "speed_y": 0, "speed_z": 0, "arrow": "\u2299", "cmd": "HOVER", "desc": "Palm normal points down (Z < -0.3)"},
    {"gesture": "PALM_DOWN", "speed_y": 0, "speed_z": 0, "arrow": "\u2299", "cmd": "HOVER", "desc": "Palm normal points down (Z < -0.3)"},

    {"gesture": "UNKNOWN", "speed_y": 0, "speed_z": 0, "arrow": "\u25cb", "cmd": "NO TRACK", "desc": "No hand detected / unknown gesture"},
    {"gesture": "UNKNOWN", "speed_y": 0, "speed_z": 0, "arrow": "\u25cb", "cmd": "NO TRACK", "desc": "No hand detected / unknown gesture"},
    {"gesture": "UNKNOWN", "speed_y": 0, "speed_z": 0, "arrow": "\u25cb", "cmd": "NO TRACK", "desc": "No hand detected / unknown gesture"},
    {"gesture": "UNKNOWN", "speed_y": 0, "speed_z": 0, "arrow": "\u25cb", "cmd": "NO TRACK", "desc": "No hand detected / unknown gesture"},
    {"gesture": "UNKNOWN", "speed_y": 0, "speed_z": 0, "arrow": "\u25cb", "cmd": "NO TRACK", "desc": "No hand detected / unknown gesture"},
    {"gesture": "UNKNOWN", "speed_y": 0, "speed_z": 0, "arrow": "\u25cb", "cmd": "NO TRACK", "desc": "No hand detected / unknown gesture"},
    {"gesture": "UNKNOWN", "speed_y": 0, "speed_z": 0, "arrow": "\u25cb", "cmd": "NO TRACK", "desc": "No hand detected / unknown gesture"},
    {"gesture": "UNKNOWN", "speed_y": 0, "speed_z": 0, "arrow": "\u25cb", "cmd": "NO TRACK", "desc": "No hand detected / unknown gesture"},
    {"gesture": "UNKNOWN", "speed_y": 0, "speed_z": 0, "arrow": "\u25cb", "cmd": "NO TRACK", "desc": "No hand detected / unknown gesture"},
    {"gesture": "UNKNOWN", "speed_y": 0, "speed_z": 0, "arrow": "\u25cb", "cmd": "NO TRACK", "desc": "No hand detected / unknown gesture"},
]


def get_gesture_set() -> list[str]:
    """Return the set of recognized gesture names."""
    return list(GESTURE_NAMES.values())


def get_rc_speed_map() -> dict[str, str]:
    """Return gesture -> RC speed mapping."""
    return RC_SPEEDS.copy()


def get_overlay_map() -> dict[str, tuple[str, str]]:
    """Return gesture -> (arrow, cmd) mapping."""
    return {g: (OVERLAY_ARROW[name], OVERLAY_CMD[name]) for g in OVERLAY_ARROW}


def get_simulated_frames() -> list[dict]:
    """Return the simulated frames dataset."""
    return FRAMES.copy()


def print_rc_table():
    print()
    print("=" * 65)
    print("TABLE 1: Gesture → RC (speed_y, speed_z)")
    print("=" * 65)
    print(f"{'Gesture':<15} {'Speed (y, z)':<20} {'Drone Action'}")
    print("-" * 65)
    ACTION_NAMES = {
        "(0, 0)": "Hover",
        "(0, 1)": "Move Up",
        "(0, -1)": "Move Down",
        "(-1, 0)": "Move Left",
        "(1, 0)": "Move Right",
    }
    for g in GESTURE_NAMES:
        name = GESTURE_NAMES[g]
        speed = RC_SPEEDS[name]
        action = ACTION_NAMES.get(speed, "Hover")
        print(f"{name:<15} {speed:<20} {action}")


def print_overlay_table():
    print()
    print("=" * 85)
    print("TABLE 2: Gesture → Overlay Labels (arrow, command text)")
    print("=" * 85)
    print(f"{'Gesture':<15} {'Arrow':<8} {'Command Text':<15} {'Video Frame Label'}")
    print("-" * 85)
    for g in GESTURE_NAMES:
        name = GESTURE_NAMES[g]
        arrow = OVERLAY_ARROW[name]
        cmd = OVERLAY_CMD[name]
        label = f"{arrow} {cmd}"
        print(f"{name:<15} {arrow:<8} {cmd:<15} {label}")


def print_frames_table():
    print()
    print("=" * 130)
    print("TABLE 3: Simulated Frames Dataset (10 frames per gesture)")
    print("=" * 130)
    print(f"{'Frame':>5} {'Gesture':<15} {'speed_y':<8} {'speed_z':<8} {'Arrow':<8} {"Command":<15} {'Description'}")
    print("-" * 130)
    for i, f in enumerate(FRAMES, 1):
        print(f"{i:>5} {f['gesture']:<15} {f['speed_y']:<8} {f['speed_z']:<8} {f['arrow']:<8} {f['cmd']:<15} {f['desc']}")


if __name__ == "__main__":
    if "--list" in sys.argv:
        print("Gesture set:")
        for name in get_gesture_set():
            print(f"  - {name}")
        sys.exit(0)

    print_rc_table()
    print_overlay_table()
    print_frames_table()
    print()
    print("Total frames:", len(FRAMES))
    print("Unique gestures:", len(set(f["gesture"] for f in FRAMES)))

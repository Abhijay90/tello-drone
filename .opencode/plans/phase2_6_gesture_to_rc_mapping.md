# Phase 2.6: Gesture to RC Command Mapping

## Goal
Map 8 hand gesture classifications to Tello RC drive controls for drone flight.

## Architecture

### Gesture to Command Table (confirmed)

| Gesture | RC Command (x, y, z, yaw) | Meaning |
|---|---|---|
| OPEN_PALM | (0, -30, 0, 0) | move back |
| CLOSED_FIST | (0, +30, 0, 0) | move forward |
| THUMBS_UP | (0, 0, +20, 0) | move up |
| THUMBS_DOWN | (0, 0, -20, 0) | move down |
| PALM_LEFT | (-30, 0, 0, 0) | move left |
| PALM_RIGHT | (+30, 0, 0, 0) | move right |
| PALM_UP | (0, 0, 0, 0) | hover |
| PALM_DOWN | (0, 0, 0, 0) | hover |
| UNKNOWN | (0, 0, 0, 0) | hover |

### RC Direction (from `djitellopy.Tello.send_rc_control`):
- x: left/right (-30 to +30)
- y: forward/backward (-30 to +30)
- z: up/down (-20 to +20)
- yaw: rotate (0, no rotation)

## Design Confirmations

* **Speed**: Confirmed ±30 for x/y, ±20 for z. Can fine-tune later.
* **PALM_UP and PALM_DOWN**: hover (no movement)
* **All unrecognized gestures**: hover (0, 0, 0, 0)
* **Debounce**: 3 frames before applying gesture command (same as existing `DEBOUNCE_FRAMES`)
* **Command change policy**: Only send `send_rc_control` when the mapped command differs from the last sent command
* **Separate file**: Mapping will be in its own module, imported by `tello_handtrack.py`

## Files Required

### `tello-drone/gesture_rc_mapping.py`
* Gesture string → RC tuple dictionary
* Function: `gesture_to_rc(gesture_name) -> tuple[int, int, int, int]`

### Modification to `tello_handtrack.py`
* Import gesture_rc_mapping
* Add debounce logic for gesture stabilization (reuse existing `DEBOUNCE_FRAMES=3`)
* In main loop: classify gesture → debounce → map to RC → if changed → send
* Track `last_rc_sent` to avoid redundant sends

## Code Flow (per frame)

```
captur frame → detect hand landmarks → classify gesture → debounce stable
→ map to RC tuple → if tuple != last_rc_sent → me.send_rc_control(x,y,z,yaw)
```

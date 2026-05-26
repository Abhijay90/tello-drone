# Hand Gesture Drone Control - Discussion

**Phase Status:**
- Phase 1 (Detection & Verification): ✅ Implemented
- Phase 2 (Flight Control): 🟡 Planned

## Goal
Implement palm detection-based drone control for a Tello drone. Use palm position in the camera frame to continuously control drone movement (forward, backward, up, down, distance maintenance) via real-time PD feedback control.

## Architecture Decisions

### Detection: MediaPipe Hands
- Chosen over Haar Cascade + Skin Color for superior landmark accuracy, wrist-as-center reliability, and robustness across skin tones.
- Output: palm center `(cx, cy)`, bounding box, and `area` (bounding box area as distance proxy).

### Control Strategy: Relative Position & Area PID
Palm position maps to drone RC commands via proportional-derivative (PD) controllers.

| Signal | Source | Drives | Movement |
|---|---|---|---|
| `cx` | Palm x-coordinate | Y-axis (forward/back) | Move Y depending on center offset |
| `area` | Bounding box area | Z-axis (altitude) | Maintain target distance |
| Hover/Dead Zone | `abs(error) < 20` | Holds hover | Prevents jitter near target |

No hand detected → Hover.

### PID Strategy: Separate Gains per Axis
- `pid_y` controls Y-axis (forward/back). Horizontal settling is faster with less drift.
- `pid_z` controls Z-axis (up/down). Vertical movement fights gravity and requires different tuning.
- Starting gains for both: `[0.4, 0.4, 0]`. Will be tunable during flight.
- Dead zone: ±20px around center or `target_area` to prevent oscillation when close to goal.

## Open Questions & Tuning Targets
1. Optimal `dead_zone` value for 360×240 frame stability.
2. `target_area` value for comfortable 1-meter drone-to-hand distance.
3. `pid_z` gains vs `pid_y` gains (expect `pid_z` to need higher Kp to lift drone weight).
4. Handling temporary loss of tracking (should we hold course or hover?).

# Phase 3: Control Stability & Smoothing

**Goal:** Improve drone tracking smoothness and steady-state accuracy — no detection changes, only control-layer fixes.

## Context

Phase 2 gave us a functional detection → RC control loop with separate PID axes for forward/back and altitude. Three control-quality issues remain:

| Issue | Symptom | Fix |
|---|---|---|
| Integral term not integrated | Drone holds steady *offset* — never settles exactly on center | Proper integral accumulation with anti-windup clamp |
| Raw detection jitter | Drone chatters back-and-forth at rest | EMA smoothing on `cx, cy, area` before PID |
| Hardcoded `dt=1` in PID | Derivative damping is ~30x wrong at 30 FPS — overreacts to fast motion, under-reacts to slow | Real time delta for derivative term |

## Change Set

Only `tello_handtrack.py` — zero changes to `vision/palmtracker.py` or `requirement.txt`.

### 1. PID Integral Fix

**Current** (`tello_handtrack.py:108`):
```python
def compute_pid(error, prev_error, dt, pid_gains):
    p = pid_gains[0] * error
    d = pid_gains[1] * (error - prev_error)
    i = pid_gains[2] * error  # Ki * error — NOT integral, just another proportional gain!
    return p + d + i
```

Ki is applied once to current error, not accumulated over time. Mathematically identical to `(Kp + Ki) * error`. The drone will never reach zero error because the integral term contributes nothing.

**Fix:**
```python
def compute_pid(error, prev_error, integral, dt, pid_gains):
    p = pid_gains[0] * error
    d = pid_gains[1] * (error - prev_error) / dt if dt > 0 else 0
    integral_new = integral + pid_gains[2] * error * dt
    # Anti-windup clamp
    if abs(integral_new) > 100:
        integral_new = 100 if integral_new > 0 else -100
    return p + d + integral_new, integral_new
```

New loop state: `integral_y = 0, integral_z = 0` (replaces the previous `prev_error` variable semantics).

Call site:
```python
speed_y, integral_y = compute_pid(error_y, prev_error_y, integral_y, dt, FB_PID)
speed_z, integral_z = compute_pid(error_z, prev_error_z, integral_z, dt, VD_PID)
```

### 2. EMA Smoothing

Lightweight filter applied to raw detection output before PID:

```python
class EMAFilter:
    def __init__(self, alpha=0.3, initial=0.0):
        self.alpha = alpha
        self.value = initial
    def update(self, new_val):
        self.value = self.alpha * new_val + (1 - self.alpha) * self.value
        return self.value
```

Applied on 3 values after `findPalm()`:
```python
cx = ema_cx.update(cx_raw) or ema_cx.value
cy = ema_cy.update(cy_raw) or ema_cy.value
area = ema_area.update(area_raw) or ema_area.value
```

Uses `or self.value` fallback for first frame before any update has occurred. Typical values: `EMA_ALPHA = 0.3` balances responsiveness vs. smoothness. Lower (0.15) = smoother but more lag. Higher (0.5) = more responsive but less smoothing.

### 3. Real Time Delta

**Current** (`tello_handtrack.py:209, 217`):
```python
speed_y = compute_pid(error_y, prev_error_y, 1, FB_PID)  # dt=1 hardcoded
```

At 30 FPS, `dt=1` means the derivative term is computed as if 1 second elapsed between frames — that's 30x the intended gain. The controller over-reacts to fast movements and under-reacts to slow ones.

**Fix:**
```python
prev_time = time()

# inside loop:
current_time = time()
dt = current_time - prev_time
prev_time = current_time

speed_y, integral_y = compute_pid(error_y, prev_error_y, integral_y, max(dt, 0.001), FB_PID)
```

## Expected Outcome

| Metric | Before | After |
|---|---|---|
| Steady-state offset | ~5-10px residual, never settles perfectly | <1px (integral eliminates it) |
| Jitter at rest | Oscillating ±10px visible chatter | Smooth ±2px or less |
| Fast sweep response | Over-reacts (derivative 30x too high) | Proportional to actual hand velocity |
| Hand-when-returned-to-center | Drone held slightly left/right | Drone returns exactly to hover |

## Tuning Notes

- Start with `EMA_ALPHA = 0.3`, `integral_limit = 100`, `FB_PID = [0.4, 0.4, 0.1]`, `VD_PID = [0.4, 0.2, 0.15]`
- If drone still oscillates after adding integral: reduce Ki to 0.05 and increase D-term damping
- If drone is sluggish: increase EMA_ALPHA to 0.4-0.5 (less smoothing)
- Anti-windup clamp may need adjustment if large persistent errors cause slow recovery

## Implementation Checklist

- [ ] Replace `compute_pid()` with integral-accumulating version
- [ ] Add `EMAFilter` class + 3 instances (`cx`, `cy`, `area`)
- [ ] Add `prev_time` tracking + real `dt` computation
- [ ] Update PID call sites to pass `integral` and `dt`
- [ ] Test: verify drone settles exactly on center (no residual offset)
- [ ] Test: verify smooth hover when hand is still
- [ ] Tune EMA_ALPHA if needed (lag vs. jitter trade-off)

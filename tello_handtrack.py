# palm gesture flight control using pid hand detection
from djitellopy import tello
from time import sleep, time
from vision.palmtracker import findPalm
import cv2
import sys

# Configuration
RES_W, RES_H = 960, 720
TARGET_AREA = 4000
DEAD_ZONE = 20
TARGET_X = RES_W // 2

FB_PID = [0.4, 0.4, 0]   # Forward/back control (Y axis)
VD_PID = [0.4, 0.0, 0.2]  # Up/down control (Z axis)

# -- Drone mode switch --
# True  = real Tello drone (camera test, takeoff, RC commands)
# False = webcam debug mode (no drone commands, no takeoff)
DRONE_MODE = True

# Camera source index (only used when DRONE_MODE is False)
WEBCAM_IDX = 0


def test_drone_camera(me):
    """Test FPV stream with per-frame latency overlay feedback."""
    me.streamon()
    frames_received = 0
    total_latency = 0
    results = []
    last_frame = None
    print("we are here")

    for i in range(1, 4):
        start = time()
        frame = me.get_frame_read().frame
        elapsed = int((time() - start) * 1000)

        if frame is None:
            print(f"  Frame {i}: frame is None")
            results.append(elapsed)
            break

        if frame.size == 0:
            print(f"  Frame {i}: frame.size == 0")
            results.append(elapsed)
            break

        frames_received += 1
        total_latency += elapsed
        results.append(elapsed)
        print(f"  Frame {i}: latency={elapsed}ms")
        # cv2.imshow("Camera Test", frame)

        # Overlay per-frame feedback
        fb = frame.copy()
        cv2.putText(fb, f"Frame {i}: OK", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(fb, f"Latency: {elapsed} ms", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.imshow("Camera Test", fb)
        key = cv2.waitKey(300) & 0xFF
        if key == ord('q'):
            cv2.destroyAllWindows()
            me.streamoff()
            return False, "Aborted by user"

        last_frame = frame

    if frames_received == 3:
        avg_latency = total_latency // 3
        fps = 1000 / avg_latency if avg_latency > 0 else 0
        latency_str = ",".join(f"{r}ms" for r in results)
        msg = f"Stream OK: 3/3 frames, avg={avg_latency}ms, ~{int(fps)} FPS, [{latency_str}]"
        print(msg)

        summary = last_frame.copy()
        cv2.putText(summary, "Stream OK!", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(summary, f"avg={avg_latency}ms", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(summary, f"~{int(fps)} FPS", (10, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(summary, "Press any key for takeoff...", (10, 120),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.imshow("Camera Test", summary)
        cv2.waitKey(0)
        return True, msg
    else:
        fail_msg = f"Stream FAILED: only {frames_received}/3 frames received.\nCheck Wi-Fi, Tello power, and that you are on the same network."
        print(f"  {fail_msg}")
        summary = last_frame.copy() if last_frame else None
        if summary is not None:
            cv2.putText(summary, "Stream FAILED!", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            cv2.putText(summary, f"Only {frames_received}/3 frames", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            cv2.imshow("Camera Test", summary)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        return False, fail_msg


def compute_pid(error, prev_error, integral, dt, pid_gains):
    p = pid_gains[0] * error
    d = pid_gains[1] * (error - prev_error) / dt if dt > 0 else 0
    integral_new = integral + pid_gains[2] * error * dt
    if abs(integral_new) > 100:
        integral_new = 100 if integral_new > 0 else -100
    return p + d + integral_new, integral_new


class EMAFilter:
    EMA_ALPHA = 0.3

    def __init__(self, initial=0.0):
        self.value = initial
        self.first = True

    def update(self, new_val):
        if self.first:
            self.value = new_val
            self.first = False
            return new_val
        self.value = self.EMA_ALPHA * new_val + (1 - self.EMA_ALPHA) * self.value
        return self.value


def init_device():
    """Initialize either Tello drone or webcam based on DRONE_MODE. Returns (me_or_none, cap_or_none)."""
    me = None
    cap = None

    if DRONE_MODE:
        me = tello.Tello()
        me.connect()
        print(f"Battery: {me.get_battery()}%")

        # Step 1: Measure raw FPV frame resolution (one-time)
        me.streamon()
        frame_peek = me.get_frame_read().frame
        if frame_peek is not None:
            print(f"[FPV DEBUG] Raw frame shape (H×W×C): {frame_peek.shape}")
            print(f"[FPV DEBUG] Raw resolution: {frame_peek.shape[1]}×{frame_peek.shape[0]}")
            print(f"[FPV DEBUG] Detection currently upscales to: {RES_W}×{RES_H}")
            if frame_peek.shape[0] < RES_H or frame_peek.shape[1] < RES_W:
                print(f"[FPV DEBUG] WARNING: Upscaling {frame_peek.shape[1]}×{frame_peek.shape[0]} → {RES_W}×{RES_H} adds no real pixels")
        else:
            print("[FPV DEBUG] Frame is None — check stream")
        me.streamoff()

        # ok, info = test_drone_camera(me)
        # if not ok:
        #     print(f"Drone test failed: {info}")
        #     sys.exit(1)

        # me.streamoff()

        # sleep(2)
        print('closing streaming for starting new one')

        me.streamon()

        print("Taking off...")
        me.takeoff()
        print("Drone ready — start gesturing!")

    else:
        print("[webcam debug mode — no drone commands]")
        cap = cv2.VideoCapture(WEBCAM_IDX)
        if not cap.isOpened():
            print(f"Cannot open webcam index {WEBCAM_IDX}")
            sys.exit(1)

        ok_count = 0
        for i in range(1, 4):
            ret, frame = cap.read()
            if ret and frame is not None and frame.size > 0:
                ok_count += 1
                print(f"  Frame {i}: OK")
            else:
                print(f"  Frame {i}: failed")
                break

        if ok_count != 3:
            print("Webcam test failed: not enough frames")
            cap.release()
            sys.exit(1)

        print("Webcam OK — starting flight loop...")

    return me, cap


def get_frame(me, cap):
    """Get next frame from drone stream or webcam."""
    if DRONE_MODE and me:
        print('this is for drone frame')
        frame = me.get_frame_read().frame
        print(f"  Raw frame shape: {frame.shape}")  # Debug: confirm Tello stream resolution
        return frame
    elif cap is not None:
        ret, frame = cap.read()
        if ret and frame is not None:
            print(f"  Raw frame shape: {frame.shape}")  # Debug: confirm webcam resolution
            return frame
    return None


def send_rc(me, y, z):
    """Send RC only if in drone mode and armed."""
    if DRONE_MODE and me:
        print(f" y:{y} z: {z}",)
        me.send_rc_control(0, int(y), int(z), 0)


# -- Startup --
me, cap = init_device()

prev_error_y = 0
prev_error_z = 0
integral_y = 0
integral_z = 0
prev_time = time()

try:
    while True:
        img = get_frame(me, cap)
        if img is None:
            print("no image recieved")
            continue
        img = cv2.resize(img, (RES_W, RES_H))

        img, info = findPalm(img)
        cx, cy, area = info

        if cx == 0 or area == 0:
            speed_y, speed_z = 0, 0
        else:
            # Y-Axis: Forward/Back based on horizontal palm position
            error_y = cx - TARGET_X
            if abs(error_y) > DEAD_ZONE:
                current_time = time()
                dt = max(current_time - prev_time, 0.001)
                prev_time = current_time
                speed_y, integral_y = compute_pid(error_y, prev_error_y, integral_y, dt, FB_PID)
                speed_y = max(-50, min(50, speed_y))
            else:
                speed_y = 0

            # Z-Axis: Altitude based on palm area (distance to drone)
            error_z = TARGET_AREA - area
            if abs(error_z) > DEAD_ZONE:
                current_time = time()
                dt = max(current_time - prev_time, 0.001)
                prev_time = current_time
                speed_z, integral_z = compute_pid(error_z, prev_error_z, integral_z, dt, VD_PID)
                speed_z = max(-50, min(50, speed_z))
            else:
                speed_z = 0

            cv2.putText(img, f"cx:{cx} area:{area}", (10, 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.putText(img, f"speed_y:{speed_y} speed_z:{speed_z}", (10, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            mode_label = "DRONE" if DRONE_MODE else "WEBCAM"
            cv2.putText(img, f"[{mode_label} MODE]", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)

        send_rc(me, speed_y, speed_z)

        cv2.imshow("Palm Flight", img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("Landing...")
            cv2.destroyAllWindows()
            break

        prev_error_y = error_y if cx != 0 else 0
        prev_error_z = error_z if area != 0 else 0

except KeyboardInterrupt:
    print("Interrupted by user")
finally:
    if DRONE_MODE and me:
        me.streamoff()
        # me.land()
    elif cap is not None:
        cap.release()
    cv2.destroyAllWindows()

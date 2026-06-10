"""
data_collector.py - Webcam/FPV gesture dataset collector for Tello drone.

Usage:
  python test/data_collector.py              # Webcam mode
  python test/data_collector.py --fpv        # Tello FPV stream mode

Controls (both modes):
  1 - Open Palm (center)
  2 - Closed Fist (center)
  3 - Thumbs Up (center)
  4 - Thumbs Down (center)
  5 - Palm Up (center)
  6 - Palm Down (center)
  7 - Palm Left (left half of frame)
  8 - Palm Right (right half of frame)
  q - Quit and save
  s - Skip frame (abort current gesture capture)
  p - Print current gesture thresholds
  d - Toggle debug overlay

Each gesture is captured with a 3-second sliding window at 10 FPS.
Samples are saved as JSON (frame bytes + MediaPipe landmarks).

Output structure:
  data/
    open_palm/
      sample_0001.json
      sample_0002.json
      ...
    closed_fist/
    thumbs_up/
    ...
    dataset_summary.json
"""

import cv2
import json
import os
import sys
import time
import argparse
from pathlib import Path
from datetime import datetime
import mediapipe as mp
from mediapipe.tasks.python.vision import HandLandmarker
from mediapipe.tasks.python.vision.hand_landmarker import HandLandmarkerOptions
from mediapipe.tasks.python.core import base_options
from mediapipe.tasks.python.vision.core import vision_task_running_mode

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ── Gesture mappings ──────────────────────────────────────────────────────
GESTURE_KEYS = {
    49: 'open_palm',
    50: 'closed_fist',
    51: 'thumbs_up',
    52: 'thumbs_down',
    53: 'palm_up',
    54: 'palm_down',
    55: 'palm_left',
    56: 'palm_right',
}


# ── Base directories ──────────────────────────────────────────────────────
BASE_DIR = Path(__file__).parent.parent / "data"
SUMMARY_FILE = BASE_DIR / "dataset_summary.json"


# ── Data collector for webcam mode ────────────────────────────────────────
class WebcamDataCollector:
    """Captures gesture frames from a USB/webcam into the data/ directory."""

    SAMPLE_RATE = 10          # FPS for dataset
    CAPTURE_WINDOW =3.0       # seconds per gesture
    CAPTURE_INTERVAL = 1.0 / SAMPLE_RATE  # 0.1s = 100ms between frames

    def __init__(self):
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            raise RuntimeError("Could not open webcam. Check connection.")
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        self.current_gesture = None
        self.samples = []
        self._capture_start = 0.0
        self._last_sample_time = 0.0
        self.debug_mode = False

        # Create gesture directories
        for name in GESTURE_KEYS.values():
            (BASE_DIR / name).mkdir(exist_ok=True)

        # Initialize MediaPipe HandLandmarker
        script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        model_path = os.path.join(script_dir, "models", "hand_landmarker.task")
        
        self.mp_landmarker = HandLandmarker.create_from_options(
            HandLandmarkerOptions(
                base_options=base_options.BaseOptions(model_asset_path=model_path),
                running_mode=vision_task_running_mode.VisionTaskRunningMode.IMAGE,
                num_hands=2,
            )
        )

    def collect_sample(self, frame, landmarks):
        """Save one frame + landmarks for the active gesture."""
        if not self.current_gesture or landmarks is None:
            return

        if not hasattr(landmarks, '.landmark'):
            # Handle both mediapipe and raw list formats
            return

        h, w, _ = frame.shape
        landmarks_flat = [
            {'x': l.x, 'y': l.y, 'z': l.z,
             'visibility': l.visibility if hasattr(l, 'visibility') else None}
            for l in landmarks.landmark
        ]

        sample = {
            'frame': frame.tobytes().hex(),
            'width': w,
            'height': h,
            'landmarks': landmarks_flat,
            'gesture': self.current_gesture,
            'timestamp': datetime.now().isoformat(),
        }

        self.samples.append(sample)

        # Save to data/<gesture>/sample_NNNN.json
        idx = len(self.samples)
        sample_dir = BASE_DIR / self.current_gesture
        sample_path = sample_dir / f"sample_{idx:04d}.json"
        with open(sample_path, 'w') as f:
            json.dump(sample, f, indent=2)
        print(f"  Sample {idx:04d}: {self.current_gesture} -> {sample_path.name}")

    def _draw_bg_rect(self, frame, x, y, text, font_scale, thickness):
        """Draw a semi-transparent dark background behind text."""
        text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX,
                                     font_scale, thickness)[0]
        tx, ty = int(x + 4), int(y + 4)
        tw, th = int(text_size[0] + 8), int(text_size[1] + 6)
        cy = int(y)
        if hasattr(cv2, 'LINE_AA'):
            pass
        bg = frame.copy()
        cv2.rectangle(bg, (tx - 2, cy - th), (tx + tw, cy + 2), (0, 0, 0), -1)
        alpha = 0.5
        cv2.addWeighted(bg, alpha, frame, 1 - alpha, 0, frame)
        cv2.putText(frame, text, (x, y),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), thickness)

    def _draw_overlay(self, frame, gesture):
        """Draw capture status overlay on frame."""
        if gesture:
            elapsed = time.time() - self._capture_start
            remaining = max(0, self.CAPTURE_WINDOW - elapsed)

            # Progress bar
            progress = min(1.0, elapsed / self.CAPTURE_WINDOW)
            bar_w = 200
            bar_h = 15
            x, y = 15, 55
            cv2.rectangle(frame, (x, y), (x + bar_w, y + bar_h), (0, 0, 0), -1)
            cv2.rectangle(frame, (x, y),
                          (x + int(bar_w * progress), y + bar_h), (0, 255, 0), -1)
            cv2.rectangle(frame, (x, y), (x + bar_w, y + bar_h), (255, 255, 255), 1)

            self._draw_bg_rect(frame, 20, 90, f"Capturing: {gesture}", 0.6, 2)
            self._draw_bg_rect(frame, 20, 115, f"{remaining:.1f}s remaining", 0.5, 2)
        else:
            self._draw_bg_rect(frame, 15, 70, "Press 1-8 to start capture", 0.6, 2)
            self._draw_bg_rect(frame, 15, 95, "Press q to quit, s to skip, p for thresholds", 0.5, 1)

    def run(self):
        """Start the webcam data collection loop."""
        print("\n" + "=" * 60)
        print("  WEBCAM DATA COLLECTOR")
        print("=" * 60)
        print(f"  Output: {BASE_DIR.absolute()}")
        print("  Keys:")
        print("    1-8  = Start gesture capture")
        print("    q    = Quit & save dataset")
        print("    s    = Skip current gesture")
        print("    p    = Print thresholds")
        print("=" * 60)

        window = cv2.namedWindow("Data Collector", cv2.WINDOW_AUTOSIZE)

        while True:
            ret, frame = self.cap.read()
            if not ret:
                print("[WARN] Failed to read frame.")
                break

            frame = cv2.flip(frame, 1)  # Mirror for natural use
            self._draw_overlay(frame, self.current_gesture)

            # Check for auto-capture (every CAPTURE_INTERVAL seconds)
            now = time.time()
            if (self.current_gesture
                    and (now - self._capture_start) >= self.CAPTURE_WINDOW):
                print(f"\n[INFO] Gesture '{self.current_gesture}' capture complete!")
                self.current_gesture = None
                self._capture_start = 0.0
                self._last_sample_time = 0.0

            key = cv2.waitKey(1) & 0xFF
            if key != -1:
                print(f"  [KEY] {key} ('{chr(key) if 32 <= key < 127 else 'special'}')")
            if key == ord('q'):
                break
            elif key == ord('s'):
                if self.current_gesture:
                    print(f"  [SKIP] Aborted capture of {self.current_gesture}")
                    self.current_gesture = None
                    self._capture_start = 0.0
            elif key == ord('p'):
                self._print_thresholds()
            elif key == ord('d'):
                self.debug_mode = not self.debug_mode
                print(f"  [TOGGLE] Debug mode: {'ON' if self.debug_mode else 'OFF'}")
            elif key in GESTURE_KEYS:
                gesture = GESTURE_KEYS[key]
                if gesture != self.current_gesture:
                    self.current_gesture = gesture
                    self._capture_start = time.time()
                    self._last_sample_time = 0.0
                    print(f"\n  [CAPTURE START] {gesture} (3s window)")

            if self.current_gesture and (now - self._last_sample_time) >= self.CAPTURE_INTERVAL:
                # Run MediaPipe HandLandmarker detection
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
                detection_result = self.mp_landmarker.detect(mp_image)

                if detection_result.hand_landmarks and len(detection_result.hand_landmarks) > 0:
                    landmarks = detection_result.hand_landmarks[0]
                    # Convert to format compatible with collect_sample
                    class LandmarksWrapper:
                        def __init__(self, landmarks):
                            self.landmark = landmarks
                    self.collect_sample(frame, LandmarksWrapper(landmarks))
                    if self.debug_mode:
                        h, w, _ = frame.shape
                        for lm in landmarks:
                            x, y = int(lm[0] * w), int(lm[1] * h)
                            cv2.circle(frame, (x, y), 4, (0, 255, 0), -1)
                        cv2.putText(frame, "HANDS", (15, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                else:
                    if self.debug_mode:
                        cv2.putText(frame, "NO HAND", (15, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 100, 100), 1)
                self._last_sample_time = now

            cv2.imshow("Data Collector", frame)

        cv2.destroyWindow("Data Collector")
        self._save_summary()

    def _print_thresholds(self):
        """Print gesture classification thresholds (for V2 engine)."""
        from vision.gestures_v2 import (
            Gesture,
            THUMB_TIP_SEPARATION_BEYOND_IP,
            FINGER_TIP_SEPARATION_OPEN_RATIO,
            PALM_NORMAL_Z_THRESHOLD,
            CLOSED_FIST_SPREAD,
        )
        print("\n  --- V2 Gesture Thresholds ---")
        print(f"  THUMB_TIP_SEPARATION_BEYOND_IP: {THUMB_TIP_SEPARATION_BEYOND_IP:.2f} "
              "(thumb tip distance from IP)")
        print(f"  FINGER_TIP_SEPARATION_OPEN_RATIO: {FINGER_TIP_SEPARATION_OPEN_RATIO:.2f} "
              "(open/pinch threshold)")
        print(f"  PALM_NORMAL_Z_THRESHOLD: {PALM_NORMAL_Z_THRESHOLD:.2f} "
              "(palm orientation)")
        print(f"  CLOSED_FIST_SPREAD: {CLOSED_FIST_SPREAD:.0f} "
              "(max pixel spread)")
        for g in ['open_palm', 'closed_fist', 'thumbs_up',
                    'thumbs_down', 'palm_up', 'palm_down', 'palm_left', 'palm_right']:
            total = sum(1 for s in self.samples if s['gesture'] == g)
            print(f"  {g:12s}: {total:4d} samples")
        print("  ------- V2 Gesture Thresholds -------\n")

    def _save_summary(self):
        """Save dataset summary JSON."""
        counts = {g: sum(1 for s in self.samples if s['gesture'] == g)
                  for g in GESTURE_KEYS.values()}
        summary = {
            "device": "webcam",
            "total_samples": len(self.samples),
            "gesture_counts": counts,
            "data_dir": str(BASE_DIR.absolute()),
            "timestamp": datetime.now().isoformat(),
        }
        with open(SUMMARY_FILE, 'w') as f:
            json.dump(summary, f, indent=2)
        print("\n" + "=" * 60)
        print(f"  Dataset saved to: {SUMMARY_FILE}")
        for g, c in counts.items():
            print(f"    {g:12s}: {c}")
        print("=" * 60)


# ── Data collector for Tello FPV mode ─────────────────────────────────────
class FPVDataCollector:
    """Captures gesture frames from Tello's FPV stream into the data/ directory."""

    SAMPLE_RATE = 10
    CAPTURE_WINDOW = 3.0
    CAPTURE_INTERVAL = 1.0 / SAMPLE_RATE
    TELLO_IP = "192.168.10.1"
    TELLO_CMD_PORT = 8889
    TELLO_VID_PORT = 11111

    def __init__(self):
        self.samples = []
        self.current_gesture = None
        self._capture_start = 0.0
        self._last_sample_time = 0.0
        self.debug_mode = False
        self.mp_landmarker = None

        for name in GESTURE_KEYS.values():
            (BASE_DIR / name).mkdir(exist_ok=True)

    @staticmethod
    def _send_command(cmd, timeout=3):
        """Send UDP command to Tello and return response."""
        import socket
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.settimeout(timeout)
        try:
            sock.sendto(cmd.encode(), (FPVDataCollector.TELLO_IP,
                                       FPVDataCollector.TELLO_CMD_PORT))
            response = sock.recv(100)
            return response.decode('utf-8')
        except Exception as e:
            print(f"[ERR] Command '{cmd}' failed: {e}")
            return ""
        finally:
            sock.close()

    @staticmethod
    def _send_no_wait(cmd):
        """Send UDP command without waiting for response."""
        import socket
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.settimeout(0.5)
        try:
            sock.sendto(cmd.encode(), (FPVDataCollector.TELLO_IP,
                                       FPVDataCollector.TELLO_CMD_PORT))
        finally:
            sock.close()

    def _connect_tello(self):
        """Initialize Tello connection and start video stream + load hand detector."""
        print("  Sending 'command'...")
        resp = self._send_command("command")
        if not resp.strip().lower().startswith('ok'):
            print(f"[ERR] Tello not responding: {resp}")
            return False
        print("  Tello connected! Starting video stream...")
        self._send_no_wait("streamon")
        time.sleep(1)
        
        # Initialize HandLandmarker
        script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        model_path = os.path.join(script_dir, "models", "hand_landmarker.task")
        self.mp_landmarker = HandLandmarker.create_from_options(
            HandLandmarkerOptions(
                base_options=base_options.BaseOptions(model_asset_path=model_path),
                running_mode=vision_task_running_mode.VisionTaskRunningMode.IMAGE,
                num_hands=1,
            )
        )
        return True

    def _collect_sample(self, frame, landmarks, device='tello_fpv'):
        """Save one frame + landmarks for the active gesture."""
        if not self.current_gesture or landmarks is None:
            return
        if not hasattr(landmarks, 'landmark'):
            return

        h, w, _ = frame.shape
        landmarks_flat = [
            {'x': l.x, 'y': l.y, 'z': l.z,
             'visibility': l.visibility if hasattr(l, 'visibility') else None}
            for l in landmarks.landmark
        ]

        sample = {
            'frame': frame.tobytes().hex(),
            'width': w, 'height': h,
            'landmarks': landmarks_flat,
            'gesture': self.current_gesture,
            'timestamp': datetime.now().isoformat(),
            'device': 'tello_fpv',
        }
        self.samples.append(sample)
        idx = len(self.samples)
        sample_dir = BASE_DIR / self.current_gesture
        p = sample_dir / f"fpv_sample_{idx:04d}.json"
        with open(p, 'w') as f:
            json.dump(sample, f, indent=2)
        print(f"  Sample {idx:04d}: {self.current_gesture} -> {p.name}")

    def _draw_overlay(self, frame, gesture):
        """Draw capture status overlay on FPV frame."""
        if gesture:
            elapsed = time.time() - self._capture_start
            remaining = max(0, self.CAPTURE_WINDOW - elapsed)
            progress = min(1.0, elapsed / self.CAPTURE_WINDOW)
            bar_w, bar_h = 200, 15
            x, y = 15, 55
            cv2.rectangle(frame, (x, y), (x + bar_w, y + bar_h),
                          (0, 0, 0), -1)
            cv2.rectangle(frame, (x, y),
                         (x + int(bar_w * progress), y + bar_h),
                         (0, 255, 0), -1)
            cv2.rectangle(frame, (x, y), (x + bar_w, y + bar_h),
                          (255, 255, 255), 1)
            cv2.putText(frame, f"Capturing: {gesture}", (15, 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.putText(frame, f"{remaining:.1f}s", (15, 95),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        else:
            cv2.putText(frame, "Press 1-8 to start FPV capture", (15, 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    def run(self):
        """Start the FPV video data collection loop."""
        print("\n" + "=" * 60)
        print("  TELLO FPV DATA COLLECTOR")
        print("=" * 60)

        # Connect to Tello
        if not self._connect_tello():
            return

        # Bind to video stream
        import socket
        vid_ip = "0.0.0.0"
        vid_port = FPVDataCollector.TELLO_VID_PORT
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.bind((vid_ip, vid_port))
        sock.setblocking(False)
        print(f"  Video socket bound to UDP {vid_ip}:{vid_port}")

        # Set up OpenCV for MJPEG decoding
        # Tello sends raw MJPEG frames over UDP
        cap = cv2.VideoCapture()
        cap.open("udp://" + vid_ip + ":" + str(vid_port))

        # Check that the source is valid
        if not cap.isOpened():
            print("[ERR] Could not open video stream. "
                  "Is the drone in range?")
            sock.close()
            return

        window = cv2.namedWindow("FPV Collector", cv2.WINDOW_AUTOSIZE)
        cv2.resizeWindow("FPV Collector", 640, 480)

        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                continue

            self._draw_overlay(frame, self.current_gesture)
            cv2.imshow("FPV Collector", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                if self.current_gesture:
                    print(f"  [SKIP] Aborted {self.current_gesture}")
                    self.current_gesture = None
                    self._capture_start = 0.0
            elif key in GESTURE_KEYS:
                gesture = GESTURE_KEYS[key]
                if gesture != self.current_gesture:
                    self.current_gesture = gesture
                    self._capture_start = time.time()
                    print(f"  [CAPTURE START] {gesture} (3s window)")

            # Hand detection for active gesture capture
            now = time.time()
            if self.current_gesture and (now - self._last_sample_time) >= self.CAPTURE_INTERVAL:
                if self.mp_landmarker:
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
                    detection_result = self.mp_landmarker.detect(mp_image)
                    if detection_result.hand_landmarks and len(detection_result.hand_landmarks) > 0:
                        landmarks = detection_result.hand_landmarks[0]
                        class _LM:
                            def __init__(self, lm):
                                self.landmark = lm
                        self._collect_sample(frame, _LM(landmarks))
                        if self.debug_mode and len(detection_result.hand_landmarks) > 0:
                            h, w2, _ = frame.shape
                            for lm in detection_result.hand_landmarks[0]:
                                x, y = int(lm[0] * w2), int(lm[1] * h)
                                cv2.circle(frame, (x, y), 4, (0, 255, 0), -1)
                self._last_sample_time = now

            # Check capture window timeout
            if self.current_gesture and (now - self._capture_start) >= self.CAPTURE_WINDOW:
                print(f"\n[INFO] Gesture '{self.current_gesture}' capture complete!")
                self.current_gesture = None
                self._capture_start = 0.0
                self._last_sample_time = 0.0

        cap.release()
        sock.close()
        cv2.destroyWindow("FPV Collector")
        self._send_no_wait("streamoff")
        print("  Tello video stream stopped.")

        # Save summary
        counts = {g: sum(1 for s in self.samples if s['gesture'] == g)
                  for g in GESTURE_KEYS.values()}
        with open(SUMMARY_FILE, 'w') as f:
            json.dump({
                "device": "tello_fpv",
                "total_samples": len(self.samples),
                "gesture_counts": counts,
                "data_dir": str(BASE_DIR.absolute()),
                "timestamp": datetime.now().isoformat(),
            }, f, indent=2)
        print("\n" + "=" * 60)
        print("  FPV dataset summary saved.")
        print("=" * 60)


# ── Main entry point ──────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Collect gesture data for Tello drone (webcam or FPV)")
    parser.add_argument('--fpv', action='store_true',
                        help="Use Tello FPV stream instead of webcam")
    parser.add_argument('--data-dir', type=str, default=None,
                        help="Override base data directory")
    args = parser.parse_args()

    global BASE_DIR, SUMMARY_FILE
    if args.data_dir:
        BASE_DIR = Path(args.data_dir)
        SUMMARY_FILE = BASE_DIR / "dataset_summary.json"

    if args.fpv:
        collector = FPVDataCollector()
        collector.run()
    else:
        collector = WebcamDataCollector()
        collector.run()


if __name__ == "__main__":
    main()
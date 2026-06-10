"""
drone_fpv_analyzer.py - FPV stream-based gesture dataset collector for Tello drone.

Uses the same UI as data_collector.py but ingests Tello FPV stream instead of webcam.
Captures gesture performance in compression/composite format.

Controls:
  1 - Open Palm (center)
  2 - Closed Fist (center)
  3 - Thumbs Up (center)
  4 - Thumbs Down (center)
  5 - Palm Up (center)
  6 - Palm Down (center)
  7 - Palm Left (left half of frame)
  8 - Palm Right (right half of frame)
  q - Quit and save
  s - Skip frame
  p - Print current thresholds
"""

import array
import numpy as np
import cv2
import json
import os
import time
import socket
from pathlib import Path


# Gesture mappings (same as data_collector.py)
GESTURE_KEYS = {
    '1': 'open_palm',
    '2': 'closed_fist',
    '3': 'thumbs_up',
    '4': 'thumbs_down',
    '5': 'palm_up',
    '6': 'palm_down',
    '7': 'palm_left',
    '8': 'palm_right',
}

# Base directories
BASE_DIR = Path(__file__).parent.parent / "data_fpv"
SUMMARY_FILE = BASE_DIR / "dataset_summary.json"

# Tello settings
TELLO_IP = "192.168.10.1"
TELLO_COMMAND_PORT = 8889
TELLO_VIDEO_PORT = 11111


class DroneFPVAnalyzer:
    def __init__(self):
        self.current_gesture = None
        self.capture_window = 3.0  # seconds
        self.samples_per_gesture = []  # List of (frame_data, landmark_data, gesture)
        
        # Create FPV dataset directories
        for gesture_name in GESTURE_KEYS.values():
            gesture_dir = BASE_DIR / gesture_name
            gesture_dir.mkdir(exist_ok=True)
            print(f"Created directory: {gesture_dir}")

    def _send_command(self, command: str, timeout: int = 3) -> str:
        """Send command to Tello drone."""
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.settimeout(timeout)
        
        try:
            sock.sendto(command.encode('utf-8'), (TELLO_IP, TELLO_COMMAND_PORT))
            response = sock.recv(100)
            return response.decode('utf-8')
        except Exception as e:
            print(f"Error sending command: {e}")
            return ""
        finally:
            sock.close()

    def _send_command_no_wait(self, command: str):
        """Send command to Tello drone without waiting for response."""
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.sendto(command.encode('utf-8'), (TELLO_IP, TELLO_COMMAND_PORT))
        sock.close()

    def cleanup(self):
        # Stop video stream
        self._send_command_no_wait("streamoff")
        # Land the drone safely
        self._send_command("land")
        
        # Save dataset summary
        self._save_summary()

    def _save_summary(self):
        summary = {
            "total_samples": len(self.samples_per_gesture),
            "gestures_collected": list(set(s[2] for s in self.samples_per_gesture)),
            "data_dir": str(BASE_DIR),
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "device": "tello_fpv",
            "samples_per_gesture": {
                gesture: sum(1 for s in self.samples_per_gesture if s[2] == gesture)
                for gesture in GESTURE_KEYS.values()
            }
        }
        # Convert to JSON-compatible dict
        with open(SUMMARY_FILE, 'w') as f:
            json.dump(summary, f, indent=2)
        print(f"\nFPV dataset summary saved to: {SUMMARY_FILE}")
        print(json.dumps(summary, indent=2))

    def collect_sample(self, frame, landmarks, gesture_name):
        """Collect a single sample for the dataset."""
        if landmarks is None:
            return
        
        # Normalize coordinates for consistency
        h, w, _ = frame.shape
        normalized_landmarks = []
        for lm in landmarks:
            normalized_landmarks.append({
                'x': lm.x,
                'y': lm.y,
                'z': lm.z,
                'visibility': lm.visibility if hasattr(lm, 'visibility') else None
            })
        
        sample_data = {
            'frame': frame.tobytes().hex(),
            'width': w,
            'height': h,
            'landmarks': normalized_landmarks,
            'gesture': gesture_name,
            'timestamp': time.strftime("%Y-%m-%d %H:%M:%S")
        }
        
        self.samples_per_gesture.append(sample_data)
        
        # Save individual sample
        sample_dir = BASE_DIR / gesture_name
        sample_path = sample_dir / f"fpv_sample_{len(self.samples_per_gesture):04d}.json"
        with open(sample_path, 'w') as f:
            json.dump(sample_data, f, indent=2)

        print(f"Sample saved: {sample_path.name} (gesture: {gesture_name})")

    def run_collection(self):
        print("Starting FPV gesture dataset collection...")
        print(f"Collecting to: {BASE_DIR}")
        print("Press 'q' to quit and save dataset")
        print(f"Press 's' to skip frame")
        print(f"Press 'p' to print current thresholds")
        
        window_start = None
        
        # Start FPV stream
        print("Attempting to connect to Tello...")
        response = self._send_command("command")
        if "ok" not in response.lower():
            print(f"Failed to connect to Tello. Response: {response}")
            return
        print("Tello connected!")
        
        # Start video stream
        self._send_command("streamon")
        print("Video stream started.")
        
        # Connect video stream
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.bind(('', TELLO_VIDEO_PORT))
        print("Video socket bound to port 11111")
        
        # Initialize display window
        cv2.namedWindow("FPV Gesture Collection", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("FPV Gesture Collection", 640, 480)
        
        while True:
            try:
                data, addr = sock.recvfrom(1460 * 12)
                # Reassemble the stream (simplified)
                frame_bytes = b""
                while len(data) < 1464 * 12:
                    remaining_data, _ = sock.recvfrom(1460 * 12)
                    data += remaining_data
                frame_data = bytes.fromhex(data.hex())
                
                frame = cv2.imdecode(np.asarray(bytearray(frame_data)), cv2.IMREAD_COLOR)
                if frame is None:
                    continue
                    
                frame_copy = frame.copy()  # Use a copy for drawing
                h, w, _ = frame_copy.shape
                
                # Display current status
                cv2.putText(frame_copy, f"Frame: {h}x{w}", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                if self.current_gesture:
                    cv2.rectangle(frame_copy, (10, 50), (200, 100), (0, 255, 0), 2)
                    cv2.putText(frame_copy, f"Capturing: {self.current_gesture}", (20, 85),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    
                    # Update capture window timing if needed
                    if window_start is None:
                        window_start = time.time()
                    elapsed = time.time() - window_start
                    remaining = max(0, self.capture_window - elapsed)
                    
                    if remaining > 0:
                        cv2.putText(frame_copy, f"Time left: {remaining:.1f}s", (20, 120),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                        
                        if remaining == 0:
                            # Process the frame when window expires
                            # Since landmarks are not actually available, we'd typically use MediaPipe here
                            # For testing/development, let's create a placeholder
                            pass
                else:
                    # No capture in progress, show info
                    cv2.rectangle(frame_copy, (10, 60), (300, 140), (0, 0, 255), 2)
                    cv2.putText(frame_copy, "Press 1-8 to start capture", (20, 100),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
                    cv2.putText(frame_copy, "Press q to quit", (20, 125),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
                    cv2.putText(frame_copy, "Press s to skip frame", (20, 150),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
                    cv2.putText(frame_copy, "Press p to print thresholds", (20, 175),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
                
                # Process keys
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('s'):
                    if self.current_gesture:
                        print("Skipping frame...")
                        self.current_gesture = None
                        window_start = None
                elif key == ord('p'):
                    print("Current thresholds would be printed here (placeholder implementation)")
                elif key in GESTURE_KEYS:
                    gesture = GESTURE_KEYS[str(key)]
                    if gesture != self.current_gesture:
                        self.current_gesture = gesture
                        window_start = time.time()
                        print(f"Starting capture for: {gesture}")
                
                cv2.imshow("FPV Gesture Collection", frame_copy)
                
            except KeyboardInterrupt:
                print("\nInterrupted by user")
                break
            except Exception as e:
                print(f"Error processing frame: {e}")
                continue
        
        cv2.destroyWindow("FPV Gesture Collection")
    
    def cleanup_and_exit(self):
        self.cleanup()
        cv2.destroyAllWindows()


def main():
    collector = DroneFPVAnalyzer()
    try:
        collector.run_collection()
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    finally:
        collector.cleanup_and_exit()


if __name__ == "__main__":
    main()
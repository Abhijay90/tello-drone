# drone_controller.py — Central Tello movement layer.
# All control systems (keyboard, gestures, ...) drive the drone through this class.
# Imports only djitellopy + cv2 — no input or vision dependencies.

import time

import cv2
from djitellopy import tello
from djitellopy.tello import TelloException


def _int(value):
    try:
        return int(value)
    except (TypeError, ValueError):
        return -1


class DroneController:
    """Reusable wrapper around djitellopy with a small movement API."""

    MAX_VELOCITY = 100

    def __init__(self):
        self.drone = tello.Tello()
        self.connected = False
        self.airborne = False
        self.last_command = None
        self._frame_read = None

    # --- lifecycle ---

    def connect(self):
        """Connect to the drone. Raises RuntimeError with a clear message on failure."""
        try:
            self.drone.connect()
        except TelloException as e:
            raise RuntimeError(
                f"No response from Tello ({e}). "
                "Is it powered on and on the same Wi-Fi network?"
            )
        self.connected = True
        print(f"Drone connected. Battery: {self.get_state()['battery']}%")

    def disconnect(self):
        """Safe shutdown: land if airborne, stop stream, close connection."""
        try:
            if self.airborne:
                self.land()
                time.sleep(2)
            self.streamoff()
        except Exception as e:
            print(f"Warning during shutdown: {e}")
        finally:
            try:
                self.drone.end()
            except Exception:
                pass
            self.connected = False
            self.airborne = False

    # --- flight ---

    def takeoff(self):
        """Automatic takeoff (drone default height). Blocks until the drone acks."""
        if not self.connected:
            raise RuntimeError("Drone not connected")
        self.last_command = None
        self.drone.takeoff()
        self.airborne = True
        print("Takeoff command sent.")

    def land(self):
        """Hover briefly to kill momentum, then land."""
        self.hover()
        time.sleep(1)
        self.drone.land()
        self.airborne = False
        print("Landing...")

    # --- movement ---

    def move(self, lr=0, fb=0, ud=0, yaw=0):
        """Send RC control (clamped to ±100). Only transmits when the command changes."""
        cmd = tuple(self._clamp(v) for v in (lr, fb, ud, yaw))
        if cmd != self.last_command:
            self.drone.send_rc_control(*cmd)
            self.last_command = cmd
        return cmd

    def hover(self):
        """Hold position (zero velocity on all axes)."""
        return self.move(0, 0, 0, 0)

    # --- video ---

    def streamon(self):
        self.drone.streamon()
        self._frame_read = self.drone.get_frame_read()

    def streamoff(self):
        try:
            self.drone.streamoff()
        finally:
            self._frame_read = None

    def get_frame(self):
        """Latest BGR frame for cv2, or None if no video yet."""
        if self._frame_read is None:
            return None
        rgb = self._frame_read.frame
        if rgb is None:
            return None
        return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

    # --- telemetry ---

    def get_state(self):
        """Telemetry dict: altitude (cm), vgx/vgy (velocity), battery (%). -1 when unknown."""
        state = self.drone.get_current_state() or {}
        return {
            'altitude': _int(state.get('h')),
            'vgx': _int(state.get('vgx')),
            'vgy': _int(state.get('vgy')),
            'battery': _int(state.get('bat')),
        }

    @staticmethod
    def _clamp(v):
        v = int(v)
        return max(-DroneController.MAX_VELOCITY, min(DroneController.MAX_VELOCITY, v))

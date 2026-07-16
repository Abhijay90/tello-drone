# gesture_rc_mapping.py — Maps gesture names to Tello RC control commands
# send_rc_control(left_right_velocity, forward_backward_velocity, up_down_velocity, yaw_velocity)

HOVER_COMMAND = (0, 0, 0, 0)

GESTURE_RC_MAP = {
    "OPEN_PALM":     (0, -30, 0, 0),   # Backward
    "CLOSED_FIST":   (0, 30, 0, 0),    # Forward
    "THUMBS_UP":     (0, 0, 20, 0),    # Up
    "THUMBS_DOWN":   (0, 0, -20, 0),   # Down
    "PALM_LEFT":     (30, 0, 0, 0),   # Left
    "PALM_RIGHT":    (-30, 0, 0, 0),    # Right
}

def gesture_to_rc(gesture_name):
    return GESTURE_RC_MAP.get(gesture_name, HOVER_COMMAND)
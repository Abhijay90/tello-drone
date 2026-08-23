# drone_control.py — Unified flight control entrypoint for Tello.
# Auto-detects input source: gamepad if connected, else keyboard. Hot-plug safe:
# plug/unplug the pad at any time and the input source switches live (drone hovers
# during a switch). The video window never needs focus in gamepad mode.
#
# Gamepad (cine-style):
#   Left stick X/Y  -> left-right / forward-backward
#   Right stick X/Y -> yaw / up-down
#   A (btn 0) takeoff    B (btn 1) land + exit    Start (btn 9) emergency hover
# Keyboard (fallback):
#   W/S fwd/back  A/D left/right  R/F up/down  Q/E yaw
#   SPACE takeoff  L land + exit  ESC emergency

import sys
import time

import cv2
import numpy as np
import pygame

from drone_controller import DroneController

# --- gamepad config (remap here for your pad) ---
GAMEPAD_INDEX = 0
# (axis_index, rc_channel, scale) — channel: 0=lr 1=fb 2=ud 3=yaw; scale -1 flips direction
GAMEPAD_AXES = (
    (0, 0, +1),   # LX -> left/right
    (1, 1, -1),   # LY -> forward/backward (push forward = move forward)
    (2, 3, +1),   # RX -> yaw
    (3, 2, -1),   # RY -> up/down (pull down = descend)
)
BTN_TAKEOFF = 0      # A
BTN_LAND = 1         # B
BTN_EMERGENCY = 9    # Start
GAMEPAD_MAX_VELOCITY = 50  # full-stick speed in cm/s (drone max is 100)
AXIS_DEADZONE = 0.15

# --- keyboard config ---
KEY_MAP = {
    pygame.K_w: (0, 30, 0, 0),     # forward
    pygame.K_s: (0, -30, 0, 0),    # backward
    pygame.K_a: (-30, 0, 0, 0),    # left
    pygame.K_d: (30, 0, 0, 0),     # right
    pygame.K_r: (0, 0, 20, 0),     # up
    pygame.K_f: (0, 0, -20, 0),    # down
    pygame.K_q: (0, 0, 0, 30),     # yaw left
    pygame.K_e: (0, 0, 0, -30),    # yaw right
}

LOOP_HZ = 30
STATE_INTERVAL = 1.0       # seconds between telemetry reads
CONSOLE_INTERVAL = 2.0     # seconds between console state prints
CONTROL_WINDOW = (420, 180)
VIDEO_TITLE = "Tello Drone Control"

KEYBOARD_HELP = [
    "W/S fwd/back   A/D left/right",
    "R/F up/down    Q/E yaw",
    "SPACE takeoff  L land+exit  ESC emergency",
    "(click this window to control the drone)",
]
GAMEPAD_HELP = [
    "L-stick: move    R-stick: yaw / up-down",
    "A takeoff   B land+exit   Start emergency",
    "(gamepad works without clicking any window)",
]


def get_gamepad(index):
    if pygame.joystick.get_count() > index:
        js = pygame.joystick.Joystick(index)
        js.init()
        return js
    return None


def read_gamepad_cmd(js):
    """Analog command from sticks (deadzone applied), plus a display string."""
    cmd = [0, 0, 0, 0]
    for axis_idx, channel, scale in GAMEPAD_AXES:
        v = js.get_axis(axis_idx)
        if abs(v) < AXIS_DEADZONE:
            v = 0.0
        cmd[channel] = int(round(scale * v * GAMEPAD_MAX_VELOCITY))
    display = "  ".join(
        f"A{axis_idx}={js.get_axis(axis_idx):+.2f}" for axis_idx, _, _ in GAMEPAD_AXES
    )
    return cmd, display


def read_new_buttons(js, prev_pressed):
    """Edge-triggered: buttons pressed since last poll."""
    num = js.get_numbuttons()
    pressed = {i for i in range(num) if js.get_button(i)}
    new = [b for b in pressed if b not in prev_pressed]
    return new, pressed


def draw_control_window(screen, font, input_mode, js, status_line, focused):
    screen.fill((30, 30, 30))
    y = 8
    if input_mode == "gamepad" and js is not None:
        lines = GAMEPAD_HELP + [f"pad: {js.get_name()[:40]}"]
        focused = True  # focus irrelevant for gamepad
    else:
        lines = KEYBOARD_HELP
    for line in lines:
        color = (255, 255, 255) if focused else (120, 120, 120)
        screen.blit(font.render(line, True, color), (10, y))
        y += 24
    screen.blit(font.render(status_line, True, (0, 255, 0)), (10, y + 4))
    pygame.display.flip()


def make_placeholder():
    frame = np.zeros((360, 640, 3), dtype=np.uint8)
    cv2.putText(frame, "Waiting for drone video...", (90, 170),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 200), 2)
    cv2.putText(frame, "(stream may take a few seconds)", (110, 205),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (140, 140, 140), 1)
    return frame


def draw_overlay(frame, state, input_mode, status_line, focused):
    status = "AIRBORNE" if state.get('airborne') else "GROUNDED"
    line1 = f"Alt: {state['altitude']} cm   Vg: {state['vgx']}/{state['vgy']}   Bat: {state['battery']}%"
    line2 = f"{status}   [{input_mode}]   {status_line}"
    y = 24
    for line in (line1, line2):
        cv2.putText(frame, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3)
        cv2.putText(frame, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
        y += 24
    if input_mode == "keyboard" and not focused:
        msg = "CLICK THE CONTROL WINDOW TO USE KEYBOARD"
        cv2.putText(frame, msg, (10, y + 6), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3)
        cv2.putText(frame, msg, (10, y + 6), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 1)


def main():
    pygame.init()
    pygame.joystick.init()
    screen = pygame.display.set_mode(CONTROL_WINDOW)
    pygame.display.set_caption("Tello Drone Control")
    font = pygame.font.SysFont(None, 20)

    js = get_gamepad(GAMEPAD_INDEX)
    input_mode = "gamepad" if js else "keyboard"
    if js:
        print(f"[INPUT] Gamepad connected: {js.get_name()} — gamepad mode")
    else:
        print("[INPUT] No gamepad found — keyboard mode (plug in a pad to switch)")

    controller = DroneController()
    try:
        controller.connect()
    except RuntimeError as e:
        print(f"ERROR: {e}")
        pygame.quit()
        sys.exit(1)

    try:
        controller.streamon()
    except Exception as e:
        print(f"Warning: video stream unavailable: {e}")

    clock = pygame.time.Clock()
    last_state_time = 0.0
    last_console_time = 0.0
    state = {'altitude': -1, 'vgx': -1, 'vgy': -1, 'battery': -1}
    prev_buttons = set()
    focused = True
    running = True

    try:
        while running:
            now = time.time()

            # --- events: hot-plug + edge-triggered actions ---
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.WINDOWFOCUSLOST:
                    focused = False
                elif event.type == pygame.WINDOWFOCUSGAINED:
                    focused = True
                elif event.type == pygame.JOYDEVICEADDED and js is None:
                    js = get_gamepad(event.device_index)
                    if js:
                        input_mode = "gamepad"
                        prev_buttons = set()
                        print(f"[INPUT] Gamepad connected: {js.get_name()} — gamepad mode")
                elif event.type == pygame.JOYDEVICEREMOVED and js is not None:
                    js = None
                    input_mode = "keyboard"
                    focused = True
                    print("[INPUT] Gamepad disconnected — keyboard mode (drone hovering)")
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_SPACE and not controller.airborne:
                        print("Taking off...")
                        controller.takeoff()
                        time.sleep(2)  # wait until stable before accepting movement
                    elif event.key == pygame.K_ESCAPE:
                        print("ESC — hovering, then landing...")
                        controller.hover()
                        time.sleep(1)
                        running = False
                    elif event.key == pygame.K_l:
                        print("L — landing and exiting...")
                        running = False

            # --- input source -> command ---
            if js is not None:
                cmd, status_line = read_gamepad_cmd(js)
                new_buttons, prev_buttons = read_new_buttons(js, prev_buttons)
                for b in new_buttons:
                    if b == BTN_TAKEOFF and not controller.airborne:
                        print("Gamepad A — taking off...")
                        controller.takeoff()
                        time.sleep(2)
                    elif b == BTN_LAND:
                        print("Gamepad B — landing and exiting...")
                        running = False
                    elif b == BTN_EMERGENCY:
                        print("Gamepad Start — hovering, then landing...")
                        controller.hover()
                        time.sleep(1)
                        running = False
            else:
                pressed = pygame.key.get_pressed()
                cmd = [0, 0, 0, 0]
                active_keys = []
                for key, rc in KEY_MAP.items():
                    if pressed[key]:
                        for i in range(4):
                            cmd[i] += rc[i]
                        active_keys.append(pygame.key.name(key))
                status_line = f"keys: {', '.join(active_keys) if active_keys else '--'}"

            controller.move(*cmd)

            # --- telemetry (1 Hz reads, 2 Hz console prints) ---
            if now - last_state_time >= STATE_INTERVAL:
                last_state_time = now
                state = controller.get_state()
                state['airborne'] = controller.airborne
                if now - last_console_time >= CONSOLE_INTERVAL:
                    last_console_time = now
                    print(f"[STATE] mode={input_mode} Alt={state['altitude']}cm "
                          f"Vgx={state['vgx']} Vgy={state['vgy']} Bat={state['battery']}%")

            # --- FPV window ---
            frame = controller.get_frame()
            if frame is None:
                frame = make_placeholder()
            draw_overlay(frame, state, input_mode, status_line, focused)
            cv2.imshow(VIDEO_TITLE, frame)
            if cv2.waitKey(1) & 0xFF == ord('l'):
                print("L (video window) — landing and exiting...")
                running = False

            draw_control_window(screen, font, input_mode, js, status_line, focused)
            clock.tick(LOOP_HZ)
    finally:
        controller.disconnect()
        cv2.destroyAllWindows()
        pygame.quit()
        print("Exit complete.")


if __name__ == '__main__':
    main()

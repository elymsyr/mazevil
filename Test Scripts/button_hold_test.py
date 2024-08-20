import win32api
import win32con, pydirectinput

# W: 0x57
# A: 0x41
# S: 0x53
# D: 0x44

def press_key(hexKeyCode):
    win32api.keybd_event(hexKeyCode, 0, 0, 0)

def release_key(hexKeyCode):
    win32api.keybd_event(hexKeyCode, 0, win32con.KEYEVENTF_KEYUP, 0)

def is_key_held(hexKeyCode):
    return win32api.GetAsyncKeyState(hexKeyCode) != 0

# import time

# # Press W key
# time.sleep(1)  # Hold the key for 1 second
# press_key(0x57)
# time.sleep(1)  # Hold the key for 1 second

# # Check if W key is held
# if is_key_held(0x57):
#     print("W key is currently held")

# # Release W key
# release_key(0x57)

import pyautogui
import time

def hold_key(key, hold_time):
    start = time.time()
    while time.time() - start < hold_time:
        pyautogui.keyDown(key)

time.sleep(1)  # Hold the key for 1 second
pydirectinput.keyDown('W') # Simulate pressing dwon the Alt key.
time.sleep(1)  # Hold the key for 1 second
pydirectinput.keyUp('W') # Simulate releasing the Alt key.

# pydirectinput.press('S')
# pydirectinput.press('S')

pydirectinput.press('s', presses=2, interval=0.1, delay=0.01, duration=0.01)
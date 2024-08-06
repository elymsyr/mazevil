import os
import time
import pyautogui
import pygetwindow as gw

def get_last_screenshot_number(folder_path):
    """Finds the last screenshot number in the specified folder."""
    screenshots = [f for f in os.listdir(folder_path) if f.startswith('screenshot_') and f.endswith('.png')]
    if not screenshots:
        return 0
    last_screenshot = max(screenshots, key=lambda f: int(f.split('_')[1].split('.')[0]))
    last_number = int(last_screenshot.split('_')[1].split('.')[0])
    return last_number

def take_screenshot_of_window(window_title, folder_path, duration=None, number_of_screenshots=None, interval=1):
    """Takes screenshots of a specific window sequentially for a set time or a number of screenshots.
    
    Parameters:
        window_title (str): The title of the window to capture.
        folder_path (str): The folder to save the screenshots.
        duration (int): The duration in seconds to take screenshots. If None, `number_of_screenshots` is used.
        number_of_screenshots (int): The number of screenshots to take. If None, `duration` is used.
        interval (int): The interval in seconds between each screenshot.
    """
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    
    window = gw.getWindowsWithTitle(window_title)
    if not window:
        print(f"No window found with title '{window_title}'")
        return
    window = window[0]
    
    last_number = get_last_screenshot_number(folder_path)
    avg = 6
    if duration is not None:
        end_time = time.time() + duration
        while time.time() < end_time:
            last_number += 1
            screenshot_path = os.path.join(folder_path, f'screenshot_{last_number}.png')
            time.sleep(0.5)  # Give the window some time to come into focus
            left, top, width, height = window.left, window.top, window.width, window.height
            screenshot = pyautogui.screenshot(region=(left+8, top+31, width-16, height-39))
            screenshot.save(screenshot_path)
            print(f'Screenshot saved as {screenshot_path}')
            time.sleep(interval)
    elif number_of_screenshots is not None:
        for _ in range(number_of_screenshots):
            last_number += 1
            screenshot_path = os.path.join(folder_path, f'screenshot_{last_number}.png')
            time.sleep(0.5)  # Give the window some time to come into focus
            left, top, width, height = window.left, window.top, window.width, window.height
            screenshot = pyautogui.screenshot(region=(left+8, top+31, width-16, height-39))
            screenshot.save(screenshot_path)
            print(f'Screenshot saved as {screenshot_path}')
            time.sleep(interval)
    else:
        print("Either duration or number_of_screenshots must be provided.")

# Usage examples
folder_path = 'YOLO/Data/Images'
window_title = 'Mazevil'

take_screenshot_of_window(window_title, folder_path, duration=10, interval=0.4)

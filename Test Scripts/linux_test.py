import cv2
import numpy as np
import mss, Xlib
import Xlib.display

def capture_and_display(x, y, width, height):
    with mss.mss() as sct:
        # Define the region to capture
        monitor = {
            "top": y,
            "left": x,
            "width": width,
            "height": height
        }

        while True:
            # Capture the region
            img = sct.grab(monitor)
            # Convert to a format compatible with OpenCV
            img_np = np.array(img)
            # Convert from BGRA to BGR (OpenCV uses BGR)
            img_bgr = cv2.cvtColor(img_np, cv2.COLOR_BGRA2BGR)
            # Display the image
            cv2.imshow('Window Capture', img_bgr)

            # Break the loop if 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cv2.destroyAllWindows()

def create_monitor(self):
    geometry = window.query_tree().parent.get_geometry()
    monitor = {
        "top": geometry.y,
        "left": geometry.x,
        "width": geometry.width,
        "height": geometry.height
    }

def get_window_geometry():
    geometry = window.query_tree().parent.get_geometry()
    x = geometry.x
    y = geometry.y
    print(f"{y=}, {x=}, {y + geometry.height=}, {x + geometry.width=}, {geometry.width=}, {geometry.height=}")
    return y, x, y + geometry.height, x + geometry.width, geometry.width, geometry.height


display = Xlib.display.Display()
root = display.screen().root
window_ids = root.get_full_property(display.intern_atom('_NET_CLIENT_LIST'), Xlib.X.AnyPropertyType).value        
for window_id in window_ids:
    window = display.create_resource_object('window', window_id)
    window_name_str = window.get_wm_name()
    if window_name_str and 'Mazevil' in window_name_str:        
        window = window
        window_id = window_id

top, left, bottom, right, width, height = get_window_geometry()
window_x = top
window_y = left

top_offset, bottom_offset = 50, 58
monitor = {
    "y": top,
    "x": left,
    "width": width,
    "height": height,
    "top" : top,
    "left" : left,
    "bottom" : bottom,
    "right" : right,
}

# Example region (you can adjust these values to fit your needs)
capture_and_display(x=monitor['x'], y=monitor['y'], width=monitor['width'], height=monitor['height'])

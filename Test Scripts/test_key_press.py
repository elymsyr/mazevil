from pynput import keyboard

# Set to keep track of currently pressed keys
pressed_keys = set()

def on_press(key):
    try:
        pressed_keys.add(key.char)
    except AttributeError:
        pressed_keys.add(str(key))

def on_release(key):
    try:
        pressed_keys.discard(key.char)
    except AttributeError:
        pressed_keys.discard(str(key))

def get_currently_pressed_keys():
    global pressed_keys
    return pressed_keys

# Start listening for key events
with keyboard.Listener(on_press=on_press, on_release=on_release) as listener:
    try:
        while True:
            keys = get_currently_pressed_keys()
            print("Currently pressed keys:", keys)
    except KeyboardInterrupt:
        listener.stop()

'''
libclicker: Lightweight Python library to
simulate mouse clicks, scrolls and key presses
on X11 and Wayland. Does not bother itself with
ANYTHING ELSE.
'''

import uinput
import time
import string
import os, sys
from typing import Union

keys = [uinput.KEY_LEFTSHIFT,
    uinput.KEY_SPACE,
    uinput.BTN_LEFT,
    uinput.BTN_MIDDLE,
    uinput.BTN_RIGHT,
    uinput.REL_X,
    uinput.REL_Y,
    uinput.REL_WHEEL,
    uinput.REL_HWHEEL,
    uinput.KEY_TAB,
    uinput.KEY_ENTER
            ]

punctuation = {',':'COMMA', '.':'DOT', '/':'SLASH', ';':'SEMICOLON', "'":'APOSTROPHE',
            '[':'LEFTBRACE', ']':'RIGHTBRACE', '\\':'BACKSLASH', '-':'MINUS', '=':'EQUAL', '`':'GRAVE'}
punctuation_shift = {'<':'COMMA', '>':'DOT', '?':'SLASH', ':':'SEMICOLON', '"':'APOSTROPHE',
            '{':'LEFTBRACE', '}':'RIGHTBRACE', '|':'BACKSLASH', '_':'MINUS', '+':'EQUAL', '~':'GRAVE'}
punctuation_ = ',./;\'[]\\-=`'
punctuation_shift_ = '<>?:\"{}|_+~'
digit_symbols = {'!':'1', '@':'2', '#':'3', '$':'4', '%':'5', '^':'6', '&':'7', '*':'8', '(':'9', ')':'0'}

for i in string.ascii_lowercase:
    keys.append(eval('uinput.KEY_{}'.format(i.upper())))
for i in string.digits:
    keys.append(eval('uinput.KEY_{}'.format(i)))
for i in digit_symbols:
    keys.append(eval('uinput.KEY_{}'.format(digit_symbols[i])))
for i in punctuation_:
    keys.append(eval('uinput.KEY_{}'.format(punctuation[i])))
for i in punctuation_shift_:
    keys.append(eval('uinput.KEY_{}'.format(punctuation_shift[i])))

device = uinput.Device(keys)

# Wait for the system to detect the new device
time.sleep(1)

# Move the mouse to a given position

def move_mouse(x : int, y : int):
    device.emit(uinput.REL_X, -25000)
    device.emit(uinput.REL_Y, -25000)
    device.emit(uinput.REL_X, x//2)
    device.emit(uinput.REL_Y, y//2)

# Click at a given position

def click(x : int, y : int, btn : int = 0, count : int = 1):
    # Check each argument

    if btn > 2 or btn < 0 or type(btn) != int:
        raise ValueError('btn must be 0, 1 or 2')
    if count > 3 or count < 1 or type(count) != int:
        raise ValueError('count must be 1, 2 or 3')
    
    move_mouse(x, y)
    
    # Click the desired button
    if btn == 0:
        for i in range(count):
            device.emit(uinput.BTN_LEFT, 1)
            device.emit(uinput.BTN_LEFT, 0)
    elif btn == 1:
        for i in range(count):
            time.sleep(0.3)
            device.emit(uinput.BTN_MIDDLE, 1)
            device.emit(uinput.BTN_MIDDLE, 0)
    elif btn == 2:
        for i in range(count):
            time.sleep(0.3)
            device.emit(uinput.BTN_RIGHT, 1)
            device.emit(uinput.BTN_RIGHT, 0)

# Scroll up or down

def scroll(x : int, y : int, count : int, direction : str = 'down'):
    if type(direction) != str or direction.lower() not in ['up', 'down', 'left', 'right']:
        raise ValueError('direction must be up, down, left or right')

    move_mouse(x, y)
    
    if direction.lower() == 'up' or direction.lower() == 'left':
        val = 1
    elif direction.lower() == 'down' or direction.lower() == 'right':
        val = -1
    if direction.lower() == 'up' or direction.lower() == 'down':
        for i in range(count):
            device.emit(uinput.REL_WHEEL, val)
    elif direction.lower() == 'left' or direction.lower() == 'right':
        for i in range(count):
            device.emit(uinput.REL_HWHEEL, val)

# Press a key

def press_key(key : str):
    if type(key) != str:
        raise ValueError('key must be a string')
    if key not in string.printable + '\t\n':
        raise ValueError('key must be printable')
    if len(key) > 1:
        raise ValueError('key must be a single character')
    
    if key in string.ascii_lowercase:
        device.emit(eval('uinput.KEY_{}'.format(key.upper())), 1)
        device.emit(eval('uinput.KEY_{}'.format(key.upper())), 0)
    elif key in string.ascii_uppercase:
        device.emit(uinput.KEY_LEFTSHIFT, 1)
        device.emit(eval('uinput.KEY_{}'.format(key.upper())), 1)
        device.emit(eval('uinput.KEY_{}'.format(key.upper())), 0)
        device.emit(uinput.KEY_LEFTSHIFT, 0)
    elif key in string.digits:
        device.emit(eval('uinput.KEY_{}'.format(key)), 1)
        device.emit(eval('uinput.KEY_{}'.format(key)), 0)
    elif key in digit_symbols:
        device.emit(uinput.KEY_LEFTSHIFT, 1)
        device.emit(eval('uinput.KEY_{}'.format(digit_symbols[key])), 1)
        device.emit(eval('uinput.KEY_{}'.format(digit_symbols[key])), 0)
        device.emit(uinput.KEY_LEFTSHIFT, 0)
    elif key in punctuation_:
        device.emit(eval('uinput.KEY_{}'.format(punctuation[key])), 1)
        device.emit(eval('uinput.KEY_{}'.format(punctuation[key])), 0)
    elif key in punctuation_shift_:
        device.emit(uinput.KEY_LEFTSHIFT, 1)
        device.emit(eval('uinput.KEY_{}'.format(punctuation_shift[key])), 1)
        device.emit(eval('uinput.KEY_{}'.format(punctuation_shift[key])), 0)
        device.emit(uinput.KEY_LEFTSHIFT, 0)
    elif key == ' ':
        device.emit(uinput.KEY_SPACE, 1)
        device.emit(uinput.KEY_SPACE, 0)
    elif key == '\t':
        device.emit(uinput.KEY_TAB, 1)
        device.emit(uinput.KEY_TAB, 0)
    elif key == '\n':
        device.emit(uinput.KEY_ENTER, 1)
        device.emit(uinput.KEY_ENTER, 0)
    
# Type text

def type_text(text : str):
    if type(text) != str:
        raise ValueError('text must be a string')
    for i in text:
        if i not in string.printable:
            raise ValueError('charactert {} is not printable'.format(i))
    
    for key in text:
        press_key(key)

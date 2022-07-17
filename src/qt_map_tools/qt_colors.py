'''
Colors for maps and plots

Source: https://www.rapidtables.com/web/color/green-color.html
'''

from PyQt5 import QtGui
import numpy as np

class Colors:

    # Red
    lightsalmon = QtGui.QColor(255, 160, 122),
    salmon = QtGui.QColor(250, 128, 114),
    darksalmon = QtGui.QColor(233, 150, 122),
    lightcoral = QtGui.QColor(240, 128, 128),
    indianred = QtGui.QColor(205, 92, 92),
    crimson = QtGui.QColor(220, 20, 60),
    firebrick = QtGui.QColor(178, 34, 34),
    red = QtGui.QColor(255, 0, 0),
    darkred = QtGui.QColor(139, 0, 0),
    maroon = QtGui.QColor(128, 0, 0),
    palevioletred = QtGui.QColor(219, 112, 147),

    # Orange
    coral = QtGui.QColor(255, 127, 80),
    tomato = QtGui.QColor(255, 99, 71),
    orangered = QtGui.QColor(255, 69, 0),
    gold = QtGui.QColor(255, 215, 0),
    orange = QtGui.QColor(255, 165, 0),
    darkorange = QtGui.QColor(255, 140, 0),

    # Blue
    deepskyblue = QtGui.QColor(0, 191, 255),
    dodgerblue = QtGui.QColor(30, 144, 255),
    mediumslateblue = QtGui.QColor(123, 104, 238),
    blue = QtGui.QColor(0, 0, 255),
    mediumblue = QtGui.QColor(0, 0, 205),
    darkblue = QtGui.QColor(0, 0, 139),
    navy = QtGui.QColor(0, 0, 128),
    midnightblue = QtGui.QColor(25, 25, 112),
    blueviolet = QtGui.QColor(138, 43, 226),
    indigo = QtGui.QColor(75, 0, 130),

    # Brown
    sandybrown = QtGui.QColor(244, 164, 96),
    goldenrod = QtGui.QColor(218, 165, 32),
    peru = QtGui.QColor(205, 133, 63),
    chocolate = QtGui.QColor(210, 105, 30),
    saddlebrown = QtGui.QColor(139, 69, 19),
    sienna = QtGui.QColor(160, 82, 45),
    brown = QtGui.QColor(165, 42, 42),
    maroon = QtGui.QColor(128, 0, 0),

    # Green
    teal = QtGui.QColor(0, 128, 128),
    green = QtGui.QColor(0, 128, 0),
    darkgreen = QtGui.QColor(0, 100, 0),
    mediumseagreen = QtGui.QColor(60, 179, 113),
    lightseagreen = QtGui.QColor(32, 178, 170),
    seagreen = QtGui.QColor(46, 139, 87),
    olive = QtGui.QColor(128, 128, 0),
    darkolivegreen = QtGui.QColor(85, 107, 47),
    olivedrab = QtGui.QColor(107, 142, 35),
    limegreen = QtGui.QColor(50, 205, 50),

    # Purple
    mediumpurple = QtGui.QColor(147, 112, 219),
    blueviolet = QtGui.QColor(138, 43, 226),
    darkviolet = QtGui.QColor(148, 0, 211),
    darkorchid = QtGui.QColor(153, 50, 204),
    darkmagenta = QtGui.QColor(139, 0, 139),
    purple = QtGui.QColor(128, 0, 128),
    indigo = QtGui.QColor(75, 0, 130),

    # Turquoise
    paleturquoise = QtGui.QColor(175, 238, 238),
    turquoise = QtGui.QColor(64, 224, 208),
    mediumturquoise = QtGui.QColor(72, 209, 204),
    darkturquoise = QtGui.QColor(0, 206, 209),

    # Gray
    gainsboro = QtGui.QColor(220, 220, 220),
    lightgray = QtGui.QColor(211, 211, 211),
    silver = QtGui.QColor(192, 192, 192),
    darkgray = QtGui.QColor(169, 169, 169),
    gray = QtGui.QColor(128, 128, 128),
    dimgray = QtGui.QColor(105, 105, 105),
    lightslategray = QtGui.QColor(119, 136, 153),
    slategray = QtGui.QColor(112, 128, 144),
    darkslategray = QtGui.QColor(47, 79, 79),
    darkgray = QtGui.QColor(50, 50, 50),
    charcoalgray = QtGui.QColor(25, 25, 25),
    black = QtGui.QColor(0, 0, 0),
    white = QtGui.QColor(255, 255, 255)

    default_colors = [
        firebrick, darkorange, orange, gold, goldenrod, darkgreen, seagreen, mediumseagreen, lightseagreen, limegreen,
        olivedrab, mediumblue, deepskyblue, dodgerblue, darkviolet, purple, salmon, turquoise, darkturquoise, brown,
        darkslategray]


def hex_to_rgb(hex_str: str) -> QtGui.QColor:
    '''
    Convert a hex color string, e.g., '#0155FF', to an RGB value (QtGui.QColor)
    '''
    hex_str = hex_str.lstrip('#')
    lv = len(hex_str)
    t = tuple(int(hex_str[i:i + lv // 3], 16) for i in range(0, lv, lv // 3))
    return rgb(t[0], t[1], t[2])


def rgb(r: int, g: int, b: int):
    '''
    Return an RGB color object (QtGui.QColor), given red, green, and blue values
    '''
    return QtGui.QColor(r, g, b)


def get_color_index(value: float, vmin: float, vmax: float, ncolors: int) -> int:
    '''
    Get the color index, given a floating point value, minimum, maximum, and number of colors
    '''
    cmin = 0
    cmax = ncolors - 1
    return int((value - vmin)/(vmax - vmin) * (cmax - cmin))


def get_color(value, vmin, vmax, colors, step_type='linear') -> QtGui.QColor:
    '''
    Get the color index, given a floating point value, minimum, maximum, and number of colors
    '''
    R, G, B = [], [], []
    for c in colors:
        rgb = c.getRgb()
        R.append(rgb[0])
        G.append(rgb[1])
        B.append(rgb[2])
    ncolors = len(colors)
    cmin = 0
    cmax = ncolors - 1
    if step_type == 'step':
        ci = int((value - vmin)/(vmax - vmin) * (cmax - cmin))
    elif step_type == 'linear':
        ci = (value - vmin)/(vmax - vmin) * (cmax - cmin)
    else:
        ci = (value - vmin)/(vmax - vmin) * (cmax - cmin)
    r = np.interp(ci, np.arange(len(colors)), R)
    g = np.interp(ci, np.arange(len(colors)), G)
    b = np.interp(ci, np.arange(len(colors)), B)
    return QtGui.QColor(r, g, b)

from PyQt5 import QtWidgets as qtw
from PyQt5 import QtCore as qtc
from PyQt5 import QtGui as qtg
import numpy as np
from .qt_geometry import Point
from .qt_colors import Colors

class Map:

    def __init__(self, map_width: float, map_height: float, xmin: float = 0,
                xmax: float = 1, ymin: float = 0, ymax: float = 1,
                dx: float = 0.1, dy: float = 0.1, buffer: float = 40):
        self.map_width = map_width
        self.map_height = map_height
        self.xmin = xmin
        self.xmax = xmax
        self.ymin = ymin
        self.ymax = ymax
        self.dx = dx
        self.dy = dy
        self.buffer = buffer

    def draw(self, painter):
        painter.setOpacity(1.0)
        brush = qtg.QBrush(Colors.white)
        painter.setBrush(brush)
        painter.drawRect(0, 0, self.map_width, self.map_height)


class MapBorder:

    def __init__(self, gis_map: Map):
        self.map = gis_map

    def draw(self, painter):
        # Draw the background
        painter.setOpacity(1.0)
        pen = qtg.QPen(Colors.black, 2, qtc.Qt.SolidLine)
        painter.setPen(pen)
        # brush = qtg.QBrush(qtg.QColor(0, 0, 0, 1.0))
        painter.setBrush(qtc.Qt.NoBrush)
        width = self.map.map_width - 2*self.map.buffer
        height = self.map.map_height - 2*self.map.buffer
        painter.drawRect(0 + self.map.buffer, 0 + self.map.buffer, width, height)


class MapGrid:

    def __init__(self, gis_map: Map):
        self.map = gis_map

    def draw(self, painter):
        path = qtg.QPainterPath()
        painter.setRenderHint(qtg.QPainter.Antialiasing)
        painter.setPen(Colors.lightgray)

        for i in np.arange(self.map.xmin, self.map.xmax + self.map.dx, self.map.dx):
            if self.map.xmin <= i <= self.map.xmax:
                point1 = Point(i, self.map.ymin, self.map)
                point2 = Point(i, self.map.ymax, self.map)
                path.moveTo(point1.x, point1.y)
                path.lineTo(point2.x, point2.y)
        for i in np.arange(self.map.ymin, self.map.ymax + self.map.dy, self.map.dy):
            if self.map.ymin <= i <= self.map.ymax:
                point1 = Point(self.map.xmin, i, self.map)
                point2 = Point(self.map.xmax, i, self.map)
                path.moveTo(point1.x, point1.y)
                path.lineTo(point2.x, point2.y)

        painter.drawPath(path)


class MapScale:

    def __init__(self, minval: float, maxval: float, step: float,
                mapx: float, mapy: float, units: float, height: float,
                gis_map: Map, colors=[Colors.black, Colors.white]):
        self.minval = minval
        self.maxval = maxval
        self.step = step
        self.mapx = mapx
        self.mapy = mapy
        self.units = units
        self.height = height
        self.map = gis_map
        self.colors = colors

    def draw(self, painter):
        ci = 0
        val = self.minval
        write_units = True
        for i in np.arange(self.mapx, self.mapx + (self.maxval - self.minval) + self.step, self.step):
            color = self.colors[ci % len(self.colors)]
            brush = qtg.QBrush(color)
            pen = qtg.QPen(Colors.black)
            painter.setBrush(brush)
            painter.setPen(pen)
            point = Point(i, self.mapy, self.map)
            dimensions = Dimensions(self.step, self.height, self.map)

            # Draw scale (don't draw last scale box)
            if i <= self.mapx + self.maxval - self.step:
                painter.drawRect(point.x, point.y - dimensions.height,
                                 dimensions.width, dimensions.height)

            # Draw tick marks
            brush = qtg.QBrush(Colors.black)
            pen = qtg.QPen(Colors.black)
            painter.setPen(pen)
            painter.setBrush(brush)
            if i <= self.mapx + self.maxval - self.step:
                # Note, this draws downward on the figure
                painter.drawRect(point.x, point.y, 0, 5)
            else:
                # Note, this draws downward on the figure
                painter.drawRect(point.x - 0.5, point.y, 0, 5)

            # Label units
            if write_units:
                font_size = 16
                painter.setFont(
                    qtg.QFont('Helvetica', font_size, qtg.QFont.Bold))
                painter.drawText(point.x - font_size *
                                 len(self.units)/2 - 0, point.y, self.units)
                write_units = False

            # Label all values, including right edge of last box
            painter.drawText(point.x, point.y + 20, str(val))

            # Increment counters
            val += self.step
            ci += 1


class MapWidget(qtw.QWidget):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.objects = []

    def add(self, obj):
        self.objects.append(obj)

    def paintEvent(self, paint_event):
        painter = qtg.QPainter(self)
        for obj in self.objects:
            obj.draw(painter)


class NorthArrow:

    def __init__(self, mapx: float, mapy: float, base_width: float, base_height: float,
                arrowhead_width: float, arrowhead_height: float, gis_map: Map,
                fill_color=Colors.white, line_color=Colors.black):
        self.mapx = mapx
        self.mapy = mapy
        self.base_width = base_width
        self.base_height = base_height
        self.arrowhead_width = arrowhead_width
        self.arrowhead_height = arrowhead_height
        self.map = gis_map
        self.fill_color = fill_color
        self.line_color = line_color

    def draw(self, painter):
        # Draw base
        brush = qtg.QBrush(self.fill_color)
        pen = qtg.QPen(self.line_color)
        painter.setBrush(brush)
        painter.setPen(pen)
        bottom_left_base_point = Point(self.mapx, self.mapy, self.map)
        # bottom_center_base_point = Point(self.mapx + self.base_width/2, self.mapy, self.map)
        top_left_base_point = Point(
            self.mapx, self.mapy + self.base_height, self.map)
        # top_center_base_point = Point(self.mapx + self.base_width/2, self.mapy + self.base_height, self.map)
        base_dimensions = Dimensions(
            self.base_width, self.base_height, self.map)
        painter.drawRect(top_left_base_point.x, top_left_base_point.y,
                         base_dimensions.width, base_dimensions.height)

        # Draw arrowhead
        arrowhead_dimensions = Dimensions(
            self.arrowhead_width, self.arrowhead_height, self.map)
        path = qtg.QPainterPath()
        left_point = Point(self.mapx + self.base_width/2 -
                           self.arrowhead_width/2, self.mapy + self.base_height, self.map)
        tip_point = Point(self.mapx + self.base_width/2, self.mapy +
                          self.base_height + self.arrowhead_height, self.map)
        right_point = Point(self.mapx + self.base_width/2 +
                            self.arrowhead_width/2, self.mapy + self.base_height, self.map)
        path.moveTo(left_point.x, left_point.y)
        path.lineTo(tip_point.x, tip_point.y)
        path.lineTo(right_point.x, right_point.y)
        path.lineTo(left_point.x, left_point.y)
        painter.drawPath(path)

        # Label arrow
        painter.setFont(qtg.QFont('Helvetica', 16, qtg.QFont.ExtraBold))
        label_base_point = Point(
            self.mapx, self.mapy + self.base_height + self.arrowhead_height/10, self.map)
        painter.drawText(label_base_point.x - 1, label_base_point.y, 'N')


class Dimensions:

    def __init__(self, width_in_map: float, height_in_map: float, gis_map: Map):
        scale_width = gis_map.map_width - 2*gis_map.buffer
        scale_height = gis_map.map_height - 2*gis_map.buffer
        xdist = gis_map.xmax - gis_map.xmin
        ydist = gis_map.ymax - gis_map.ymin
        self.width = scale_width/xdist * width_in_map
        self.height = scale_height/ydist * height_in_map


class Text:

    def __init__(self, x: float, y: float, text: str, gis_map: Map,
                color=Colors.black, font='Helvetica', font_size=24):
        self.x = x
        self.y = y
        self.text = text
        self.map = gis_map
        self.text_color = color
        self.font = font
        self.font_size = font_size

    def draw(self, painter):
        point = Point(self.x, self.y, self.map)
        painter.setPen(qtg.QPen(self.text_color, 1, qtc.Qt.SolidLine))
        painter.setFont(qtg.QFont(self.font, self.font_size))
        painter.drawText(point.x, point.y, self.text)


class ColorScale:

    def __init__(self, minval: float, maxval: float, step: float, mapx: float,
                mapy: float, scale_factor: float, units: str, height: float,
                gis_map: Map, colors):
        self.minval = minval
        self.maxval = maxval
        self.step = step
        self.mapx = mapx
        self.mapy = mapy
        self.scale_factor = scale_factor
        self.units = units
        self.height = height
        self.map = gis_map
        self.colors = colors

    def draw(self, painter):
        ci = 0
        val = self.minval
        write_units = True
        map_min = self.mapx
        map_step = self.step * self.scale_factor
        map_max = self.mapx + (self.maxval - self.minval) * self.scale_factor + map_step
        for i in np.arange(map_min, map_max, map_step):
            color = self.colors[ci % len(self.colors)]
            brush = qtg.QBrush(color)
            pen = qtg.QPen(Colors.black)
            painter.setBrush(brush)
            painter.setPen(pen)
            point = Point(i, self.mapy, self.map)
            dimensions = Dimensions(map_step, self.height, self.map)

            # Draw scale (don't draw last scale box)
            if i <= map_min + map_max - map_step:
                painter.drawRect(point.x, point.y - dimensions.height,
                                 dimensions.width, dimensions.height)

            # Draw tick marks
            brush = qtg.QBrush(Colors.black)
            pen = qtg.QPen(Colors.black)
            painter.setPen(pen)
            painter.setBrush(brush)
            if i <= map_min + map_max - map_step:
                # Note, this draws downward on the figure
                painter.drawRect(point.x, point.y, 0, 5)
            else:
                # Note, this draws downward on the figure
                painter.drawRect(point.x - 0.5, point.y, 0, 5)

            # Label units
            if write_units:
                font_size = 16
                painter.setFont(
                    qtg.QFont('Helvetica', font_size, qtg.QFont.Bold))
                painter.drawText(point.x - font_size *
                                 len(self.units)/2 - 0, point.y, self.units)
                write_units = False

            # Label all values, including right edge of last box
            painter.drawText(point.x, point.y + 20, str(val))

            # Increment counters
            val += self.step
            ci += 1


def savefig(widget, filename: str):
    widget.grab().save(filename)


def tolist(coord_dict: dict):
    coordinates = []
    for c in coord_dict:
        coordinates.append([c['lng'], c['lat']])
    return coordinates


def read_elements(infile: str, skiprows: int = 0):
    f = open(infile, 'r')
    lines = f.readlines()
    lines = lines[skiprows:]
    elements = []
    for line in lines:
        data = line.strip().split()
        data = list(map(int, data))
        elements.append(data)
    return elements


def read_nodes(infile: str, skiprows: int = 0):
    f = open(infile, 'r')
    lines = f.readlines()
    lines = lines[skiprows:]
    nodes = []
    for line in lines:
        data = line.strip().split()
        data = list(map(float, data))
        nodes.append(data)
    return nodes


def read_ras_elements(infile: str, skiprows: int = 0):
    f = open(infile, 'r')
    lines = f.readlines()
    lines = lines[skiprows:]
    elements = []
    for line in lines:
        data = line.strip().split()
        data = list(map(int, data[2:]))
        elements.append(data)
    return elements


def read_ras_nodes(infile: str, skiprows: int = 0):
    f = open(infile, 'r')
    lines = f.readlines()
    lines = lines[skiprows:]
    nodes = []
    for line in lines:
        data = line.strip().split()
        data = list(map(float, data[1:3]))
        nodes.append(data)
    return nodes


def read_values(infile: str):
    f = open(infile, 'r')
    lines = f.readlines()
    values = []
    for line in lines:
        values.append(float(line.strip()))
    return values

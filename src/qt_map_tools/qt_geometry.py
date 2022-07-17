from typing import Callable, Iterator, Union, Optional, List
from PyQt5 import QtGui
from PyQt5 import QtCore
from ..qt_map_tools import Colors, Map

class Cell:

    def __init__(self, name: str, boundary: List[float, ...], gis_map: Map,
                line_color=Colors.white, fill_color=Colors.blue, text_color=Colors.black, 
                label=False):
        self.name = name
        self.boundary = boundary
        self.map = gis_map
        self.fill_color = fill_color
        self.line_color = line_color
        self.text_color = text_color
        self.label = label

    def draw(self, painter):
        painter.setOpacity(1.0)
        path = QtGui.QPainterPath()
        pen = QtGui.QPen(self.line_color, 0)
        painter.setPen(pen)
        painter.setRenderHint(QtGui.QPainter.Antialiasing)
        if self.fill_color:
            brush = QtGui.QBrush(self.fill_color)
            painter.setBrush(brush)
        first_coord = self.boundary[0]
        first_point = Point(first_coord[0], first_coord[1], self.map)
        path.moveTo(first_point.x, first_point.y)

        for coord in self.boundary:
            point = Point(coord[0], coord[1], self.map)
            path.lineTo(point.x, point.y)
        path.lineTo(first_point.x, first_point.y)
        painter.drawPath(path)
        if self.label:
            mapx_min = min(self.boundary, key=lambda coord: coord[0])[0]
            mapx_max = max(self.boundary, key=lambda coord: coord[0])[0]
            mapy_min = min(self.boundary, key=lambda coord: coord[1])[1]
            mapy_max = max(self.boundary, key=lambda coord: coord[1])[1]
            # Offset by 0.6 degrees to approx. center of state
            mapx = (mapx_min + mapx_max)/2 - 0.6
            mapy = (mapy_min + mapy_max)/2
            point1 = Point(mapx, mapy, self.map)
            painter.setPen(self.text_color)
            painter.setFont(QtGui.QFont('Helvetica', 16, QtGui.QFont.ExtraBold))
            painter.drawText(point1.x, point1.y, self.name)
            
class Node:

    def __init__(self, name: str, center: list, gis_map: Map, fill_color=Colors.black,
                line_color=Colors.black, text_color=Colors.black, marker_size=2, 
                label=False):
        self.name = name
        self.center = center
        self.map = gis_map
        self.label = label
        self.fill_color = fill_color
        self.line_color = line_color
        self.text_color = text_color
        self.marker_size = marker_size

    def draw(self, painter):
        painter.setOpacity(1.0)
        point = Point(self.center[0], self.center[1], self.map)
        painter.setPen(QtGui.QPen(self.line_color, 0, QtCore.Qt.SolidLine))
        painter.setBrush(QtGui.QBrush(self.fill_color, QtCore.Qt.SolidPattern))
        painter.drawEllipse(
            point.x, point.y, self.marker_size, self.marker_size)
        # Label the node
        if self.label:
            painter.setPen(QtGui.QPen(self.text_color, 1, QtCore.Qt.SolidLine))
            painter.setFont(QtGui.QFont('Helvetica', 12))
            painter.drawText(point.x + 3, point.y - 3, self.name)
            
class Point:

    def __init__(self, mapx: float, mapy: float, gis_map: Map):
        self.map = gis_map
        xdist = gis_map.xmax - gis_map.xmin
        ydist = gis_map.ymax - gis_map.ymin
        scale_width = gis_map.map_width - 2*gis_map.buffer
        scale_height = gis_map.map_height - 2*gis_map.buffer
        self.x = scale_width/xdist * (mapx - gis_map.xmin) + gis_map.buffer
        self.y = gis_map.map_height - \
            (scale_height/ydist * (mapy - gis_map.ymin) + gis_map.buffer)

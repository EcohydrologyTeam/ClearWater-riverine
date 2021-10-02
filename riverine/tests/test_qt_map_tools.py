import unittest
import os
import sys
from PyQt5 import QtWidgets as qtw
import numpy as np

'''
Set paths to the qt_map_tools directory
'''

# Clearwater repo path
repo_path = os.path.dirname(
        os.path.dirname(
            os.path.dirname(
                os.path.realpath(__file__))))
print(f'ClearWater repo path: {repo_path}')  # ../../

# riverine/qt_map_tools path
src_path = os.path.join(repo_path, 'riverine', 'src')
tests_path = os.path.join(repo_path, 'riverine', 'tests')
print(f'Source path: {src_path}')  # ../../riverine/src/riverine
print(f'Tests path: {tests_path}')  # ../../riverine/src/riverine
sys.path.append(src_path)

from riverine.qt_map_tools import Colors, hex_to_rgb, rgb, get_color_index, get_color
from riverine.qt_map_tools import Point, Cell, Node
from riverine.qt_map_tools import Map, MapBorder, MapGrid, MapScale, MapWidget, Dimensions, Text, NorthArrow, ColorScale
from riverine.qt_map_tools import savefig, tolist, read_elements, read_nodes, read_ras_elements, read_ras_nodes, read_values

class Dragon(qtw.QMainWindow):

    def __init__(self):
        super().__init__()

        window_width = 500
        window_height = 500

        map_widget = MapWidget(self)
        self.setCentralWidget(map_widget)

        xmin = 0
        xmax = 1
        ymin = 0
        ymax = 1
        gis_map = Map(window_width, window_height, xmin=xmin,
                      xmax=xmax, ymin=ymin, ymax=ymax)
        map_widget.add(gis_map)

        map_grid = MapGrid(gis_map)
        map_widget.add(map_grid)

        # Sample mesh obtained from:
        # http://people.math.sc.edu/Burkardt/data/hex_mesh/hex_mesh.html

        elements = read_elements(os.path.join(tests_path, 'input_files', 'dragon_elements.txt'), skiprows=3)
        nodes = read_nodes(os.path.join(tests_path, 'input_files', 'dragon_nodes.txt'), skiprows=3)

        for i, node_points in enumerate(elements):
            boundary = []
            for n in node_points:
                boundary.append(nodes[n-1])
            color = Colors.default_colors[i % len(Colors.default_colors)]
            cell = Cell(str(i+1), boundary, gis_map,
                        line_color=Colors.black, fill_color=color, label=False)
            map_widget.add(cell)

        for i, coords in enumerate(nodes):
            node = Node(str(i+1), coords, gis_map, fill_color=Colors.black,
                        line_color=Colors.black, text_color=Colors.black, label=True)
            map_widget.add(node)

        map_border = MapBorder(gis_map)
        map_widget.add(map_border)

        map_widget.update()

        self.resize(window_width, window_height)
        self.show()
        savefig(map_widget, os.path.join(tests_path, 'input_files', 'dragon.png'))


class Lake(qtw.QMainWindow):

    def __init__(self):
        super().__init__()

        window_width = 800
        window_height = 800

        map_widget = MapWidget(self)
        self.setCentralWidget(map_widget)

        # scale_colors = [Colors.crimson, Colors.tomato, Colors.orange, Colors.gold, Colors.green]
        # scale_colors = [rgb(247, 252, 240), rgb(224, 243, 219), rgb(204, 235, 197), rgb(168, 221, 181), rgb(
        #     123, 204, 196), rgb(78, 179, 211), rgb(43, 140, 190), rgb(8, 104, 172), rgb(8, 64, 129)]
        # scale_colors = [rgb(255,247,251),rgb(236,226,240),rgb(208,209,230),rgb(166,189,219),rgb(103,169,207),rgb(54,144,192),rgb(2,129,138),rgb(1,108,89),rgb(1,70,54)]
        # s = ["#f44321","#5091cd","#f9a541","#7ac143"]
        # scale_colors = [hex_to_rgb(x) for x in s]
        # scale_colors = [rgb(0, 163, 226), rgb(27, 165, 72), rgb(253, 200, 0), rgb(241, 134, 14), rgb(228, 27, 19)]
        # s = ["#e6261f","#eb7532","#f7d038","#a3e048","#49da9a","#34bbe6","#4355db","#d23be7"]
        # s = ["#e6261f","#eb7532","#f7d038","#a3e048","#49da9a","#34bbe6"]
        # s = ["#f7fcf0","#e0f3db","#ccebc5","#a8ddb5","#7bccc4","#4eb3d3","#2b8cbe","#0868ac","#084081"]
        s = ["#f5542e", "#f2c327", "#008b6e", "#00aede", "#0067ad"]
        scale_colors = [hex_to_rgb(x) for x in s[::-1]]

        xmin = 0
        xmax = 800
        ymin = 100
        ymax = 900
        gis_map = Map(window_width, window_height, xmin=xmin, xmax=xmax, ymin=ymin,
                      ymax=ymax, dx=100, dy=100)
        map_widget.add(gis_map)

        map_grid = MapGrid(gis_map)
        map_widget.add(map_grid)

        # Sample mesh obtained from:
        # http://people.math.sc.edu/Burkardt/data/fem2d/fem2d.html

        elements = read_elements(os.path.join(tests_path, 'input_files', 'lake_elements.txt'))
        nodes = read_nodes(os.path.join(tests_path, 'input_files', 'lake_nodes.txt'))
        values = read_values(os.path.join(tests_path, 'input_files', 'lake_values.txt'))
        print('min: ', min(values))
        print('max: ', max(values))

        for i, node_points in enumerate(elements):
            boundary = []

            p = node_points[0]
            v = values[p-1]
            # idx = int(v * 20)
            idx = get_color_index(v, -2, 3, len(scale_colors))

            for n in node_points:
                boundary.append(nodes[n-1])
            color = scale_colors[idx % len(scale_colors)]
            cell = Cell(str(i+1), boundary, gis_map, line_color=Colors.lightgray,
                        fill_color=color, text_color=Colors.black, label=False)
            map_widget.add(cell)

        for i, coords in enumerate(nodes):
            node = Node(str(i+1), coords, gis_map, fill_color=Colors.orange,
                        line_color=Colors.orange, label=False)
            map_widget.add(node)

        map_scale = MapScale(0, 200, 50, 550, 130,
                             'meters', 10, gis_map)
        map_widget.add(map_scale)

        north_arrow = NorthArrow(
            750, 800, 10, 50, 30, 30, gis_map, fill_color=Colors.white, line_color=Colors.black)
        map_widget.add(north_arrow)

        map_border = MapBorder(gis_map)
        map_widget.add(map_border)

        map_widget.update()

        self.resize(window_width, window_height)
        self.show()
        savefig(map_widget, os.path.join(tests_path, 'inputs', 'lake.png'))

class Channel(qtw.QMainWindow):

    def __init__(self):
        super().__init__()

        window_width = 800
        window_height = 800

        map_widget = MapWidget(self)
        self.setCentralWidget(map_widget)

        xmin = -5
        xmax = 15
        ymin = -5
        ymax = 5
        gis_map = Map(window_width, window_height, xmin=xmin, xmax=xmax, ymin=ymin,
                      ymax=ymax, dx=1, dy=1)
        map_widget.add(gis_map)

        map_grid = MapGrid(gis_map)
        map_widget.add(map_grid)

        # Sample mesh obtained from:
        # http://people.math.sc.edu/Burkardt/data/fem2d/fem2d.html

        elements = read_elements(os.path.join(tests_path,'input_files', 'channel_elements.txt'))
        nodes = read_nodes(os.path.join(tests_path,'input_files','channel_nodes.txt'))

        for i, node_points in enumerate(elements):
            boundary = []
            for n in node_points:
                boundary.append(nodes[n-1])
            color = Colors.default_colors[i % len(Colors.default_colors)]
            cell = Cell(str(i+1), boundary, gis_map, line_color=Colors.lightgray,
                        fill_color=color, text_color=Colors.black, label=False)
            map_widget.add(cell)

        for i, coords in enumerate(nodes):
            node = Node(str(i+1), coords, gis_map, fill_color=Colors.orange,
                        line_color=Colors.orange, label=False)
            map_widget.add(node)

        map_border = MapBorder(gis_map)
        map_widget.add(map_border)

        map_widget.update()

        self.resize(window_width, window_height)
        self.show()
        savefig(map_widget, os.path.join(tests_path, 'inputs', 'channel.png'))


class Greenland(qtw.QMainWindow):

    def __init__(self):
        super().__init__()

        window_width = 1000
        window_height = 1000

        map_widget = MapWidget(self)
        self.setCentralWidget(map_widget)

        xmin = -50
        xmax = 350
        ymin = 0
        ymax = 550
        gis_map = Map(window_width, window_height, xmin=xmin, xmax=xmax, ymin=ymin,
                      ymax=ymax, dx=100, dy=100)
        map_widget.add(gis_map)

        map_grid = MapGrid(gis_map)
        map_widget.add(map_grid)

        # Sample mesh obtained from:
        # http://people.math.sc.edu/Burkardt/data/fem2d/fem2d.html

        elements = read_elements(os.path.join(tests_path,'input_files','greenland_elements.txt'))
        nodes = read_nodes(os.path.join(tests_path,'input_files','greenland_nodes.txt'))

        for i, node_points in enumerate(elements):
            boundary = []
            for n in node_points:
                boundary.append(nodes[n-1])
            # color = colors[i % len(colors)]
            color = Colors.lightseagreen
            cell = Cell(str(i+1), boundary, gis_map, line_color=Colors.blue,
                        fill_color=color, text_color=Colors.black, label=False)
            map_widget.add(cell)

        # for i, coords in enumerate(nodes):
        #     node = Node(str(i+1), coords, gis_map, fill_color=Colors.orange, line_color=Colors.orange, label=False)
        #     map_widget.add(node)

        map_border = MapBorder(gis_map)
        map_widget.add(map_border)

        map_widget.update()

        self.resize(window_width, window_height)
        self.show()
        savefig(map_widget, os.path.join(tests_path, 'inputs', 'greenland.png'))


class BigCavity(qtw.QMainWindow):

    def __init__(self):
        super().__init__()

        window_width = 800
        window_height = 800

        map_widget = MapWidget(self)
        self.setCentralWidget(map_widget)

        xmin = 0
        xmax = 1
        ymin = 0
        ymax = 1
        gis_map = Map(window_width, window_height, xmin=xmin, xmax=xmax, ymin=ymin,
                      ymax=ymax, dx=0.1, dy=0.1)
        map_widget.add(gis_map)

        map_grid = MapGrid(gis_map)
        map_widget.add(map_grid)

        # Sample mesh obtained from:
        # http://people.math.sc.edu/Burkardt/data/fem2d/fem2d.html

        elements = read_elements(os.path.join(tests_path,'input_files','big_cavity_elements.txt'))
        nodes = read_nodes(os.path.join(tests_path,'input_files','big_cavity_nodes.txt'))

        for i, node_points in enumerate(elements):
            boundary = []
            for n in node_points:
                boundary.append(nodes[n-1])
            color = Colors.default_colors[i % len(Colors.default_colors)]
            color = None
            cell = Cell(str(i+1), boundary, gis_map, line_color=Colors.lightgray,
                        fill_color=color, text_color=Colors.black, label=False)
            map_widget.add(cell)

        for i, coords in enumerate(nodes):
            node = Node(str(i+1), coords, gis_map, fill_color=Colors.orange,
                        line_color=Colors.orange, label=False)
            map_widget.add(node)

        map_border = MapBorder(gis_map)
        map_widget.add(map_border)

        map_widget.update()

        self.resize(window_width, window_height)
        self.show()
        savefig(map_widget, os.path.join(tests_path, 'inputs', 'big_cavity.png'))


class Web(qtw.QMainWindow):

    def __init__(self):
        super().__init__()

        window_width = 800
        window_height = 800

        map_widget = MapWidget(self)
        self.setCentralWidget(map_widget)

        xmin = -5
        xmax = 5
        ymin = -1
        ymax = 6
        gis_map = Map(window_width, window_height, xmin=xmin, xmax=xmax, ymin=ymin,
                      ymax=ymax, dx=1, dy=1)
        map_widget.add(gis_map)

        map_grid = MapGrid(gis_map)
        map_widget.add(map_grid)

        # Sample mesh obtained from:
        # http://people.math.sc.edu/Burkardt/data/fem2d/fem2d.html

        elements_file = os.path.join(tests_path, 'input_files', 'web_elements.txt')
        print(f'elements_file = {elements_file}')
        elements = read_elements(elements_file, skiprows=2)
        nodes = read_nodes(os.path.join(tests_path, 'input_files', 'web_nodes.txt'), skiprows=4)

        for i, node_points in enumerate(elements):
            boundary = []
            for n in node_points:
                boundary.append(nodes[n-1])
            color = Colors.default_colors[i % len(Colors.default_colors)]
            cell = Cell(str(i+1), boundary, gis_map, line_color=Colors.lightgray,
                        fill_color=color, text_color=Colors.black, label=False)
            map_widget.add(cell)

        for i, coords in enumerate(nodes):
            node = Node(str(i+1), coords, gis_map, fill_color=Colors.orange,
                        line_color=Colors.orange, label=True)
            map_widget.add(node)

        map_border = MapBorder(gis_map)
        map_widget.add(map_border)

        map_widget.update()

        self.resize(window_width, window_height)
        self.show()
        savefig(map_widget, os.path.join(tests_path, 'inputs', 'web.png'))


class RAS2D_ADCIRC(qtw.QMainWindow):

    def __init__(self):
        super().__init__()

        window_width = 1500
        window_height = 1000

        map_widget = MapWidget(self)
        self.setCentralWidget(map_widget)

        xmin = 404500
        xmax = 412500
        ymin = 1800500
        ymax = 1805500
        gis_map = Map(window_width, window_height, xmin=xmin, xmax=xmax, ymin=ymin,
                      ymax=ymax, dx=500, dy=500)
        map_widget.add(gis_map)

        map_grid = MapGrid(gis_map)
        map_widget.add(map_grid)

        # NOTE: USING SPECIAL FUNCTIONS HERE FOR RAS
        elements = read_ras_elements(os.path.join(tests_path, 'inputs', '2D_Interior_Area_TIN_elements.txt'))
        nodes = read_ras_nodes(os.path.join(tests_path, 'inputs', '2D_Interior_Area_TIN_nodes.txt'))

        for i, node_points in enumerate(elements):
            boundary = []
            for n in node_points:
                boundary.append(nodes[n-1])
            # color = colors[i % len(colors)]
            color = Colors.lightseagreen
            cell = Cell(str(i+1), boundary, gis_map, line_color=Colors.black,
                        fill_color=color, text_color=Colors.black, label=False)
            map_widget.add(cell)

        # for i, coords in enumerate(nodes):
        #     node = Node(str(i+1), coords, gis_map, fill_color=Colors.white,
        #                 line_color=Colors.white, marker_size=1, label=False)
        #     map_widget.add(node)

        map_scale = MapScale(0, 1000, 250, 411250, 1800700,
                             'meters', 50, gis_map)
        map_widget.add(map_scale)

        north_arrow = NorthArrow(
            412250, 1804750, 50, 400, 150, 150, gis_map, fill_color=Colors.white, line_color=Colors.black)
        map_widget.add(north_arrow)

        map_border = MapBorder(gis_map)
        map_widget.add(map_border)

        map_widget.update()

        self.resize(window_width, window_height)
        self.show()
        savefig(map_widget, os.path.join(tests_path, 'inputs', '2D_Interior_Area_TIN.png'))


class RAS2D_HDF(qtw.QMainWindow):

    def __init__(self, timestep):
        super().__init__()

        window_width = 1500
        window_height = 1000

        map_widget = MapWidget(self)
        self.setCentralWidget(map_widget)

        xmin = 404500
        xmax = 412500
        ymin = 1800500
        ymax = 1805500
        gis_map = Map(window_width, window_height, xmin=xmin, xmax=xmax, ymin=ymin,
                      ymax=ymax, dx=500, dy=500)
        map_widget.add(gis_map)

        map_grid = MapGrid(gis_map)
        map_widget.add(map_grid)

        # NOTE: USING SPECIAL FUNCTIONS HERE FOR RAS
        import h5py
        f = h5py.File(os.path.join(tests_path, 'input_files', 'Muncie.p04.hdf'), 'r')
        # max value: 5773, shape(5765, 7)
        elements_array = f['Geometry/2D Flow Areas/2D Interior Area/Cells FacePoint Indexes'][()]
        # shape(5774, 2)
        nodes_array = f['Geometry/2D Flow Areas/2D Interior Area/FacePoints Coordinate'][()]

        depth = f['Results/Unsteady/Output/Output Blocks/Base Output/Unsteady Time Series/2D Flow Areas/2D Interior Area/Depth'][()
                                                                                                                                 ][timestep]
        node_x_vel = f['Results/Unsteady/Output/Output Blocks/Base Output/Unsteady Time Series/2D Flow Areas/2D Interior Area/Node X Vel'][()
                                                                                                                                           ][timestep]
        node_y_vel = f['Results/Unsteady/Output/Output Blocks/Base Output/Unsteady Time Series/2D Flow Areas/2D Interior Area/Node Y Vel'][()
                                                                                                                                           ][timestep]
        node_vel = np.sqrt(node_x_vel**2 + node_y_vel**2)

        s = ["#f5542e", "#f2c327", "#008b6e", "#00aede", "#0067ad"]
        scale_colors = [hex_to_rgb(x) for x in s[::-1]]

        elements = []
        nodes = []
        for i in range(len(elements_array)):
            elements.append(list(elements_array[i]))
        for i in range(len(nodes_array)):
            nodes.append(list(nodes_array[i]))

        for i, node_points in enumerate(elements):
            boundary = []
            for n in node_points:
                if n > -1:
                    boundary.append(nodes[n])
            # color = colors[i % len(colors)]
            # color = lightseagreen

            # Color cells by depth at specified time step, range 0 - 20
            value = depth[i]
            # ci = get_color_index(value, 0, 20, len(scale_colors))
            # color = scale_colors[ci]
            color = get_color(value, 0, 20, scale_colors)

            # Color cells by velocity magnitude at specified time step, range 0.0 - 3.5
            # value = node_vel[i]
            # color = get_color(value, 0, 2.5, scale_colors)

            cell = Cell(str(i+1), boundary, gis_map, line_color=Colors.navy,
                        fill_color=color, text_color=Colors.black, label=False)
            map_widget.add(cell)

        # for i, coords in enumerate(nodes):
        #     node = Node(str(i+1), coords, gis_map, fill_color=Colors.white,
        #                 line_color=Colors.white, marker_size=1, label=False)
        #     map_widget.add(node)

        color_scale = ColorScale(0, 20, 5, 405000, 1800700, 100, 'depth (ft)', 75, gis_map, scale_colors)
        map_widget.add(color_scale)

        # map_scale_colors = [Colors.crimson, Colors.tomato, Colors.orange, Colors.gold]
        map_scale_colors = [Colors.black, Colors.white]
        map_scale = MapScale(0, 1000, 250, 411250, 1800700,
                             'dist (ft)', 50, gis_map, colors=map_scale_colors)
        map_widget.add(map_scale)

        north_arrow = NorthArrow(
            412250, 1804750, 50, 400, 150, 150, gis_map, fill_color=Colors.white, line_color=Colors.navy)
        map_widget.add(north_arrow)

        # timestep_label = Text(404520, 1805490, 'Time step: %03d' % (timestep + 1))
        timestep_label = Text(
            404600, 1805300, 'Time step: %03d' % (timestep + 1), gis_map)
        map_widget.add(timestep_label)

        map_border = MapBorder(gis_map)
        map_widget.add(map_border)

        map_widget.update()

        self.resize(window_width, window_height)
        self.show()
        outfilename = f'2D_Interior_Area_HDF_depth_{timestep:.3f}.png'
        savefig(map_widget, os.path.join(tests_path, 'inputs', outfilename))


class Test_qt_map_tools(unittest.TestCase):
    def setUp(self):
        pass

    def test_Dragon(self):
        print('test_Dragon')
        app = qtw.QApplication(sys.argv)
        mesh = Dragon()
        sys.exit(app.exec())

    def test_Lake(self):
        print('test_Lake')
        app = qtw.QApplication(sys.argv)
        mesh = Lake()
        sys.exit(app.exec())

    def test_Channel(self):
        print('test_Channel')
        app = qtw.QApplication(sys.argv)
        mesh = Channel()
        sys.exit(app.exec())

    def test_Greenland(self):
        print('test_Greenland')
        app = qtw.QApplication(sys.argv)
        mesh = Greenland()
        sys.exit(app.exec())

    def test_BigCavity(self):
        print('test_BigCavity')
        app = qtw.QApplication(sys.argv)
        mesh = BigCavity()
        sys.exit(app.exec())

    def test_Web(self):
        print('test_Web')
        app = qtw.QApplication(sys.argv)
        mesh = Web()
        sys.exit(app.exec())

    def test_RAS2d_ADCIRC(self):
        print('test_RAS2d_ADCIRC')
        app = qtw.QApplication(sys.argv)
        mesh = RAS2D_ADCIRC()
        sys.exit(app.exec())

    def test_RAS2D_HDF(self):
        print('test_RAS2D_HDF')
        app = qtw.QApplication(sys.argv)

        for i in range(30, 175):
            mesh = RAS2D_HDF(i)
            mesh.close()

        sys.exit(app.exec())


if __name__ == '__main__':
    unittest.main()

from collections import OrderedDict
import numpy
import geopandas
import pandas
from geopandas import GeoSeries
from shapely.geometry import Polygon
import matplotlib.pyplot as plt
import matplotlib
import contextily as ctx
import h5py
import datetime
import os
import sys
import numba

local_path = os.path.dirname(os.path.realpath(__file__))
repo_path = os.path.join(local_path.split('ClearWater')[0], 'ClearWater')
module_path = os.path.join(repo_path, 'modules', 'python', 'src')
riverine_path = os.path.join(repo_path, 'riverine', 'src')
print(f'module_path = {module_path}')
sys.path.insert(0, module_path)
sys.path.insert(0, riverine_path)

from TSM import TSM
from riverine.stations import MetStation

def make_colormap(colors: list, position: list = None, eight_bit: bool = False):
    '''
    Create a matplotlib colormap.

    Parmeters:
        colors (list):      A list of tuples that contain RGB values. These may be specified
                            in 8-bit format (0 - 255) or floating point (0.0 - 1.0) values.
                            The tuples must be arraned to that the first color is the lowest 
                            value for the colormap and colorbar and the last value is the highest.
        position (list):    Contains values from 0 to 1 to indicate the location of each color.
        eight_bit (bool):   If True, the 8-bit format is used. If False, the floating point 
                            format is used.
    Returns:
        A matplotlib colormap with the colors spaced at regular intervals.
    '''

    bit_rgb = numpy.linspace(0, 1, 256)

    if position == None:
        position = numpy.linspace(0, 1, len(colors))
    else:
        if len(position) != len(colors):
            sys.exit("position length must be the same as colors")
        elif position[0] != 0 or position[-1] != 1:
            sys.exit("position must start with 0 and end with 1")

    if eight_bit:
        for i in range(len(colors)):
            colors[i] = (bit_rgb[colors[i][0]],
                         bit_rgb[colors[i][1]],
                         bit_rgb[colors[i][2]])

    cdict = {'red':[], 'green':[], 'blue':[]}

    for pos, color in zip(position, colors):
        cdict['red'].append((pos, color[0], color[0]))
        cdict['green'].append((pos, color[1], color[1]))
        cdict['blue'].append((pos, color[2], color[2]))

    cmap = matplotlib.colors.LinearSegmentedColormap('my_colormap', cdict, 256)
    return cmap

class Container:
    pass

class RAS_HDF5:
    '''
    Read HEC-RAS 2D geometry and variables and return as a dictionary
    '''

    def __init__(self, hdf5_file_path: str, variables: list = []):

        self.variables = {}
        self.hdf5_file_path = hdf5_file_path
        self.results = {}
        self.geometry = {}

    def read(self):
        with h5py.File(self.hdf5_file_path, 'r') as infile:
            '''
            Read the Geometry data
            '''
            # For the Muncie data set: max value: 5773, shape(5765, 7)
            self.geometry['elements_array'] = infile['Geometry/2D Flow Areas/2D Interior Area/Cells FacePoint Indexes'][()]
            # For the Muncie data set: shape(5774, 2)
            self.geometry['nodes_array'] = infile['Geometry/2D Flow Areas/2D Interior Area/FacePoints Coordinate'][()]
            self.geometry['faces_cell_indexes'] = infile['Geometry/2D Flow Areas/2D Interior Area/Faces Cell Indexes'][()]
            self.geometry['cells_surface_area'] = infile['Geometry/2D Flow Areas/2D Interior Area/Cells Surface Area'][()]
            self.geometry['faces_normal_unit_vector_and_length'] = infile['Geometry/2D Flow Areas/2D Interior Area/Faces NormalUnitVector and Length'][()]
            self.geometry['cells_center_coordinate'] = infile['Geometry/2D Flow Areas/2D Interior Area/Cells Center Coordinate'][()]
            # faces_area_elevation_values = infile['Geometry/2D Flow Areas/2D Interior Area/Faces Area Elevation Values'][()]

            self.geometry['face_length'] = self.geometry['faces_normal_unit_vector_and_length'][:,2]

            '''
            Read the Results data
            '''
            self.results['depth'] = infile['Results/Unsteady/Output/Output Blocks/Base Output/Unsteady Time Series/2D Flow Areas/2D Interior Area/Depth'][()]
            self.results['node_x_velocity'] = infile['Results/Unsteady/Output/Output Blocks/Base Output/Unsteady Time Series/2D Flow Areas/2D Interior Area/Node X Vel'][()]
            self.results['node_y_velocity'] = infile['Results/Unsteady/Output/Output Blocks/Base Output/Unsteady Time Series/2D Flow Areas/2D Interior Area/Node Y Vel'][()]
            self.results['face_velocity'] = infile['Results/Unsteady/Output/Output Blocks/Base Output/Unsteady Time Series/2D Flow Areas/2D Interior Area/Face Velocity'][()]
            self.results['face_q'] = infile['Results/Unsteady/Output/Output Blocks/Base Output/Unsteady Time Series/2D Flow Areas/2D Interior Area/Face Q'][()]
            self.results['node_speed'] = numpy.sqrt(self.results['node_x_velocity']**2 + self.results['node_y_velocity']**2)
            time_stamps_binary = infile['Results/Unsteady/Output/Output Blocks/Base Output/Unsteady Time Series/Time Date Stamp'][()]

            # Read the specified variables, if any
            for variable in self.variables:
                data_path = f'Results/Unsteady/Output/Output Blocks/Base Output/Unsteady Time Series/2D Flow Areas/2D Interior Area/{variable}'
                self.results['variable'] = infile[data_path]

        # Convert from binary strings to utf8 strings
        time_stamps = [x.decode("utf8") for x in time_stamps_binary]
        self.results['dates'] = [datetime.datetime.strptime(x, '%d%b%Y %H:%M:%S') for x in time_stamps] # '02JAN1900 22:55:00'

        # Convert all lists to numpy arrays
        for key, value in self.geometry.items():
            self.geometry[key] = numpy.array(value)
        for key, value in self.results.items():
            self.results[key] = numpy.array(value)


def plot_ras2d(elements_array: list, nodes_array: list, variable: str, dates: list, outfile_interp_path: str,
    xmin: float = None, xmax: float = None, ymin: float = None, ymax: float = None, 
    vmin: float = None, vmax: float = None, variable_name: str = 'Depths', 
    xlabel: str = "", ylabel: str = "", clabel: str = "", crs: str = "epsg:4326", 
    figsize: tuple = (10,6), cmap: matplotlib.colors.LinearSegmentedColormap = None,
    alpha: float = 1.0, scheme: str = None, start_index: int = None, end_index: int = None):
    '''
    plot_ras2d takes lists of elements, nodes, variables, and timestamps 
    read from an HEC-RAS 2D HDF5 file and plots a time series of maps.
    '''

    # Assemble the geometry
    elements = []
    nodes = []
    polygons = []
    points_to_plot = [] # keep track of which indexes that have elements that contain at least three sides. Why do those exist???

    for i in range(len(elements_array)):
        elements.append(tuple(elements_array[i]))
    for i in range(len(nodes_array)):
        nodes.append(tuple(nodes_array[i]))

    for i, node_points in enumerate(elements):
        boundary = []
        for n in node_points:
            if n > -1:
                boundary.append(nodes[n])

        if len(boundary) > 3:
            polygons.append(Polygon(boundary))
            points_to_plot.append(i)

    cells = GeoSeries(polygons)

    # Remove the values from the plot variable that correspond to elements that have less than three sides.
    variable_filtered = variable[:, points_to_plot]

    # Plot the data
    if start_index is None:
        iteration_list = range(len(variable_filtered))
    else:
        iteration_list = range(start_index, end_index)

    images = []

    for i in iteration_list:
        outfile = outfile_interp_path % i
        variable_to_plot = variable_filtered[i,:]
        df = geopandas.GeoDataFrame({variable_name: variable_to_plot}, geometry=cells, crs=crs)

        # Plot the data
        # Other options include antialiased=False and edgecolor="face"
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=figsize)

        if scheme is None:
            legend_kwds={'label':clabel}
        else:
            legend_kwds=None

        if cmap is not None:
            my_plot = df.plot(ax=ax, column=variable_name, vmin=vmin, vmax=vmax, legend=True, legend_kwds=legend_kwds, edgecolor="black", linewidth=0.1, cmap=cmap, alpha=alpha, scheme=scheme)
        else:
            my_plot = df.plot(ax=ax, column=variable_name, vmin=vmin, vmax=vmax, legend=True, legend_kwds=legend_kwds, edgecolor="black", linewidth=0.1, alpha=alpha, scheme=scheme)

        # Customize the plot
        ax.axis('equal')
        ax.grid(True, alpha=0.2, linewidth=0.1)
        if xmin is not None and xmax is not None:
            ax.set_xlim([xmin, xmax])
        if ymin is not None and ymax is not None:
            ax.set_ylim([ymin, ymax])
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        title_string = dates[i].strftime('%d%b%Y %H:%M')
        ax.set_title(title_string)

        # https://stackoverflow.com/questions/56559520/change-background-map-for-contextily
        # ctx.add_basemap(ax, source=ctx.providers.OpenStreetMap.Mapnik)
        ctx.add_basemap(ax, crs='epsg:2245', source=ctx.providers.OpenTopoMap)
        # final_plot = ctx.add_basemap(ax, crs=crs, source=ctx.providers.OpenTopoMap)
        images.append(my_plot)

        # Save the map to a PNG file
        plt.savefig(outfile, dpi=300)
        # plt.close()
    # plt.show()
    return images


class WQ_Cell:

    def __init__(self, initial_conc: float = 0.0, volume: float = None,
                area: float = 0.0, depth: float = 0.0, diffusion_coef: float = 0.2):
        self.conc = initial_conc
        self.diffusion_coef = diffusion_coef
        self.depth = depth
        self.area = area
        if volume is None:
            self.volume = self.area * self.depth
        else:
            self.volume = volume
        self.mass = self.conc * self.volume

    def add_mass(self, dm: float):
        self.mass += dm
        if self.volume > 0.0:
            self.conc = self.mass / self.volume
        else:
            self.conc = 0.0

class Temperature_Cell(WQ_Cell):
    def __init__(self, initial_TwaterC: float = 0.0, volume: float = None,
                area: float = 0.0, depth: float = 0.0, diffusion_coef: float = 0.2):
        super().__init__(initial_conc=initial_TwaterC, volume=volume, area=area, depth=depth, diffusion_coef=diffusion_coef)

def initialize_cells(ncells: int, initial_conc: float, cell_surface_area: list, initial_depth=0.0, diffusion_coef=0.1) -> list:
    cells = [WQ_Cell(initial_conc=initial_conc, area=cell_surface_area[i], depth=0.0, diffusion_coef=diffusion_coef) for i in range(ncells)]
    return cells

# def initialize_cells(ncells: int, initial_conc: float, cell_surface_area: list, initial_depth=0.0, diffusion_coef=0.1) -> pandas.DataFrame:
#     cells = numpy.arange(ncells, dtype=numpy.int64)
#     df_cells = pandas.DataFrame(cells, columns=['cells'])
#     df_cells['TwaterC'] = initial_conc
#     df_cells['surface_area'] = cell_surface_area
#     df_cells['depth'] = initial_depth
#     df_cells['diffusion_coeff'] = diffusion_coef
#     df_cells['mass'] = 0.0
#     df_cells = df_cells.set_index('cells')
#     return df_cells

# @numba.njit
def compute_cells(ti: int, cell_surface_area: numpy.array, depths: numpy.array, TwaterC: float, TairC: float, q_solar: float, pressure_mb: float, eair_mb: float, cloudiness: float, wind_speed: float, wind_a: float, wind_b: float, wind_c: float, wind_kh_kw: float):
    ncells = len(cell_surface_area)
    for i in range(ncells):
        surface_area = cell_surface_area[i]
        depth = depths[ti,i]
        volume = surface_area * depth
        TSM.energy_budget_method(TwaterC, surface_area, volume, TairC, q_solar, pressure_mb, eair_mb, cloudiness, wind_speed, wind_a, wind_b, wind_c, wind_kh_kw)

# def compute_cells(cells, TwaterC, TairC, q_solar, pressure_mb, eair_mb, cloudiness, wind_speed, wind_a, wind_b, wind_c, wind_kh_kw):
#     # Compute temperature
#     for cell in cells:
#         cell.add_mass(0.1 * cell.area + 0.1)
#         TSM2.energy_budget_method(TwaterC, cell.area, cell.volume, TairC, q_solar, pressure_mb, eair_mb, cloudiness, wind_speed, wind_a, wind_b, wind_c, wind_kh_kw)
# 
# def compute_cells(df_cells, TwaterC, TairC, q_solar, pressure_mb, eair_mb, cloudiness, wind_speed, wind_a, wind_b, wind_c, wind_kh_kw):
#     # Compute temperature
#     for i in range(len(df_cells)):
#         df_cells['mass'].iloc[i] += 0.1 # FAKE MASS. TODO: ADD REAL CALCULATION
#         # TSM2.energy_budget_method(TwaterC, cell.area, cell.volume, TairC, q_solar, pressure_mb, eair_mb, cloudiness, wind_speed, wind_a, wind_b, wind_c, wind_kh_kw)

def wq_simulation(ras_file_path: str, initial_conc: float, met_station: MetStation, diffusion_coef: float = 0.2, dt: float = 300.0,
                    time_delta: datetime.timedelta = None):
    '''
    Simulate water quality, with advection-diffusion

    Parameters:
        ras_file_path (str): Full path to HEC-RAS 2D HDF5 file
                             that contains hydraulic and geometry
                             outputs
        intial_conc (float): Initial concentration or temperature
        diffusion_coef (float): Diffusion coefficient (m2/s)
        dt: Time step (seconds)
    '''

    # Read HEC-RAS 2D output file
    print('Reading data...')
    ras: RAS_HDF5 = RAS_HDF5(ras_file_path, variables=[])
    ras.read()
    print('Finished reading')

    # Get variables needed for the WQ simulation
    dates = ras.results['dates']
    face_velocity = ras.results['face_velocity']
    face_cell_indexes = ras.geometry['faces_cell_indexes']
    face_length = ras.geometry['face_length']
    cell_surface_area = ras.geometry['cells_surface_area']
    depths = ras.results['depth']

    # Adjust the dates. This may be needed if the year is set to 1900, for example.
    if time_delta is not None:
        dates = [d + time_delta for d in dates]

    # df = pd.DataFrame(faces_cell_indexes, columns=['first', 'second'])
    # df['face_velocity_250'] = face_velocity[250,:]
    # print(df.head())

    ncells = len(cell_surface_area)
    nfaces = len(face_length)
    ntimes = depths.shape[0]
    print(f'ncells = {ncells}')
    print(f'nfaces = {nfaces}')

    # Set initial temperature
    TwaterC = initial_conc # Initial temperature

    # Initialize cells
    print('Initializing cells...')
    # df_cells = initialize_cells(ncells, initial_conc, cell_surface_area, initial_depth=0.0, diffusion_coef=0.1)
    cells = initialize_cells(ncells, initial_conc, cell_surface_area, initial_depth=0.0, diffusion_coef=0.1)
    cells = numpy.array(cells)

    '''
    WQ Simulation loop

    Advection-Diffusion Equation:
    source = advection + diffusion

    '''

    # Iterate through times
    print('Computing WQ...')

    masses = depths*0.0

    for ti in range(0, ntimes):
        date_time = dates[ti]
        print(f'Step {ti}, Date: {date_time}')
        date_time_str = date_time.strftime('%Y-%m-%d %H:%M:%S')
        TairC = met_station.interpolate_variable('Tair', date_time_str)
        q_solar = met_station.interpolate_variable('Solar Radiation', date_time_str)
        wind_speed = met_station.interpolate_variable('Wind Speed', date_time_str)
        cloudiness = met_station.interpolate_variable('Cloudiness', date_time_str)
        pressure_mb = met_station.interpolate_variable('Air Pressure', date_time_str)
        eair_mb = met_station.interpolate_variable('Vapor Pressure', date_time_str)

        # Iterate through faces
        # for fi in range(nfaces):
        #     left_cell_index = face_cell_indexes[fi,0]
        #     right_cell_index = face_cell_indexes[fi,1]
        #     lcell = cells[left_cell_index]
        #     rcell = cells[right_cell_index]
        #     lcell.depth = depths[ti,left_cell_index]
        #     rcell.depth = depths[ti,right_cell_index]
        #     # Compute advective flux through cell face (dm = velocity * area * conc * time) [L/T * L2 * M/L3 * T = M]
        #     lflux = face_velocity[ti,fi] * face_length[fi] * lcell.depth * lcell.conc * dt
        #     rflux = face_velocity[ti,fi] * face_length[fi] * rcell.depth * rcell.conc * dt
        #     lcell.add_mass(lflux)
        #     rcell.add_mass(rflux)
        #     # Compute diffusive flux through cell face (dm = velocity * area * conc * time) [L/T * L2 * M/L3 * T = M]
        #     lflux = face_velocity[ti,fi] * face_length[fi] * lcell.depth * lcell.conc * dt * diffusion_coef
        #     rflux = face_velocity[ti,fi] * face_length[fi] * rcell.depth * rcell.conc * dt * diffusion_coef
        #     lcell.add_mass(lflux)

        ldepths = depths[ti, face_cell_indexes[:,0]]
        rdepths = depths[ti, face_cell_indexes[:,1]]
        lmasses = masses[ti, face_cell_indexes[:,0]]
        rmasses = masses[ti, face_cell_indexes[:,1]]

        # Compute advective flux through cell face (dm = velocity * area * conc * time) [L/T * L2 * M/L3 * T = M]
        lfluxes = face_velocity[ti,:] * face_length * ldepths * lmasses * dt
        rfluxes = face_velocity[ti,:] * face_length * rdepths * rmasses * dt
        lmasses += lfluxes
        rmasses += rfluxes

        # Compute diffusive flux through cell face (dm = velocity * area * conc * time) [L/T * L2 * M/L3 * T = M]
        lfluxes = face_velocity[ti,:] * face_length * ldepths * lmasses * dt * diffusion_coef
        rfluxes = face_velocity[ti,:] * face_length * rdepths * rmasses * dt * diffusion_coef
        lmasses += lfluxes
        rmasses += rfluxes

        # Compute temperature
        compute_cells(ti, cell_surface_area, depths, TwaterC, TairC, q_solar, pressure_mb, eair_mb, cloudiness, wind_speed, met_station.wind_a, met_station.wind_b, met_station.wind_c, met_station.wind_kh_kw)

        '''
        # Plot cell values for current time index here
        if ti == 250:
            concs = []
            for i, cell in cells.items():
                concs.append(cell.conc)
            plt.plot(concs)
            plt.show()
        '''
import h5py
import numpy as np
import pandas as pd
import xarray as xr
import numba

def parse_attributes(dataset):
    '''Parse the HDF5 attributes array, convert binary strings to Python strings, and return a dictionary of attributes'''
    attrs = {}
    for key, value in dataset.attrs.items():
        # print('type(value) = ', type(value))
        if type(value) == np.bytes_:
            attrs[key] = value.decode('ascii')
        elif type(value) == np.ndarray:
            values = []
            for v in value:
                if type(v) == np.bytes_:
                    values.append(v.decode('ascii'))
                else:
                    values.append(v)
            attrs[key] = values
        else:
            attrs[key] = value
    return attrs
    
def read_dataset_xarray(filepath, datapath, dims) -> xr.DataArray:
    '''Read n-dimensional HDF5 dataset and return it as an xarray.DataArray'''
    with h5py.File(filepath) as infile:
        dataset = infile[datapath]
        attrs = parse_attributes(dataset)
        coords = {}
        for i in range(len(dataset.shape)):
            coords[dims[i]] = list(range(dataset.shape[i]))
        data_array = xr.DataArray(dataset[()], coords = coords, dims = dims, attrs = attrs)
        return data_array

def read_dataset_pandas(filepath, datapath) -> pd.DataFrame:
    '''Read n-dimensional HDF5 dataset and return it as an xarray.DataArray'''
    with h5py.File(filepath) as infile:
        dataset = infile[datapath]
        attrs = parse_attributes(dataset)
        df = pd.DataFrame(dataset[()], columns = attrs['Column'])
        return df

@numba.njit
def interp(x0: float, x1: float, y0: float, y1: float, xi: float):
    '''
    Linear interpolation:

    Inputs:
        x0: Lower x value
        x1: Upper x value
        y0: Lower y value
        y1: Upper y value
        xi: x value to interpolate

    Returns:
        y1: interpolated y value
    '''
    m = (y1 - y0)/(x1 - x0)
    yi = m * (xi - x0) + y0
    return yi

@numba.njit
def dist(x0, x1, y0, y1):
    ''' Compute distance between two points '''
    return np.sqrt((x1 - x0)**2 + (y1 - y0)**2)
    
@numba.njit
def compute_cell_volumes(water_surface_elev_arr: np.ndarray, cells_surface_area_arr: np.ndarray, starting_index_arr: np.ndarray, count_arr: np.ndarray, elev_arr: np.ndarray, vol_arr: np.ndarray, ntimes: int, ncells: int, VERBOSE=False) -> float:
    '''Compute the volumes of the RAS cells using lookup tables'''
    cell_volumes = np.zeros((ntimes, ncells))

    for time in range(ntimes):
        for cell in range(ncells):
            water_surface_elev = water_surface_elev_arr[time, cell]
            surface_area = cells_surface_area_arr[cell]
            index = starting_index_arr[cell] # Start index in the volume-elevation table for this cell
            count = count_arr[cell] # Number of points in the table for this cell

            # A number of cells have an index that is just past the end of the array. According to Mark Jensen, 
            # these are ghost cells and have a volume of 0.0. The count for these cells should also always be zero. 
            # The code checks for either condition.
            if index >= len(elev_arr) or count == 0:
                cell_volumes[time, cell] = 0.0
            else:
                elev = elev_arr[index:index + count] # Get the water surface elevation array for this cell
                vol = vol_arr[index:index + count] # Get the volume array for this cell

                if water_surface_elev > elev[-1]:
                    '''
                    Compute the net volume: the max volume in the lookup table plus the volume of the water above the max 
                    elevation in the lookup table.
                    
                    Note: this assumes a horizontal water surface, i.e., that the slope of the water surface across a cell
                    is negligible. The error increases with cell size.
                    
                    The validity of this method was confirmed by Mark Jensen on Jul 29, 2022.
                    '''
                    cell_volumes[time, cell] = vol[-1] + (water_surface_elev - elev[-1]) * surface_area
                elif water_surface_elev == elev[-1]:
                    cell_volumes[time, cell] = vol[-1]
                elif water_surface_elev <= elev[0]:
                    cell_volumes[time, cell] = vol[0]
                else:
                    # Interpolate
                    cell_volumes[time, cell] = 0.0 # Default
                    npts = len(elev)
                    for i in range(npts-1, -1, -1):
                        if elev[i] < water_surface_elev:
                            cell_volumes[time, cell] = interp(elev[i], elev[i+1], vol[i], vol[i+1], water_surface_elev)

    return cell_volumes

@numba.njit
def compute_face_areas(water_surface_elev_arr: np.ndarray, faces_lengths_arr: np.ndarray, faces_cell_indexes_arr: np.ndarray, starting_index_arr: np.ndarray, count_arr: np.ndarray, elev_arr: np.ndarray, area_arr: np.ndarray, ntimes: int, ncells: int, nfaces: int):
    '''Compute the areas of the RAS cell faces using lookup tables'''
    face_areas = np.zeros((ntimes, nfaces))
    for time in range(ntimes):
        for face in range(nfaces):
            cell = faces_cell_indexes_arr[face]
            water_surface_elev = water_surface_elev_arr[time, cell]
            index = starting_index_arr[face] # Start index in the area-elevation table for this cell (note: this is indexed by faces)
            count = count_arr[face] # Number of points in the table for this cell (note: this is indexed by faces)

            # A number of cells have an index that is just past the end of the array. According to Mark Jensen, 
            # these are ghost cells and have a volume of 0.0. The count for these cells should also always be zero. 
            # The code checks for either condition.
            if index >= len(elev_arr) or count == 0:
                face_areas[time, face] = 0.0
            else:
                elev = elev_arr[index:index + count] # Get the water surface elevation (Z) array for this face
                area = area_arr[index:index + count] # Get the face area array for this face

                if water_surface_elev > elev[-1]:
                    '''
                    Compute the net face surface area: the max face area in the lookup table plus the face area of 
                    the water above the max elevation in the lookup table.
                    
                    Note: this assumes a horizontal water surface, i.e., that the slope of the water surface across a cell face
                    is negligible. The error increases with cell size.
                    
                    The validity of this method was confirmed by Mark Jensen on Jul 29, 2022.
                    '''
                    face_areas[time, face] = area[-1] + (water_surface_elev - elev[-1]) * faces_lengths_arr[face]
                elif water_surface_elev == elev[-1]:
                    face_areas[time, face] = area[-1]
                elif water_surface_elev <= elev[0]:
                    face_areas[time, face] = area[0]
                else:
                    # Interpolate
                    face_areas[time, face] = 0.0 # Default
                    npts = len(elev)
                    for i in range(npts-1, -1, -1):
                        if elev[i] < water_surface_elev:
                            x = water_surface_elev
                            face_areas[time, face] = interp(elev[i], elev[i+1], area[i], area[i+1], water_surface_elev)
                            # print('i, x, m, x1, y1, y: ', i, x, m, x1, y1, y)
                            if face_areas[time, face] < 0:
                                print('Computed face area = ', face_areas[time, face])
                                print('Time Step: ', time)
                                print('Cell number: ', cell)
                                print('Face number: ', face)
                                print('water_surface_elev: ', water_surface_elev)
                                print('elev: ', elev)
                                print('area: ', area)
                                # msg = 'Negative face area: computed face area = ' + str(face_areas[time, face])
                                raise(ValueError('Negative face area'))

    return face_areas
from shapely.geometry import Polygon
import numpy as np
import geopandas as gpd
import pandas as pd
import holoviews as hv
import geoviews as gv
from datetime import datetime
import matplotlib.pyplot as plt
from pathlib import Path

from typing import Optional

from clearwater_riverine.variables import (
    CHANGE_IN_TIME,
    FACE_NODES,
    FACE_X,
    FACE_Y,
    NFACE,
    NODE_X,
    NODE_Y,
    NUMBER_OF_REAL_CELLS,
    VOLUME,
)
from clearwater_data.variables import VariableRegistry
from clearwater_data.io.zarr import ZarrDataSource

hv.extension('bokeh')

class RiverinePlotter:
    def __init__(
            self,
            registry: Optional[VariableRegistry] = None,
            zarr_data_source_path: Optional[str] = None,
            crs: Optional[str] = None,
        ):
        """
        Initialize the plotter, either from a variable registry or a data source.
        """
        self.registry = registry
        self.zarr_data_source_path = zarr_data_source_path

        if self.zarr_data_source_path is not None:
            # TODO: update to ChunkedZarrDataStore
            self.data_source = ZarrDataSource(store_path=self.data_source_path)
            self.__plotting_data = self._data_source
        elif self.registry is not None:
            self.__plotting_data = self.registry
        else:
            raise ValueError("Exactly one of 'registry' or 'data_source' must be provided.")

        self.polygon_gdf = None
        self.crs = crs
    
    def dynamic_plot(
        self,
        constituent_name: str,
        **kwargs
    ):
        """
        Creates a dynamic polygon plot of constituents in the model domain.
        """
        cmap: str = kwargs.get("cmap", "OrRd")
        clim: tuple[float, float] = kwargs.get("clim", self.__set_clim(constituent_name))
        # TODO: model start and end time not in registry when in chunk mode 
        # TODO: update self._model._start_datetime and endtime to dynamically read from plotting data source
        datetime_range: tuple[datetime, datetime] = kwargs.get("datetime_range", self.__get_datetime_range(constituent_name))
        # TODO: check if volume in plotting data source
        filter_empty: bool = kwargs.get("filter_empty", True)
        # TODO: update to dates in ZarrDataStore - improve efficiency for chunked
        datetimes = pd.to_datetime(
            self.__plotting_data.get(constituent_name)
            .sel(time=slice(datetime_range[0], datetime_range[1]))['time']
            .values
        )
        if self.polygon_gdf is None:
            self.__prep_gdf()
        
        def map_generator(datetime):
            # TODO: update to get directly from data source or registry.
            gdf_plot = self.__get_plotting_gdf(constituent_name, datetime, filter_empty)

            mesh_map = gv.Polygons(
                gdf_plot,
                vdims = [constituent_name, "nface"]).opts(
                    height = 400,
                    width = 800,
                    color=constituent_name,
                    colorbar = True,
                    cmap = cmap,
                    clim = clim,
                    line_width = 0.1,
                    tools = ['hover'],
                    clabel = f"{constituent_name}"                   
                )
            return (mesh_map * gv.tile_sources.CartoLight())
        
        dmap = hv.DynamicMap(map_generator, kdims=['datetime'])
        return dmap.redim.values(datetime=datetimes)              


    def static_plot(self,
        constituent_name: str,
        plotting_datetime: datetime,
        output_path: Optional[str | Path] = None,
        **kwargs
    ):
        """
        Static map at a given datetime.
            
            constituent_name: name of constituent to plot.):
            time_index_range (tuple, optional): minimum and maximum time index to plot.
            filter_empty (boolean, optional): provides users the ability to filter out empty cells. 
        """
        cmap: str = kwargs.get("cmap", "OrRd")
        clim: tuple[float, float] = kwargs.get("clim", self.__set_clim(constituent_name, at_datetime=plotting_datetime))
        filter_empty: bool = kwargs.get("filter_empty", True)

        if self.polygon_gdf is None:
            self.__prep_gdf()

        date_value = self.__plotting_data.get(constituent_name)['time'].sel(
            time=plotting_datetime
        ).values

        gdf_plot = self.__get_plotting_gdf(constituent_name, date_value, filter_empty)
        print(clim)
        fig, ax = plt.subplots()
        gdf_plot.plot(
            column=constituent_name,
            cmap=cmap,
            vmin=clim[0],
            vmax=clim[1],
            edgecolor = 'white',
            linewidth = 0.1,
            ax=ax,
        )
        ax.set_xticks([])
        ax.set_yticks([])
        ax.axis('off')
        if output_path is not None:
            fig.savefig(output_path)
        plt.show()
    
    def quick_plot(self, constituent_name: str, **kwargs):
        """Creates a dynamic scatterplot of cell centroids colored by cell concentration."""
        clim: tuple[float, float] = kwargs.get("clim", self.__set_clim(constituent_name))
        cmap: str = kwargs.get("cmap", "OrRd")
        datetime_range: tuple[datetime, datetime] = kwargs.get("datetime_range", self.__get_datetime_range(constituent_name))
        datetimes = pd.to_datetime(
            self.__plotting_data.get(constituent_name)
            .sel(time=slice(datetime_range[0], datetime_range[1]))['time']
            .values
        )

        def quick_map_generator(datetime):
            """This function generates plots for the DynamicMap"""
            ds = self.__plotting_data.get_at_time(constituent_name, datetime)
            real_cell_index = self.__plotting_data.get(NUMBER_OF_REAL_CELLS)

            ind = np.where(
                ds[constituent_name][0:real_cell_index] > 0
            )
            nodes = np.column_stack(
                [
                    ds[FACE_X][ind], ds[FACE_Y][ind],
                    ds[constituent_name][ind], ds[NFACE][ind]
                ]
            )
            nodes = hv.Points(nodes, vdims=[constituent_name, NFACE])

            point_map = hv.Scatter(
                nodes,
                vdims=['x', 'y', constituent_name, 'nface']
            ).opts(
                width = 1000,
                height = 500,
                color = constituent_name,
                cmap = cmap, 
                clim = clim,
                tools = ['hover'], 
                colorbar = True
            )
            return point_map

        return hv.DynamicMap(quick_map_generator, kdims=['time']).redim.values(time=datetimes)


    def __prep_gdf(self):
        """
        Creates a geodataframe of polygons to represent RAS cells.
        """
        # TODO: can I pull these from the zarr output?
        real_cell_index = self.__plotting_data.get(NUMBER_OF_REAL_CELLS)
        real_face_node_connectivity = self.__plotting_data.get(FACE_NODES)[0:real_cell_index]
        node_x = self.__plotting_data.get(NODE_X)
        node_y = self.__plotting_data.get(NODE_Y)
       
        # turn real cells into polygons
        polygon_list = [
            Polygon(np.column_stack((
                node_x[cell[cell != -1]],
                node_y[cell[cell != -1]]
            )))
            for cell in real_face_node_connectivity
        ]

        self.polygon_gdf = gpd.GeoDataFrame(
            {
                NFACE: np.arange(0, real_cell_index),
                'geometry': polygon_list
            },
            geometry='geometry',
            crs=self.crs
        ).set_index(NFACE)
        
        # convert to WGS84
        self.polygon_gdf = self.polygon_gdf.to_crs('EPSG:4326')
    
    def __get_plotting_gdf(
            self,
            constituent_name: str,
            time: datetime,
            filter_empty: bool,
    ):
        """Create GDF for plotting.
        
        Ultimately, this should be refactored to use xvec (more efficient).
        """
        plotting_values = self.__plotting_data.get_at_time(constituent_name, time)
        
        if filter_empty:
            try:
                # Note: Ensure VOLUME is defined in the data source
                volume = self.__plotting_data.get_at_time(VOLUME, time)
                plotting_values = plotting_values.where(volume != 0)
            except Exception:
                print("Volume filter not working.")
    
        df = (
            plotting_values
            .to_dataframe(name=constituent_name)
            .reset_index()
            .set_index(NFACE)
        )
        return self.polygon_gdf.join(df)
    
    def __set_clim(self, constituent_name: str, at_datetime: Optional[str | datetime] = None):
        """Get minimum and maximum value."""
        # TODO: will this slow things down in chunked mode?
        if at_datetime is not None:
            mn_val = self.__plotting_data.get_at_time(constituent_name, at_datetime).values.min()
            mx_val = self.__plotting_data.get_at_time(constituent_name, at_datetime).values.max()
        else:
            mn_val = self.__plotting_data.get(constituent_name).values.min()
            mx_val = self.__plotting_data.get(constituent_name).values.max()
        return (mn_val, mx_val)


    def __get_datetime_range(self, constituent_name: str):
        """Parse datetime range from plotting data source"""
        # TODO: adapt this to work with ChunkedDataSource for zarr (don't load whole dataset)
        start_datetime = self.__plotting_data.get(constituent_name)["time"][0]
        end_datetime = self.__plotting_data.get(constituent_name)["time"][-1]
        return (start_datetime, end_datetime)
    
        
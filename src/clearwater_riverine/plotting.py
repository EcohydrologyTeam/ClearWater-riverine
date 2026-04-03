from shapely.geometry import Polygon
import numpy as np
import geopandas as gpd
import pandas as pd
import holoviews as hv
import geoviews as gv
from datetime import datetime, timedelta

from typing import Optional

from clearwater_riverine.variables import (
    CHANGE_IN_TIME,
    FACE_NODES,
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
            plotting_values = self.__plotting_data.get_at_time(constituent_name, datetime)
            if filter_empty:
                try:
                    volume = self.__plotting_data.get_at_time(VOLUME, datetime)
                    plotting_values = plotting_values.where(volume != 0)
                except:
                    print("Volume filter not working.")
            # TODO: fix unit handling
            # units = plotting_values.Units

            # join to gdf
            # TODO: explore xvec
            df = (
                plotting_values
                .to_dataframe(name="value")
                .reset_index()
                .set_index("nface")
            )
            gdf_plot = self.polygon_gdf.join(df)
            # TODO: don't make these model attrs, make them plotting atrs
            self.gdf_plot = gdf_plot
            self.df = df
            self.plotting_values = plotting_values

            mesh_map = gv.Polygons(
                gdf_plot,
                vdims = ["value", "nface"]).opts(
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
        plotting_timestep: datetime,
        constituent_name: Optional[str] = None,
        **kwargs
    ):
        """
        Static map at a given datetime.
            
            constituent_name: name of constituent to plot.):
            time_index_range (tuple, optional): minimum and maximum time index to plot.
            filter_empty (boolean, optional): provides users the ability to filter out empty cells. 
        """
        return
    
    def quick_plot(self, **kwargs):
        """"""
        return
    
    def __prep_gdf(self):
        """
        Creates a geodataframe of polygons to represent RAS cells.
        """
        # TODO: can I pull these from the zarr output?
        real_cell_index = self.__plotting_data.get(NUMBER_OF_REAL_CELLS)
        real_face_node_connectivity = self.__plotting_data.get(FACE_NODES)[0:real_cell_index]
        node_x = self.__plotting_data.get(NODE_X)
        node_y = self.__plotting_data.get(NODE_Y)
       
        # turn real mesh cells into polygons
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


    
    def __set_clim(self, constituent_name: str):
        """Get minimum and maximum value."""
        # TODO: will this slow things down in chunked mode?
        mn_val = self.__plotting_data.get(constituent_name).values.min()
        mx_val = self.__plotting_data.get(constituent_name).values.max()
        return (mn_val, mx_val)


    def __get_datetime_range(self, constituent_name: str):
        """Parse datetime range from plotting data source"""
        # TODO: adapt this to work with ChunkedDataSource for zarr (don't load whole dataset)
        start_datetime = self.__plotting_data.get(constituent_name)["time"][0]
        end_datetime = self.__plotting_data.get(constituent_name)["time"][-1]
        return (start_datetime, end_datetime)
    
        
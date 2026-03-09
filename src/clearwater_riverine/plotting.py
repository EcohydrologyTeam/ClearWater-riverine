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

class RiverinePlotter:
    def __init__(self, model_instance):
        self._model = model_instance
        self.polygon_gdf = None
        self.default_constituent = None
    
    def dynamic_plot(
        self,
        **kwargs
    ):
        """Creates a dynamic polygon plot of constituents in the model domain."""
        constituent_name: str = kwargs.get("constituent_name", self.default_constituent)
        cmap: str = kwargs.get("cmap", "OrRd")
        clim: tuple[float, float] = kwargs.get("clim", self.__set_clim(constituent_name))
        # TODO: model start and end time not in registry when in chunk mode 
        datetime_range: tuple[datetime, datetime] = kwargs.get("datetime_range", (self._model._start_datetime, self._model._end_datetime))
        filter_empty: bool = kwargs.get("filter_empty", True)
        datetimes = pd.date_range(
            start=datetime_range[0],
            end=datetime_range[1],
            freq=timedelta(seconds=self._model.registry.get(CHANGE_IN_TIME))
        )

        if self.polygon_gdf is None:
            self.__prep_gdf()
        
        def map_generator(datetime):
            plotting_values = self._model.registry.get_at_time(constituent_name, datetime)
            if filter_empty:
                try:
                    volume = self._model.registry.get_at_time(VOLUME, datetime)
                    plotting_values = plotting_values.where(volume != 0)
                except:
                    print("Volume filter not working.")
            units = plotting_values.Units 

            # join to gdf
            # TODO: explore xvec
            df = (
                plotting_values
                .to_dataframe(name="value")
                .reset_index()
                .set_index("nface")
            )
            gdf_plot = self.polygon_gdf.join(df)
            print(gdf_plot)
            self._model.gdf_plot = gdf_plot
            self._model.df = df
            self_model.plotting_values = plotting_values

            mesh_map = gv.Polygons(
                gdf_plot,
                vdims = ["value", "nface"].opts(
                    height = 400,
                    width = 800,
                    color=constituent_name,
                    colorbar = True,
                    cmap = cmap,
                    clim = clim,
                    line_width = 0.1,
                    tools = ['hover'],
                    clabel = f"{constituent_name} ({units})"                   
                )
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
        
    
    def quick_plot(self, **kwargs):
        """"""

    
    def __prep_gdf(self):
        """
        Creates a geodataframe of polygons to represent RAS cells.
        """
        real_cell_index = self._model.registry.get(NUMBER_OF_REAL_CELLS)
        real_face_node_connectivity = self._model.registry.get(FACE_NODES)[0:real_cell_index]
        node_x = self._model.registry.get(NODE_X)
        node_y = self._model.registry.get(NODE_Y)
       
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
            crs=self._model.crs
        ).set_index(NFACE)

    
    def __set_clim(self, constituent_name: str):
        """Get minimum and maximum value."""
        constituent = self._model._constituents[constituent_name]
        mn_val = constituent.get_minimum_value(self._model.registry)
        mx_val = constituent.get_maximum_value(self._model.registry)
        return (mn_val, mx_val)
    
        
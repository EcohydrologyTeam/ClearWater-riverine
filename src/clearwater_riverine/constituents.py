from typing import (
    Dict,
    Literal,
    Optional
)
from pathlib import Path
import warnings

import pandas as pd
import xarray as xr
import numpy as np
from datetime import datetime

from clearwater_data.variables import VariableRegistry, DataArrayVariable
from clearwater_riverine.linalg import RHS
from clearwater_riverine.variables import (
    BOUNDARY_CONDITION_LINE_ID,
    BOUNDARY_FACE_INDEX,
    BOUNDARY_NAME,
    CHANGE_IN_TIME,
    COEFFICIENT_TO_DIFFUSION_TERM,
    FLOW_ACROSS_FACE,
    EDGE_FACE_CONNECTIVITY,
    NEDGE,
    NFACE,
    NUMBER_OF_REAL_CELLS,
    VOLUME,
)


def _validate_constituent_values(
    values,
    *,
    constituent_name: str,
    source_label: str,
    raise_on_nan: bool = True,
    warn_on_negative: bool = True,
) -> None:
    """Validate a constituent IC/BC source array before it enters the registry.

    Phase F (2026-05-21) T2-D: catches the silent-propagation bug in
    both repos where a malformed CSV (NaN rows, negative values) flowed
    through the transport solve and produced non-physical output with
    no diagnostic.

    Args:
        values: Scalar (int/float), numpy array, or xr.DataArray.
        constituent_name: For diagnostic messages.
        source_label: "initial_conditions" or "boundary_conditions".
        raise_on_nan: When True (default), raise ValueError on any
            NaN in the source. Set False to allow NaN through (e.g., a
            data source that uses NaN as a sentinel for "use default").
        warn_on_negative: When True (default), emit a UserWarning on
            any negative value. Most water-quality constituents
            (concentrations, biomass, temperature in deg C) are
            physically non-negative; an early warning catches CSV
            sign errors or interpolation artifacts.
    """
    if isinstance(values, (int, float)):
        arr = np.asarray([values], dtype=np.float64)
    elif isinstance(values, xr.DataArray):
        arr = np.asarray(values.values)
    else:
        arr = np.asarray(values)
    if arr.size == 0:
        return
    if raise_on_nan:
        nan_mask = np.isnan(arr)
        n_nan = int(nan_mask.sum())
        if n_nan > 0:
            raise ValueError(
                f"Constituent {constituent_name!r}: {source_label} source "
                f"contains {n_nan} NaN value(s) (of {arr.size} total). "
                "NaN in IC/BC inputs propagates silently through the "
                "transport solve and produces non-physical output. Fix "
                "the source CSV/array (drop missing rows, fill via "
                "interpolation, or supply a scalar default), or pass "
                "``raise_on_nan=False`` if NaN is intentional."
            )
    if warn_on_negative:
        try:
            neg_mask = arr < 0
            n_neg = int(neg_mask.sum())
        except TypeError:
            n_neg = 0  # non-numeric (e.g. boolean wet mask)
        if n_neg > 0:
            min_val = float(np.nanmin(arr))
            warnings.warn(
                f"Constituent {constituent_name!r}: {source_label} source "
                f"contains {n_neg} negative value(s) (of {arr.size} total; "
                f"min = {min_val:.4g}). Most water-quality constituents "
                "are physically non-negative; check for sign errors in "
                "the source CSV or interpolation artifacts.",
                stacklevel=3,
            )


class Constituent:
    """Constituent class."""
    def __init__(
        self,
        constituent_name: str,
        registry: VariableRegistry,
        initial_conditions: xr.DataArray,
        boundary_conditions: xr.DataArray,
        constituent_config: dict,
        start_datetime: datetime,
        point_sources_path: Optional[str | Path] = None,
    ):
        self._name = constituent_name
        self.__units = constituent_config.get("units", None)
        # Phase-D Unit D1: intensive vs extensive flag. Intensive
        # scalars (e.g. water temperature) skip the LHS rule-3
        # donor-diagonal amendment on wet-dry edges (which would
        # otherwise produce spurious cooling), skip the IC-zeroing
        # pass (zeroing a temperature in a sub-threshold cell is
        # non-physical), and skip the end-of-run mass_lost_to_dry
        # warning (the BC inflow MASS denominator has the wrong units
        # for an intensive scalar). Default ``False`` preserves the
        # existing concentration-species behaviour. Consumed by
        # ``LHS.update_values``, ``zero_dry_initial_conditions``, and
        # ``emit_mass_loss_warning`` via ``getattr(c, "is_intensive",
        # False)``.
        self.is_intensive: bool = bool(
            constituent_config.get("is_intensive", False)
        )
        # Phase F T2-B (2026-05-21): first-order decay rate. Config
        # value is per day (matches streaming convention and common
        # water-quality literature for nutrients, BOD, pathogen
        # die-off); stored internally as 1/s for direct use in
        # ``k * V[t+1]`` diagonal adjustment. Default 0.0 (no decay)
        # is conservative-transport behaviour. The transport engine
        # adds ``k * V[t+1]`` to the LHS diagonal at each step when
        # decay_rate > 0.
        decay_rate_per_day = float(
            constituent_config.get("decay_rate", 0.0) or 0.0
        )
        self.decay_rate: float = decay_rate_per_day / 86400.0
        self.__initial_condition_spatial_field = constituent_config["initial_conditions"]["data"].get(
            "spatial_field", "Cell_Index"  # Default to old config requriement
        )
        self.__boundary_condition_spatial_field = constituent_config["boundary_conditions"]["data"].get(
            "spatial_field", "RAS2D_TS_Name"  # Default to old config requriement
        )
        registry.register(
            f"{self._name}_initial",
            initial_conditions,
        )
        registry.register(
            f"{self._name}_boundary",
            boundary_conditions,
        )

        ## Initialize
        self.register_constituent(registry)

        self.set_initial_conditions(
            registry=registry,
            start_datetime=start_datetime,
        )
        self.set_boundary_conditions(
            registry=registry,
        )

        # Phase F T2-A (2026-05-21): load point sources if configured.
        # Stored as DataArrays in the registry keyed by
        # ``{name}_point_source_flows`` and
        # ``{name}_point_source_concentrations``; the LHS reads the
        # sink contribution and the RHS reads the source contribution
        # from those registry entries.
        self.has_point_sources: bool = False
        if point_sources_path is not None:
            self._load_point_sources(
                registry=registry,
                filepath=Path(point_sources_path),
            )

        self.rhs = RHS(
            registry=registry,
        )


    def register_constituent(
        self,
        registry: VariableRegistry
    ):
        """Register constituent to variable registry."""
        # unregister if it already exists
        if self._name in registry:
            registry.unregister(self._name)

        # initialize
        registry.register(
            self._name,
            DataArrayVariable(
                xr.full_like(
                    registry.get_variable(VOLUME).get_data(),
                    np.nan
                )
                .rename(self._name)
                .assign_attrs({
                    'units': self.__units
                }),
                space_dimension=NFACE,
            ),
        )

    def reset_initial_conditions(
        self,
        registry: VariableRegistry,
        initial_conditions: xr.DataArray,
    ):
        """Used for chunking mode to reset initial conditions from final calculation."""
        registry.unregister(f"{self._name}_initial")
        registry.register(
            f"{self._name}_initial",
            DataArrayVariable(
                initial_conditions,
                space_dimension=NFACE,
            ),
        )
        self.__initial_condition_spatial_field = NFACE

    def set_initial_conditions(
        self,
        registry: VariableRegistry,
        start_datetime: datetime,

    ):
        """Define constituent initial conditions."""
        constituent = registry.get_at_time(self._name, start_datetime)
        initial = registry.get_at_time(f"{self._name}_initial", start_datetime)

        # Phase F (2026-05-21) T2-D: validate IC source before it enters
        # the registry. Raises on NaN, warns on negative values.
        _validate_constituent_values(
            initial,
            constituent_name=self._name,
            source_label="initial_conditions",
        )

        if isinstance(initial, xr.DataArray):
            registry.set_at_time(
                self._name,
                start_datetime,
                initial
                .rename({self.__initial_condition_spatial_field: NFACE})  # Align to mesh coords
                .reindex(nface=constituent.nface)
                .data
            )
        elif isinstance(initial, (float, int)):
            registry.set_at_time(
                self._name,
                start_datetime,
                initial
            )
        

    def set_boundary_conditions(
        self,
        registry: VariableRegistry,
    ):
        """Define boundary conditions for the Constituent."""
        # retrieve necessary variables
        boundary = registry.get_variable(f"{self._name}_boundary").get_data()
        constituent = registry.get_variable(self._name).get_data()

        # Phase F (2026-05-21) T2-D: validate BC source before it enters
        # the registry. Catches NaN/negative values that the source
        # CSV already carries (a malformed input file).
        _validate_constituent_values(
            boundary,
            constituent_name=self._name,
            source_label="boundary_conditions",
        )
        target_time = registry.get_variable(self._name).get_data().time
        boundary_index = registry.get_variable(BOUNDARY_FACE_INDEX).get_data()
        boundary_names = registry.get_variable(BOUNDARY_NAME).get_data()
        edges_face1 = registry.get_variable(EDGE_FACE_CONNECTIVITY).get_data().T[0]
        edges_face2 = registry.get_variable(EDGE_FACE_CONNECTIVITY).get_data().T[1]

        # find cells associated with each cell
        ghost_cells = edges_face2[boundary_index]

        # linear interpolation over time
        if isinstance(boundary, xr.DataArray):
            # Phase J+1 (2026-05-23): align the BC source's time
            # coordinate dtype to the target's before interp. When the
            # BC source is read from a CSV via pandas (datetime64[us])
            # and the target time comes from the chunked mesh's VOLUME
            # axis (datetime64[ns], pandas default for date_range),
            # ``boundary.interp(time=target_time, method='linear')``
            # silently returns ALL NaN despite the source covering the
            # target window. Both [us] and [ns] sources work fine when
            # the target matches; the failure mode is the dtype-unit
            # mismatch specifically. Cast the source's time coord to
            # the target's dtype before interp so the unit conversion
            # is explicit and total ordering across the two axes is
            # what xarray's interp expects. Real-world impact: any run
            # whose BC CSV is loaded via pandas and whose mesh time
            # comes via pd.date_range hit this; the chunked-mode reload
            # path made it fatal because chunk 2+ has no out-of-window
            # data to fall back on.
            tgt_dt = (target_time.values.dtype
                      if hasattr(target_time, "values")
                      else target_time.dtype)
            if boundary.time.dtype != tgt_dt:
                boundary = boundary.assign_coords(
                    time=boundary.time.values.astype(tgt_dt)
                )
            boundary = boundary.interp(
                time=target_time,
                method="linear"
            )
            # Phase H-2 amendment (2026-05-21): forward-fill then
            # backward-fill any trailing/leading NaN produced by
            # ``interp`` when the BC CSV does not cover the full
            # simulation window. Daily-cadence BC CSVs (the common
            # case for Sep 2008 Santiam-Salem) end at the last
            # midnight in the window and ``interp(method='linear')``
            # returns NaN for every hour past that timestamp. The
            # historically-consistent behaviour (matching what RAS
            # does for its own BCs, and what the pre-Phase-H code
            # accidentally allowed through) is to hold the last
            # known daily value through the trailing hours; ffill
            # for trailing NaN and bfill for any pathological
            # leading NaN restores that without re-introducing the
            # silent-NaN propagation that H-2 was added to catch
            # (the post-interp validation below is now a guard
            # against genuinely-bad CSVs, not a calendar artefact).
            boundary = boundary.ffill("time").bfill("time")
            # Phase H-2 (2026-05-21): re-validate AFTER the
            # interpolation step. Interpolating from a source CSV that
            # ends before ``end_datetime`` (or starts after
            # ``start_datetime``) extrapolates to NaN at the
            # uncovered timestamps; pre-Phase-H this NaN flowed
            # silently into the ghost-cell injection. The pre-interp
            # validation above catches malformed source CSVs; the
            # post-interp validation here catches the "BC CSV doesn't
            # cover the simulation window" case that the pre-interp
            # validation cannot.
            _validate_constituent_values(
                boundary,
                constituent_name=self._name,
                source_label="boundary_conditions (post-interpolation)",
            )

            # Internal-BC support Step 3 (2026-05-23): split the BC
            # lines into External vs Internal subsets and process them
            # along separate paths. External-type BC lines sit on the
            # 2D perimeter and mass enters via ghost cells -- the
            # existing reshape (boundary_name -> ghost_cell -> nface)
            # is correct for those. Internal-type BC lines are drawn
            # cross-channel INSIDE the mesh and have no ghost cells;
            # mass injection happens at interior cells via the
            # synthetic point-source path wired by Step 2 in model.py.
            # Passing an Internal line through the ghost-cell reshape
            # silently injects nothing (the line has no entry in
            # ``ghost_cells``), so we MUST exclude it here.
            #
            # The line-types dict is published by Step 1 onto
            # ``mesh.attrs['internal_bc_line_types']`` (in
            # ``io/hdf.py::__read_internal_bc_metadata``). We read it
            # via ``registry.get_variable(VOLUME).get_data()
            # .to_dataset().attrs`` -- the mesh attrs flow through
            # this path because Step 2 stashes them on the VOLUME
            # DataArray's containing Dataset. When the dict is missing
            # or empty (older HDFs, External-only HDFs, or runs where
            # Step 2 hasn't wired the attrs through yet), every BC
            # line is treated as External and the original code path
            # is taken bit-identically. Unknown line names default to
            # External as well -- the conservative behaviour.
            # Bugfix (2026-05-23): xarray's DataArray.to_dataset() does
            # NOT propagate the DataArray's attrs to the resulting
            # Dataset's .attrs -- they become variable-level attrs on the
            # contained variable instead. Read directly from the
            # DataArray's .attrs, which Step 2's
            # ClearwaterRiverine.__bridge_internal_bc_metadata writes
            # the 'internal_bc_line_types' / 'internal_bc_cells_by_line'
            # dicts to. Verified by a fresh-init probe of the registry
            # state: VOLUME.attrs has the keys; .to_dataset().attrs is
            # empty.
            try:
                mesh_attrs = (
                    registry.get_variable(VOLUME)
                    .get_data()
                    .attrs
                )
            except Exception:
                mesh_attrs = {}
            line_types = mesh_attrs.get('internal_bc_line_types', {}) or {}

            # Bugfix (2026-05-23): partition BC line names using the BC
            # SOURCE's RAS2D_TS_Name coord, NOT BOUNDARY_NAME. The
            # ``BOUNDARY_NAME`` registry variable contains only the
            # names of EXTERNAL faces (one entry per external mesh
            # face -- e.g., 17 entries all = 'Downstream' for the
            # Corvallis HDF). Internal-type BC lines have no external
            # faces and therefore do NOT appear in BOUNDARY_NAME at
            # all; partitioning over BOUNDARY_NAME would silently miss
            # every Internal line. The BC source DataArray's
            # ``RAS2D_TS_Name`` coord, by contrast, contains every BC
            # line name that the user's CSV provided (both Internal
            # and External). De-dupe (boundary_names from the registry
            # may carry duplicates from face-replication; the source
            # coord carries each name once).
            bc_name_list = list(dict.fromkeys(
                str(n) for n in np.asarray(
                    boundary.RAS2D_TS_Name.values
                ).tolist()
            ))
            internal_names_list = [
                n for n in bc_name_list
                if str(line_types.get(n, 'External')).lower() == 'internal'
            ]
            external_names_list = [
                n for n in bc_name_list
                if str(line_types.get(n, 'External')).lower() != 'internal'
            ]

            # Stash the interpolated Internal-line subset on the
            # registry so Step 2 (the synthetic point-source path in
            # model.py) can pick up the per-timestep concentration
            # values keyed by BC line name. The dimension is
            # ``RAS2D_TS_Name``; consumers map that to per-cell flows
            # using ``mesh.attrs['internal_bc_cells_by_line']`` and
            # ``mesh.attrs['internal_bc_flows']`` (also published by
            # Step 1). We unregister-then-register so repeated
            # ``set_boundary_conditions`` calls (e.g. chunked mode)
            # refresh the value cleanly.
            internal_key = f"{self._name}_boundary_interp_internal"
            if internal_key in registry:
                registry.unregister(internal_key)
            if len(internal_names_list) > 0:
                internal_lines_subset = boundary.sel(
                    RAS2D_TS_Name=internal_names_list
                )
                registry.register(
                    internal_key,
                    DataArrayVariable(
                        internal_lines_subset,
                        space_dimension="RAS2D_TS_Name",
                    ),
                )

            # If every BC line is Internal there is no ghost-cell
            # reshape to perform; the constituent registry value for
            # this constituent stays at its initial-condition state on
            # exit (Step 2 writes the Internal-line mass via the
            # synthetic point-source path, not by overwriting ghost
            # cells here).
            if len(external_names_list) == 0:
                return

            # reshape from (time, boundary_name) to (time, boundary_index)
            # then map boundary indices to their associated ghost cells.
            # Filter both ``boundary_names`` and ``ghost_cells`` by the
            # External-subset mask so the reshape only processes lines
            # that actually have a ghost-cell entry. When every line is
            # External (``external_mask`` all-True), this is byte-
            # equivalent to the historical reshape -- both arrays
            # collapse to their unfiltered selves and the downstream
            # ``.sel(RAS2D_TS_Name=...)`` + ``.groupby("nface").first()``
            # behave exactly as before.
            names_array = np.asarray(boundary_names)
            external_set = set(external_names_list)
            external_mask = np.array(
                [str(n) in external_set for n in names_array],
                dtype=bool,
            )
            external_boundary_names = boundary_names.isel(
                **{boundary_names.dims[0]: external_mask}
            )
            external_ghost_cells = ghost_cells.isel(
                **{ghost_cells.dims[0]: external_mask}
            )
            boundary = boundary.sel(
                RAS2D_TS_Name=external_boundary_names
            ).assign_coords(
                nface=external_ghost_cells
            ).groupby(
                "nface"
            ).first()

            # reshape to the shape of our constituent array
            boundary_reindexed = boundary.reindex(nface=constituent.nface)

            # place the boundary conditions into the constituent array
            registry.set(
                self._name,
                xr.where(
                    boundary_reindexed.notnull(),
                    boundary_reindexed,
                    constituent
                )
            )

        # TODO: does there need to be a custom set method to set at custom locations?
        elif isinstance(boundary, (float, int)):
            constituent.loc[dict(nface=ghost_cells)] = boundary


    def _load_point_sources(
        self,
        registry: VariableRegistry,
        filepath: Path,
    ) -> None:
        """Load per-cell, per-time point sources from CSV (Phase F T2-A).

        CSV schema (fixed):

            Cell_Index,Datetime,Flow_Rate,Concentration
            42,2008-09-01 12:00:00,0.5,15.0
            ...

        ``Flow_Rate`` is volumetric (m^3/s in default SI units);
        positive = source (mass added), negative = sink (cell loses
        mass at its current concentration). ``Concentration`` is in
        the constituent's reporting units and is ignored for sinks
        (sink removal uses the cell's current concentration via the
        implicit LHS diagonal).

        The CSV is sorted per-cell, outer-merged onto the registry
        time axis (so missing rows interpolate linearly between
        knots), and stored on the registry as
        ``{name}_point_source_flows`` and
        ``{name}_point_source_concentrations`` -- DataArrays of shape
        (time, nface). Cells with no source rows stay at zero.

        Point-source volumetric flow is NOT propagated through the
        hydrodynamic mesh (the HEC-RAS flow field is fixed); use this
        for trace-level loadings that do not perturb the bulk flow.

        Validation: NaN values in the CSV are not allowed (raises
        ValueError per T2-D's _validate_constituent_values contract);
        negative Concentration values warn (most species are
        physically non-negative).
        """
        df = pd.read_csv(filepath, parse_dates=['Datetime'])
        required = {'Cell_Index', 'Datetime', 'Flow_Rate', 'Concentration'}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(
                f"Constituent {self._name!r}: point_sources CSV {filepath} "
                f"missing required columns: {sorted(missing)}. "
                f"Expected schema: Cell_Index, Datetime, Flow_Rate, Concentration."
            )
        df['Cell_Index'] = df['Cell_Index'].astype(int)

        _validate_constituent_values(
            df['Flow_Rate'].to_numpy(),
            constituent_name=self._name,
            source_label="point_sources Flow_Rate",
            raise_on_nan=True,
            warn_on_negative=False,  # negative flow == sink: legitimate
        )
        # Phase I-3 (2026-05-21): sink rows (Flow_Rate < 0) are now
        # fully supported via the LHS-diagonal modification in
        # ``TransportEngine.run``. The earlier T2-A warning that said
        # sinks were unsupported has been removed; negative-Flow_Rate
        # rows now produce a per-step mass withdrawal at each sink
        # cell's current concentration.
        _validate_constituent_values(
            df['Concentration'].to_numpy(),
            constituent_name=self._name,
            source_label="point_sources Concentration",
            raise_on_nan=True,
            warn_on_negative=True,
        )

        # Validate Cell_Index range against the mesh
        nreal = int(registry.get_variable(NUMBER_OF_REAL_CELLS).get_data())
        bad = df[(df['Cell_Index'] < 0) | (df['Cell_Index'] >= nreal)]
        if len(bad) > 0:
            raise ValueError(
                f"Constituent {self._name!r}: point_sources CSV {filepath} "
                f"contains Cell_Index values outside [0, {nreal}): "
                f"{sorted(bad['Cell_Index'].unique().tolist())}"
            )

        volume = registry.get_variable(VOLUME).get_data()
        model_times = pd.DatetimeIndex(volume.time.values)
        ntime = len(model_times)
        nface = volume.sizes['nface']

        flows = np.zeros((ntime, nface), dtype=np.float64)
        concs = np.zeros((ntime, nface), dtype=np.float64)

        model_df = pd.DataFrame({'Datetime': model_times})
        for cell_idx, group in df.groupby('Cell_Index'):
            cell_idx = int(cell_idx)
            group = group.sort_values('Datetime')
            merged = pd.merge(
                model_df,
                group[['Datetime', 'Flow_Rate', 'Concentration']],
                on='Datetime',
                how='outer',
            ).sort_values('Datetime')
            merged['Flow_Rate'] = merged['Flow_Rate'].interpolate(method='linear').fillna(0)
            merged['Concentration'] = merged['Concentration'].interpolate(method='linear').fillna(0)
            merged = merged[merged['Datetime'].isin(model_df['Datetime'])]
            flows[:, cell_idx] = merged['Flow_Rate'].values
            concs[:, cell_idx] = merged['Concentration'].values

        flows_da = xr.DataArray(
            flows,
            dims=('time', NFACE),
            coords={'time': volume.time.values, NFACE: volume.nface.values},
            attrs={'units': 'm^3/s', 'long_name': f'{self._name} point-source flow rate'},
        )
        concs_da = xr.DataArray(
            concs,
            dims=('time', NFACE),
            coords={'time': volume.time.values, NFACE: volume.nface.values},
            attrs={'units': self.__units or 'unknown',
                   'long_name': f'{self._name} point-source concentration'},
        )

        flows_key = f"{self._name}_point_source_flows"
        concs_key = f"{self._name}_point_source_concentrations"
        if flows_key in registry:
            registry.unregister(flows_key)
        if concs_key in registry:
            registry.unregister(concs_key)
        registry.register(flows_key, DataArrayVariable(flows_da, space_dimension=NFACE))
        registry.register(concs_key, DataArrayVariable(concs_da, space_dimension=NFACE))
        self.has_point_sources = True


    def get_minimum_value(self, registry):
        constituent = registry.get_variable(self._name).get_data()
        return constituent.min()

    def get_maximum_value(self, registry):
        constituent = registry.get_variable(self._name).get_data()
        return constituent.max()


    def _calculate_mass_flux(
        self,
        registry: VariableRegistry,
    ):
        # advection / diffusion coefficients from n timestep are used
        advection_coefficient = registry.get_variable(FLOW_ACROSS_FACE).get_data()[:-1]
        diffusion_coefficient = registry.get_variable(COEFFICIENT_TO_DIFFUSION_TERM).get_data()[:-1]
        edges_face1 = registry.get_variable(EDGE_FACE_CONNECTIVITY).get_data().T[0]
        edges_face2 = registry.get_variable(EDGE_FACE_CONNECTIVITY).get_data().T[1]

        # concentrations from n+1 timestep are used:
        # indexing here shifts the data accordingly 
        negative_condition = advection_coefficient < 0
        parent_concentration = registry.get_variable(self._name).get_data()[1:].sel(nface=edges_face1)
        neighbor_concentration = registry.get_variable(self._name).get_data()[1:].sel(nface=edges_face2)

        # coerce times so the xarray where function will work
        # think it makes most sense to assign the coordinates of advection coefficient
        # at n timestep, mass flux is calculated to then get the concentration at the next timestep
        parent_concentration = parent_concentration.assign_coords(time=advection_coefficient.time)
        neighbor_concentration = neighbor_concentration.assign_coords(time=advection_coefficient.time)

        advection_mass_flux = xr.where(
            negative_condition,
            advection_coefficient * neighbor_concentration,
            advection_coefficient * parent_concentration
        ) * registry.get_variable(CHANGE_IN_TIME).get_data()

        diffusion_mass_flux = diffusion_coefficient * \
            (neighbor_concentration - parent_concentration) * \
            registry.get_variable(CHANGE_IN_TIME).get_data()

        total_mass_flux = advection_mass_flux + diffusion_mass_flux

        # overwrite=True: in chunked mode this is recomputed every chunk on
        # the chunk-resident window. The whole-run reduction is folded into
        # the model's cross-chunk accumulator at each __finalize_chunk
        # (Phase-C C3a), so only the current chunk's flux needs to be
        # resident here. In non-chunked mode this is the single registration
        # (overwrite is a no-op on first register).
        registry.register(
            f"{self._name}_mass_flux",
            DataArrayVariable(total_mass_flux, space_dimension=NEDGE),
            overwrite=True,
        )

# Changelog

All notable changes to ClearWater-Riverine are documented here.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/), and this project adheres to [Semantic Versioning](https://semver.org/).

---

## [v0.8.0] —  2025-05-22

> 96 commits to `main` by @sjordan29, @aufdenkampe, and @ptomasula
> spanning 2025-10-03 through 2026-05-05

This release represents a major scalability refactor that was anticipated in the v0.7.1 notes.
The core I/O layer, transport engine, and constituent system were overhauled, and significant
new capabilities were added for chunked/zarr output, visualization, and post-processing.

---

### ⚠️ Breaking Changes

- **`transport.py` renamed to `model.py`** — the file that controls the Riverine model was
  renamed to better reflect its role; a separate `transport.py` now specifically handles the
  advection-diffusion transport engine.
- **I/O overhaul** — the data-source layer was refactored around a new `RASHDFDataSource`
  class and a revised `BaseClass` interface; existing code that calls internal I/O helpers
  directly will need to be updated.
- **Pixi manifest moved to `pyproject.toml`** — the `pixi.toml` manifest was replaced by
  `pyproject.toml` as the single project descriptor (issues #118, #119).

---

### 🚀 Major New Features

#### Chunked I/O & Zarr Output
- Introduced a **chunked simulation mode** that processes the time series in user-defined
  chunks and writes results incrementally to a Zarr data store, enabling large simulations
  that exceed available memory.
- New zarr initialization, chunk-write, and finalization workflow:
  unregisters/re-registers variables between chunks; preserves the last time-step of each
  chunk as the initial condition for the next (`keep timestep between chunk writes`).
- Added `read zarr` support to the getting-started example notebook.

#### Refactored HEC-RAS HDF Data Source (`RASHDFDataSource`)
- Full re-implementation of HEC-RAS 2D HDF parsing, aligned with the new `BaseClass`
  interface.
- New `read` and `read_chunk` methods for streaming large HDF files.
- Mesh instantiation now happens inside the data source; spatial field name is configurable.

#### New Constituent System
- Overhauled `constituents.py`: `Constituent` objects now carry initial conditions
  (xarray DataArrays **and** scalar float/int), boundary conditions, and updated attributes.
- New `set` method for programmatically updating transport-engine, initial, and boundary
  conditions at runtime.
- `DataArrayVariable` objects now carry an explicit **space dimension**.

#### Transport Engine
- Created a dedicated **transport engine** for advection and diffusion, separated from the
  model-orchestration code.
- Rewrote the **left-hand-side (LHS)** matrix builder to use the variable registry.
- Rewrote **right-hand-side (RHS)** assembly; fixed internal flow-in / flow-out index logic.
- Solver fixed; matrix sizes aligned to the true real-cell count (ghost/boundary cells
  excluded).

#### Plotting Module
- New `plotting.py` module with **static** and **dynamic** (interactive slider) plot
  functions driven by the model variable registry.
- Plotter is called outside the `Riverine` class for cleaner post-run use.
- Example `refactor_testing_plotting.ipynb` notebook demonstrates usage.
- GIFs (`ClearWater-Riverine-Ohio.gif`, `ClearWater-Riverine-and-EFDC-Ohio.gif`) moved to
  `docs/gifs/`.

#### Post-Processing Utilities (`postproc_util.py`)
- New **mass flux calculation** across mesh edges.
- **Mass balance** calculation (corrected volume formula; diffusion impact documented).
- `edge_vertical_area_calculation` bug fixed (replaced `inf` values with `0`).
- Refactored to use the variable registry; option to calculate on-the-fly instead of
  creating separate output variables.
- Dedicated post-processing notebook (`notebook with post processing calculations`).

---

### ✨ Improvements

#### Boundary Conditions
- Boundary initialization restructured: boundaries are now placed before constituent
  assignment and aligned with the real-cell mesh.
- **Forward-fill interpolation replaced with linear interpolation** for boundary time
  series with unequal lengths.
- Fixed a bug where boundary CSVs with unequal row counts produced NaN values that
  corrupted results.
- Improved `fix boundary setting` logic in the new set method.

#### Calculated Variables
- Additional derived variables added: **wetted surface area**, **max depth**, **average
  depth** (previously only in the v0.7.1 wet-surface-area PR, now integrated into the new
  architecture).
- `number of real cells` computed and stored.
- Variable lookups and types updated throughout; `add flexibility to spatial field name`
  allows non-standard HEC-RAS field naming.

#### Configuration & Relative Pathing
- All file paths are now resolved **relative to the notebook's working directory** (the
  directory from which Python/Jupyter was launched), fixing coupling bugs when
  `riverine.yml` is called from ClearWater-modules (#133).
- Data paths moved into the `data/` subtree; configs aligned accordingly (related to
  ClearWater-data #7).
- `sumwere_creek_coarse` example dataset added to the repo for self-contained demos.
- `example_config.yml` updated to match new pathing conventions.

#### Examples & Documentation
- `01_getting_started_riverine.ipynb` updated end-to-end with the new API, zarr reading,
  and interactive plot slider.
- Older coupling notebooks moved to `examples/archive/`.
- `examples/readme.md` added.
- README updated to reflect `pyproject.toml` as the manifest file.
- In-code documentation updated throughout (`update documentation` commit).

---

### 🐛 Bug Fixes

| Date | Description |
|------|-------------|
| 2025-10-20 | Various bug fixes in HEC-RAS parsing |
| 2025-11-26 | Multiple bug fixes in boundary placement and chunk timestep tracking |
| 2025-12-19 | Variable name typos and formatting errors in restructured source files |
| 2025-12-22 | LHS matrix: corrected flow-in / flow-out index logic |
| 2025-12-23 | Transport engine, RHS, and solver bug fixes |
| 2025-12-23 | Matrix sizes aligned to real cell count |
| 2026-01-05 | Addressed xarray deprecation warning |
| 2026-03-09 | Fixed boundary-setting logic in new set method |
| 2026-03-23 | `edge_vertical_area_calculation`: replaced `inf` values with `0` |
| 2026-03-18 | Corrected volume formula in mass balance calculation |
| 2026-04-06 | Boundary CSV NaN bug: mismatched row counts in multi-boundary datasets |
| 2026-04-13 | Fixed unchunked output (data not written when chunking disabled) |
| 2026-04-13 | Fixed relative-pathing bug when riverine is invoked from modules |

---

### 🔧 Dependencies & Tooling

- **Pixi manifest migrated** from `pixi.toml` to `pyproject.toml` (issues #118, #119).
- Dependencies audited: core runtime dependencies restored to top-level `pyproject.toml`;
  remaining pixi-managed deps moved from PyPI to `conda-forge` for better cross-platform
  support (#131).
- `environment.yml` and `environment_working.yml` updated.
- `pixi.lock` regenerated.
- macOS compatibility fixes for pixi environment (#131).

---

### 👥 Contributors

| Contributor | Role |
|-------------|------|
| **Sarah Jordan** (@sjordan29) | Lead developer — architecture, transport engine, I/O, plotting, post-processing |
| **Anthony Aufdenkampe** (@aufdenkampe) | Pathing, configuration, demo integration, release management |
| **Paul Tomasula** (@ptomasula) | Dependency management, pyproject.toml migration |

---

## [v0.7.1] — 2025-04-16

**NSM Coupling Demo Updates** — Minor enhancements ahead of a planned scalability refactor.

### Added
- Wetted surface area calculation (PR #95, @sjordan29)
- Downstream boundary exploration (PR #101, @sjordan29)
- Gates support (PR #102, @sjordan29)
- Depth calculations (PR #105, @sjordan29)

---

## [v0.7.0] and earlier

See [GitHub Releases](https://github.com/EcohydrologyTeam/ClearWater-riverine/releases) for prior release notes.

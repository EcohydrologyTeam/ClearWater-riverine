User Guide
==========

This guide provides detailed information on using ClearWater-Riverine for water quality modeling.

Overview
--------

ClearWater-Riverine is a two-dimensional (2D) water quality transport model designed to:

* Calculate conservative advection and diffusion of constituents
* Work with unstructured grids of flows
* Model complex river systems and floodplains
* Couple with ClearWater modules for water quality processes

Key Features
------------

* **Modern Python Implementation** - Built using NumPy, SciPy, and other scientific Python libraries
* **Unstructured Grid Support** - Works with complex geometries from HEC-RAS 2D models
* **Module Integration** - Couples with TSM (Temperature) and NSM (Nutrient) simulation modules
* **Efficient Computation** - Uses sparse matrix operations for computational efficiency

Model Components
----------------

Transport Module
~~~~~~~~~~~~~~~~

The transport module handles advection and diffusion processes:

* Advection - Movement of constituents with flow
* Diffusion - Mixing and spreading of constituents
* Mass conservation - Ensures proper mass balance

Mesh Handling
~~~~~~~~~~~~~

ClearWater-Riverine uses unstructured meshes from HEC-RAS:

* Cell-centered values for constituents
* Face-based fluxes between cells
* Support for wetting and drying

Input/Output
~~~~~~~~~~~~

The model supports various input/output formats:

* HDF5 files for mesh geometry
* CSV files for initial and boundary conditions
* Parquet files for time series data
* HDF5 output for results storage

Coupling with Water Quality Modules
-----------------------------------

Temperature Simulation Module (TSM)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Models heat transport and temperature dynamics:

* Solar radiation
* Heat exchange with atmosphere
* Thermal stratification effects

Nutrient Simulation Module (NSM)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Simulates nutrient cycling and eutrophication:

* Nitrogen and phosphorus cycles
* Algae growth and decay
* Dissolved oxygen dynamics
* Organic matter decomposition

Best Practices
--------------

1. **Mesh Resolution** - Use appropriate mesh resolution to capture important features
2. **Time Steps** - Choose time steps that satisfy stability criteria
3. **Boundary Conditions** - Ensure boundary conditions are properly specified
4. **Initial Conditions** - Use realistic initial conditions for faster convergence
5. **Validation** - Compare results with observed data when available
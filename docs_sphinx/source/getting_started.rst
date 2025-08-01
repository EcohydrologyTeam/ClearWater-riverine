Getting Started
===============

After installation, you can start using ClearWater-Riverine by running our example Jupyter Notebooks.

Quick Start
-----------

1. Activate your conda environment:

   .. code-block:: shell

      conda activate ClearWater-modules

2. Launch JupyterLab:

   .. code-block:: shell

      jupyter lab

3. Navigate to the ``examples`` directory and open one of the tutorial notebooks.

Example Notebooks
-----------------

We provide several example notebooks to help you get started:

* **Ohio River.ipynb** - A comprehensive example showing E. Coli transport in the Ohio River
* **01_getting_started_riverine.ipynb** - Basic introduction to ClearWater-Riverine
* **02_coupling_riverine_modules_tsm.ipynb** - Coupling with Temperature Simulation Module (TSM)
* **03_01_coupling_riverine_modules_nsm.ipynb** - Coupling with Nutrient Simulation Module (NSM)
* **03_02_plot_coupled_nsm_simulation.ipynb** - Visualizing NSM simulation results

Basic Usage Example
-------------------

Here's a simple example of how to use ClearWater-Riverine:

.. code-block:: python

   import clearwater_riverine
   from clearwater_riverine.io import hdf, inputs, outputs
   from clearwater_riverine.transport import Transport
   
   # Load your HDF mesh file
   mesh = hdf.load_mesh('path/to/your/mesh.hdf')
   
   # Load initial conditions and boundary conditions
   initial_conditions = inputs.load_initial_conditions('path/to/initial_conditions.csv')
   boundary_conditions = inputs.load_boundary_conditions('path/to/boundary_conditions.csv')
   
   # Create and run transport model
   transport = Transport(mesh)
   results = transport.run(initial_conditions, boundary_conditions)
   
   # Save results
   outputs.save_results(results, 'output_path')
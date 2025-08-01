Examples
========

This section provides detailed examples of using ClearWater-Riverine for various water quality modeling scenarios.

Ohio River E. Coli Transport
----------------------------

The Ohio River example demonstrates modeling E. Coli transport following a contamination event:

* Sudden inflow of E. Coli at Covington
* Downstream transport and lateral spreading
* Comparison with EFDC model results
* Validation against observed data

See the ``examples/Ohio River.ipynb`` notebook for the complete analysis.

Simple Test Cases
-----------------

The repository includes several simple test cases for learning and validation:

* **Plan 01** - Basic 10x5 grid
* **Plan 02** - Simple 2x1 grid
* **Plan 03** - Alternative 2x1 configuration
* **Plan 04** - 10x5 grid with full boundary conditions
* **Plan 05** - Tidal flow with full boundaries
* **Plan 06** - Tidal flow with multiple boundaries
* **Plan 07** - Tidal flow with island
* **Plan 08** - Refined mesh with tidal flow and island

Temperature Modeling with TSM
-----------------------------

Example coupling with the Temperature Simulation Module:

.. code-block:: python

   from clearwater_riverine import Transport
   from clearwater_modules import TSM
   
   # Initialize transport and TSM
   transport = Transport(mesh)
   tsm = TSM()
   
   # Couple models
   for timestep in range(n_timesteps):
       # Transport heat
       temperature = transport.step(temperature, timestep)
       
       # Apply heat sources/sinks from TSM
       temperature = tsm.calculate_heat_flux(temperature, meteorology)

Nutrient Modeling with NSM
--------------------------

Example coupling with the Nutrient Simulation Module:

.. code-block:: python

   from clearwater_riverine import Transport
   from clearwater_modules import NSM
   
   # Initialize transport and NSM
   transport = Transport(mesh)
   nsm = NSM()
   
   # Define constituents
   constituents = ['nitrogen', 'phosphorus', 'algae', 'dissolved_oxygen']
   
   # Run coupled simulation
   for timestep in range(n_timesteps):
       # Transport constituents
       for constituent in constituents:
           values[constituent] = transport.step(values[constituent], timestep)
       
       # Apply nutrient kinetics
       values = nsm.update_kinetics(values, timestep)

Visualization Examples
----------------------

Creating animations of model results:

.. code-block:: python

   import matplotlib.pyplot as plt
   from matplotlib.animation import FuncAnimation
   
   # Create figure
   fig, ax = plt.subplots()
   
   # Animation function
   def animate(frame):
       ax.clear()
       mesh.plot_constituent(results[frame], ax=ax)
       ax.set_title(f'Time: {frame * dt} hours')
   
   # Create animation
   anim = FuncAnimation(fig, animate, frames=len(results))
   anim.save('transport_animation.gif')

Additional Resources
--------------------

* Example notebooks in ``examples/`` directory
* Test cases in ``tests/data/``
* Development sandbox in ``examples/dev_sandbox/``
Installation
============

ClearWater Riverine is designed to run with **Python 3.10**.

Follow these steps to install.

Install Miniconda or Anaconda Distribution
-------------------------------------------

We recommend installing the light-weight `Miniconda <https://docs.conda.io/projects/miniconda/en/latest/>`_ that includes Python, the `conda <https://conda.io/docs/>`_ environment and package management system, and their dependencies.

.. note::

   Follow conda defaults to install in your local user directory. DO NOT install for all users, to avoid substantial headaches with permissions.

If you have already installed the **Anaconda Distribution**, you can use it to complete the next steps, but you may need to `update to the latest version <https://docs.anaconda.com/free/anaconda/install/update-version/>`_.

Clone or Download the ClearWater-riverine Repository
----------------------------------------------------

From the `GitHub repository <https://github.com/EcohydrologyTeam/ClearWater-riverine>`_, click on the green "Code" dropdown button near the upper right. Select to either Open in GitHub Desktop (i.e. git clone) or "Download ZIP". We recommend using GitHub Desktop, to most easily receive updates.

Place your copy of this repo folder in any convenient location on your computer.

Create a Conda Environment
--------------------------

We recommend creating a custom virtual environment with the `Conda <https://conda.io/docs/>`_ package, dependency, and environment management for any language.

We provide an ``environment.yml`` file that specifies for Conda how to create a virtual environment that contains the same software dependencies that we've used in development and testing.

Create a ``ClearWater-modules`` environment using this conda command in your terminal or Anaconda Prompt console:

.. code-block:: shell

   conda env create --file environment.yml

Alternatively, use the faster ``libmamba`` solver with:

.. code-block:: shell

   conda env create -f environment.yml --solver=libmamba

.. note::

   If users are experiencing issues with plots NOT displaying in jupyter notebooks, use the ``environment_working.yml`` file:

   .. code-block:: shell

      conda env create -f environment_working.yml --solver=libmamba

Activate the environment using the instructions printed by conda after the environment is created successfully.

Add ClearWater-riverine to Python Path
---------------------------------------

To have access to the ``clearwater_riverine`` module in your Python environments, it is necessary to have a path to your copy of ClearWater Riverine in Anaconda's ``sites-packages`` directory.

The easiest way to do this is to use the ``conda develop`` command in the console or terminal:

.. code-block:: shell

   conda develop '/path/to/module/src'

Replace ``/path/to/module/src`` with the full file pathway to the local cloned ClearWater-riverine repository's ``src`` directory.

You should now be able to run the examples and create your own Jupyter Notebooks!
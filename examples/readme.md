# Guide to Examples

The `examples` directory contains Jupyter notebooks to teach users how to use the package. Notebooks designed as a tutorial will be numbered in the sequence they should be used (i.e. 1_Intro.ipynb, etc.).

Sub-directories are for supporting files:
- `examples/data` is for data files required to run the example notebooks.
- `examples/dev_sandbox` is for exploratory work by the development team as they develop new capabilities, including for informal testing.
- `examples/temp` is for temporary output files that users might create while running examples. 
  - This sub-directory will be in the `.gitignore`, so that output files are not saved to the repo.

## Data Availability
Some files containing hydrodynamic data that is fed into Clearwater Riverine is too large to store on Github, even with Large File Storage. We have provided data required to run the examples below at this [Google Drive](https://drive.google.com/drive/folders/1I_di8WrK95QwBga-W8iuJaJJMZsnIYSS?usp=drive_link).

## Example Notebooks
- [01_getting_started_riverine.ipynb](./01_getting_started_riverine.ipynb): Provides a basic example running Clearwater Riverine on a fictional, demonstrative location, "Sumwere Creek."
- [02_coupling_riverine_modules.ipynb](./02_coupling_riverine_modules.ipynb): Provides a more advanced example of coupling Clearwater Riverine and [Clearwater Modules](https://github.com/EcohydrologyTeam/ClearWater-modules) (reaction)'s Temperature Simulation Module (TSM) using a fictional, demonstrative location, "Sumwere Creek."
- [Ohio River.ipynb](./Ohio%20River.ipynb): Provides an example of running the Clearwater Riverine model on the Ohio River. This notebook compares Clearwater Riverine constituent transport to results on the same system from the Environmental Fluid Dynamics Code (EFDC) model. 
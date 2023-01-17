# Guide to Examples

The `examples` directory contains Jupyter notebooks to teach users how to use the package. Notebooks designed as a tutorial will be numbered in the sequence they should be used (i.e. 1_Intro.ipynb, etc.).

Sub-directories are for supporting files:
- `examples/data` is for data files required to run the example notebooks.
- `examples/dev_sandbox` is for exploratory work by the development team as they develop new capabilities, including for informal testing.
- `examples/temp` is for temporary output files that users might create while running examples. 
  - This sub-directory will be in the `.gitignore`, so that output files are not saved to the repo.

### Example Notebooks
- [Ohio River.ipynb](./Ohio%20River.ipynb): Provides an example of running the Clearwater Riverine model on the Ohio River. This notebook compares Clearwater Riverine constituent transport to results on the same system from the Environmental Fluid Dynamics Code (EFDC) model. 
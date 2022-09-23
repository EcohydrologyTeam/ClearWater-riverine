# Guide to Examples

The `examples` directory contains Jupyter notebooks to teach users how to use the package. Notebooks designed as a tutorial will be numbered in the sequence they should be used (i.e. 1_Intro.ipynb, etc.).

Sub-directories are for supporting files:
- `examples/data` is for data files required to run the example notebooks.
- `examples/dev_sandbox` is for exploratory work by the development team as they develop new capabilities, including for informal testing.
- `examples/temp` is for temporary output files that users might create while running examples. 
  - This sub-directory will be in the `.gitignore`, so that output files are not saved to the repo.

### Example Notebooks
- [testing.ipynb](testing.ipynb): Uses the `ras2dwq.py` to set up a mesh and run the a very simple water quality model in a 9-cell "box".
- [Ohio River.ipynb](Ohio River.ipynb): Provides an example of running a more complicated test case.
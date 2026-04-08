# Guide to Examples

The `examples` directory contains Jupyter notebooks to teach users how to use the package. Notebooks designed as a tutorial will be numbered in the sequence they should be used (i.e. 1_Intro.ipynb, etc.).

Sub-directories are for supporting files:
- `examples/data_temp` is for data files required to run the example notebooks. This folder is in the `.gitignore` since some files are too large to be managed with Github. See the [Data Availability](#data-availability) section.
- `examples/dev_sandbox` is for exploratory work by the development team as they develop new capabilities, including for informal testing.
- `examples/temp` is for temporary output files that users might create while running examples. 
  - This sub-directory will be in the `.gitignore`, so that output files are not saved to the repo.

## Data Availability
Some files containing hydrodynamic data that is fed into Clearwater Riverine is too large to store on Github, even with Large File Storage. We have provided data required to run the examples below at this [Google Drive](https://drive.google.com/drive/folders/19uCjAJPZh4g6r1ZWzk1D_B8jZGluSc4N?usp=drive_link).

## Example Notebooks
- [01_getting_started_riverine.ipynb](./01_getting_started_riverine.ipynb): Provides a basic example running Clearwater Riverine on a fictional, demonstrative location, "Sumwere Creek."

## Additional Examples
See the [ClearWater-modules](https://github.com/EcohydrologyTeam/ClearWater-modules/tree/main/examples) examples to see how ClearWater-Riverine can be linked with ClearWater-Modules. 
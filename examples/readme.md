# Guide to Examples Repo

**General Exploration**:
* [HDF_Exploration.ipynb](HDF_Exploration.ipynb): Explores data stored in RAS HDF output. 
* [File_Conversion.ipynb](File_Conversion.ipynb): Includes detailed information about UGRID-compliance for polotting 2d flexible meshes. 
* [compute_cell_volume_and_face_area.ipynb](compute_cell_volume_and_face_area.ipynb)
* [compute_distances_between_cell_centers.ipynb](compute_cell_volume_and_face_area.ipynb)
* [advection_diffusion.ipynb](advection_diffusion.ipynb)

**Application**: 
* [Box-Model.ipynb](Box-Model.ipynb): Explores RAS HDF output and geometry for a simple 3x3 box and stores necessary information in an UGRID-compliant xarray (using guidelines outlined in more detail in File_Conversion.ipynb). This includes calculation of relevant coefficients for the advection-diffusion equations. The notebook saves the xarray to a .zarr file. 
* [Sparse_Matrix_Framework.ipynb](Sparse_Matrix_Framework.ipynb): Sets up the framework for solving each timestep using the sparse matrix framework. Solves the box model set up in Box-Model.ipynb.
* [Sparse_Matrix_Framework-muncie.ipynb](Sparse_Matrix_Framework-muncie.ipynb): the script above worked for the box model, where all cells are wet. However, when we attempted to run this on a model with dry cells (the muncie model), it could not solve because the matrix was singluar (rows of all 0). This notebook consists of exploration of trying to figure out and resolve that issue. 


**Testing Module**:
* [testing.ipynb](testing.ipynb): Uses the `ras2dwq.py` to set up a mesh and run the water quality model.


**Plotting**:
* [plotting.ipynb](plotting.ipynb): plays around with different plotting options from Holoviz, including Geoviews and Datashader.

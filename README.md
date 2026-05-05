# ClearWater-Riverine

The [ClearWater-riverine]([url](https://github.com/EcohydrologyTeam/ClearWater-riverine)) package is a two-dimensional (2D) water quality transporter model to calculate conservative advection and diffusion of constituents from an unstructured grid of flows within complex river systems and floodplains. It is developed with modern Python by the the [U.S. Army Engineer Research and Development Center (ERDC)](https://www.erdc.usace.army.mil), [Environmental Laboratory (EL)](https://www.erdc.usace.army.mil/Locations/EL/). 

The goal of this model is to simulate the transport (advection and diffusion) of heat and water quality constituents in riverine systems by coupling it to ERDC's [ClearWater (Corps Library for Environmental Analysis and Restoration of Watersheds) modules](https://github.com/EcohydrologyTeam/ClearWater-modules) that simulates water quality processes and kinetics. At present, the Temperature Simulation Module (TSM) and Nutrient Simulation Module (NSM) have been successfully coupled to HEC-RAS-2D models via ClearWater-Riverine, simulating fundamental eutrophication processes such as the interactions between temperature, nutrients, algae, dissolved oxygen, and organic matter. ClearWater-Riverine assumes vertical homogeneity. Therefore, it is best suited for evaluating riverine systems during conditions where vertical stratification does not contribute significantly to the water quality dynamics, but where the longitudinal and lateral changes of water quality are important.

A secondary goal is to develop a suite of easy-to-use modern Python tools that build on community-developed scientific workflows, standards, and libraries to automate model setup, prepare input datasets, store output data, and visualize results using Python-based user interfaces such as Jupyter Notebooks.

## Example applications

The following plot shows an animation of E. Coli transport in the Ohio River in June, 2010. A sudden inflow of E. Coli enters the Ohio River at Covington on the south shore of the river. The downstream flow and lateral spread of E. Coli over time is due to the transport and mixing processes (advection-diffusion) in the river. 

![ClearWater-Riverine animation of E. Coli transport in the Ohio River](docs/gifs/ClearWater-Riverine-and-EFDC-Ohio.gif)

ClearWater-Riverine performance was compared to an existing EFDC model of the Ohio River, and both models were verified with observed data. These comparisons verified that ClearWater-Riverine is accurately capaturing the transport processes in this system. A side-by-side comparison of the two models is shown below.

![Comparison of ClearWater-Riverine and EFDC model performance for simulating E. Coli transport in the Ohio River](docs/gifs/ClearWater-Riverine-Ohio.gif)

## Repository Directories

**[`src`](src)** contains the source code to create and run the `clearwater_riverine`.

**[`examples`](examples)** contains tutorials and useful Juptyer Notebooks.

**[`docs`](docs)** contains relevant reference documentation.

**[`tests`](tests)** will contain clearwater_riverine tests once they are developed. 

## Getting Started

### Installation

Follow these steps to install the ClearWater Modeling System and its dependancies in a custom Python environment.

We recommend using [pixi](https://pixi.prefix.dev/latest/), the next-generation reproducible package management tool built on [conda](https://docs.conda.io/projects/conda/en/stable/) tooling.

If you are new to pixi but familiar with conda, this [Switching from Conda](https://pixi.prefix.dev/latest/switching_from/conda/) documentation succinctly compares similarities and differences.

Alternately, use a conda environment with the same dependencies.

#### 1. Install Pixi

Follow [Pixi Installation](https://pixi.prefix.dev/latest/installation/) instructions for your platform.

#### 2. Clone or Download the ClearWater family of repositories

There are three repositories which house the dependencies for this project. Navigate to each of the repositories listed below and follow the instructions to clone them to your local machine.

- [ClearWater-riverine](https://github.com/EcohydrologyTeam/ClearWater-riverine) transport process simulator
- [ClearWater-data](https://github.com/EcohydrologyTeam/ClearWater-data) perforamant data access and storage protocols
- [ClearWater-modules](https://github.com/EcohydrologyTeam/ClearWater-modules) (optional) water quality reaction process simulator

From these Github sites, click on the green "Code" dropdown button near the upper right. Select to either "Open in GitHub Desktop" (i.e. git clone) or "Download ZIP". 

We recommend using GitHub Desktop, to most easily manage git workflows by providing excellent visuals for stagging commits, exploring commit histories, comparing branches, and resolving merge conflicts in tight integration with Visual Studio Code.

Place your copy of these repos in any convenient location on your computer. Make sure that all are stored in the same directory OR you will need to update the `pyproject.toml` so that the `[pixi.tool.feature.dev.pypi-dependencies]` point to the correct location of your local clones.

### 3. Create Clearwater Workspace and Python Environments 

Create a project-specific Pixi workspace and Python enviornment(s) from the `pyproject.toml` manifest file.

#### Production 

Pardon our mess! Production instructions coming soon. For now you can install using the [Developer](#developer) instructions. 

#### Developers

Developer instructions install all ClearWater repos in "editable" mode in a second `dev` environment.

From your terminal or console, navigate to the directory of your `Clearwater-riverine` clone and execute the following command to create a `dev` environment:

```shell
pixi install -e dev
```

To activate this environment in your shell, run the following:

```shell
pixi shell -e dev
```

You should now be able to run the examples and create your own Jupyter Notebooks!

### Examples

Try running our [01_getting_started_riverine.ipynb](examples/01_getting_started_riverine.ipynb) Jupyter Notebook.


## Contributing

We welcome your pull request.

## Acknowledgements

This library is developed by ERDC-EL through funding from the ECOMOD project.
Dr. Todd E. Steissberg (ERDC-EL) developed the vision for this library as an example of how to couple at water transport model with a water quality reaction model.


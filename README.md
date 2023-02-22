# ClearWater Riverine
ClearWater Riverine is a library containing a water quality engine that computes the transport (advection and diffusion) of constituents in riverine systems, leveraging hydrodynamic output from HEC-RAS-2D models. ClearWater is developed by the Environmental Laboratory, U.S Army Engineer Research and Development Center (ERDC).

## Repository Directories

**[src](src)** contains the source code to create and run the clearwater_riverine.

**[examples](examples)** contains tutorials and useful Juptyer Notebooks.

**[docs](docs)** contains relevant reference documentation.

**[tests](tests)** will contain clearwater_riverine tests once they are developed. 

## Installation

Clearwater Riverine is designed to run with **Python 3.10**. 

Follow these steps to install.

#### 1. Install the Anaconda Python Distribution

We recommend installing the [latest release](https://docs.anaconda.com/anaconda/reference/release-notes/) of [**Anaconda Individual Edition**](https://www.anaconda.com/distribution). Follow their [installation](https://docs.anaconda.com/anaconda/install/) documentation.

#### 2. Clone or Download this Clearwater-riverine repository

From this Github site, click on the green "Code" dropdown button near the upper right. Select to either Open in GitHub Desktop (i.e. git clone) or "Download ZIP". We recommend using GitHub Desktop, to most easily receive updates.

Place your copy of the Clearwater-riverine folder in any convenient location on your computer.

#### 3. Create a Conda Environment for Clearwater Riverine Modeling 

We have provided an [`environment.yml`](environment.yml) file, which lists all primary dependencies, to help. Create a `clearwater_riverine` environment either with the **Import** button on [Anaconda Navigator's Environments tab](https://docs.anaconda.com/anaconda/navigator/overview/#environments-tab), or use this [Conda](https://conda.io/docs/) command in your terminal or console,  replacing `path/environment.yml` with the full file pathway to the `environment.yml` file in the local cloned repository.

```shell
conda env create --file path/environment.yml
```
To update your environment, either use Anaconda Navigator, or run the following command:

```shell
conda env update --file path/environment.yml --prune
```

or

```shell
conda env create --file path/environment.yml --force
```


#### 4. Add your Clearwater Riverine Path to Anaconda sites-packages

To have access to the `clearwater_riverine` module in your Python environments,
it is necessary to have a path to your copy of Clearwater Riverine in Anaconda's `sites-packages` directory (i.e. something like `$HOME/path/to/anaconda/lib/pythonX.X/site-packages` or `$HOME/path/to/anaconda/lib/site-packages` similar).

The easiest way to do this is to use the [conda develop](https://docs.conda.io/projects/conda-build/en/latest/resources/commands/conda-develop.html) command in the console or terminal like this, replacing `/path/to/module/` with the full file pathway to the local cloned Clearwater-riverine repository:

```console
conda-develop /path/to/module/
```

You should now be able to run the examples and create your own Jupyter Notebooks!


## Getting Started

We recommend viewing or interactively running our [Ohio River](examples/Ohio%20River.ipynb) Jupyter Notebook.

We recommend using [JupyterLab](https://jupyterlab.readthedocs.io/en/stable/) to run our tutorial [Juptyer Notebooks](https://jupyter.org/index.html) in the [example](examples) folder, due to many additional built-in features and extensions. The following JupyterLab [extensions](https://jupyterlab.readthedocs.io/en/stable/user/extensions.html) are particularly useful:
- [lckr-jupyterlab-variableinspector](https://github.com/lckr/jupyterlab-variableInspector)

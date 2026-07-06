# nist_cdsaxs
The purpose of the 'nist_cdsaxs' software is to enable reduction of 
transmission critical-dimension small-angle X-ray scattering (CD-SAXS) 
data. It converts the scattering vector, q, from beam- or 
detector-based coordinates to sample-based coordinates. 

Reduced CD-SAXS data slices, i.e. 1D scattering intensity 
as a function of the z component of q (z axis is along sample depth) can 
be exported and used directly with the 
[`nist_cdsaxs_analysis`](https://github.com/usnistgov/nist_cdsaxs_analysis)
repository.

## Installation

The package follows a standard Python layout (`src/` with `pyproject.toml`). You can install it with pip.

Basic install (recommended):

```bash
pip install --upgrade pip setuptools wheel
pip install .
```

Editable/developer install with optional dev tools:

```bash
pip install --upgrade pip setuptools wheel
pip install -e .[dev]
```

> [!NOTE]
> If you use conda/mamba, create and activate an environment first, then run the pip commands above.
>
> The `-e` (editable) install lets you modify the source and use the changes without re-installing.

## Create and Activate Environment
1. Install Python with the conda package manager on your machine if it is not already installed. One open source option is [Miniforge](https://github.com/conda-forge/miniforge).
2. Open a terminal (e.g., GitBash, Command Prompt on Windows, Terminal on Mac) and navigate to the root folder of the project using the `cd` command.
3. Create a Python environment; replace 'cdsaxs_dev' with another environment name if you wish:
```
conda create --name cdsaxs_dev python
```
> [!IMPORTANT]
> If you're using a Jupyter notebook, use `conda create --name cdsaxs_dev python ipykernel` to install ipykernel as well. Installation with pip did not work in our testing.
> 

4. Activate the environment, replacing `cdsaxs_dev` with your environment name if it is different.
```
conda activate cdsaxs_dev
```
After activating the development environment, install the repo in editable mode:

```
pip install -e .[dev]
```

This is preferred over manually modifying `PYTHONPATH`.

## Usage
An example Jupyter notebook has been provided in `examples`. It outlines  
the required reduction steps as well as some of the additional 
functionality included in this software. Please follow this notebook 
and the included descriptions carefully as some steps need to be 
performed in order.

To run the notebook via Jupyter lab in a browser:
1. Ensure that `nist_cdsaxs` is installed following the directions above.
2. In the terminal, navigate to the `exampls` folder in this repository.
3. Activate the Python environment in which nist_cdsaxs is installed. 
For example, if the environment is called cdsaxs_dev, this command would 
be:
```
conda activate cdsaxs_dev
```
4. Launch jupyter lab with the following command:
```
jupyter lab
```
5. Your browser should open up Jupyter lab autmoatically and the 
available notebooks will be shown in the file browser on the left of 
the screen. If you browser did not open, you may need to type in the 
address provided in the terminal after you run step 4.

[!NOTE]
You can also run the Jupyter notebook from any compatible IDE, such as
VS Code. Click on the file and make sure to select the appropriate 
Python environment to run the notebook.


## CD-SAXS Coordinate Conventions
All data imported into this software should follow the below coordinate
conventions. This may require assigning metadata named according to the
beamline's coordinate conventions appropriately to this software's
naming convention. Image transformations, such as rotations or flips, 
may also be required to ensure the data is processed correctly. While 
some of the beamline-specific data loaders have made an attempt to do
this upon import, we encourage the user to confirm data is loaded 
correctly as beamline conventions and data export formats can change 
over time.

## Citation Information
_More information about citation information will be available soon._

## Contact Information
You can communicate with the `nist_cdsaxs` owners via e-mail:   
Daniel Sunday daniel.sunday@nist.gov  
Joseph Kline r.kline@nist.gov  

[National Institute of Standards and Technology](http://www.nist.gov)   
[Material Measurement Laboratory](https://www.nist.gov/mml)   
[Materials Science and Engineering Division](https://www.nist.gov/mml/materials-science-and-engineering-division)   
[Polymers Processing Group](https://www.nist.gov/mml/materials-science-and-engineering-division/polymers-processing-group)   

## Contributors
Dean M. DeLongchamp  
Joseph Kline  
Logan Magaha  
Daniel Sunday  
Matthew A. Wade  
Caitlyn M. Wolf  

_Contributors are listed in alphabetical order by last name._


## Related Material
[Metrology for Nanolithography Project](https://www.nist.gov/programs-projects/metrology-nanolithography)

## Disclaimers

A portion of this repository, including source code and documentation, 
was written with the assistance of artificial intelligence (AI). All  
code and text generated by the AI models has been reviewed and tested by 
a human developer. 

Certain commercial or open-source software may be identified in this project to foster understanding. Such identification does not imply recommendation or endorsement by the National Institute of Standards and Technology, nor does it imply that the software identified are necessarily the best available for the purpose.

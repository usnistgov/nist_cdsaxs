# nist_cdsaxs
_Insert a description of our software, including a statement of purpose and maturity and description of repo contents._

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

Notes:
- If you use conda/mamba, create and activate an environment first, then run the pip commands above.
- The `-e` (editable) install lets you modify the source and use the changes without re-installing.

## Legay GUI
The legacy GUI for CD-SAXS data reduction is no longer supported in this package.
It can be found at the following location: https://github.com/usnistgov/nist_cdsaxs_legacy_gui

## Installation & Usage Instructions (Developers)
1. Install Python with the conda package manager on your machine if it is not already installed. One open source option is [Miniforge](https://github.com/conda-forge/miniforge).
2. Open a terminal (e.g., GitBash, Command Prompt on Windows, Terminal on Mac) and navigate to the root folder of the project using the `cd` command.
3. Create an environment with the `environment_legacy.yml` file:
```
conda env create -f environment_dev.yml
```
4. Activate the environment, replacing `cdsaxs_legacy` with your environment name if it is different.
```
conda activate cdsaxs_dev
```
After activating the development environment, install the repo in editable mode:

```
pip install -e .[dev]
```

This is preferred over manually modifying `PYTHONPATH`.

## Citation Information
_Insert information here about how to cite this software._

## Contact Information
You can communicate with the `nist_cdsaxs` owners via e-mail:   
Daniel Sunday daniel.sunday@nist.gov  
Joseph Kline r.kline@nist.gov  

[National Institute of Standards and Technology](http://www.nist.gov)   
[Material Measurement Laboratory](https://www.nist.gov/mml)   
[Materials Science and Engineering Division](https://www.nist.gov/mml/materials-science-and-engineering-division)   
[Polymers Processing Group](https://www.nist.gov/mml/materials-science-and-engineering-division/polymers-processing-group)   

## Related Material
[Metrology for Nanolithography Project](https://www.nist.gov/programs-projects/metrology-nanolithography)

## Disclaimer
Certain commercial or open-source software may be identified in this project to foster understanding. Such identification does not imply recommendation or endorsement by the National Institute of Standards and Technology, nor does it imply that the software identified are necessarily the best available for the purpose.

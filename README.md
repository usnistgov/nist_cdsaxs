# nist_cdsaxs
_Insert a description of our software, including a statement of purpose and maturity and description of repo contents._

## Installation Instructions
_Insert basic installation instructions here._

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

To access the developers version of the code until proper installation is enabled,
you may need to add the 'cdsaxs/src' directory to your python path.

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

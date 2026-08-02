# Data from "A ModEx Framework for Watershed Subsurface Investigation With Limited Geophysical Data Using Machine Learning and Hydrologic Modeling"

This repository contains the data and code associated with the paper titled *A ModEx Framework for Watershed Subsurface Investigation With Limited Geophysical Data Using Machine Learning and Hydrologic Modeling*.

## Data Reference

Data was developed on GitHub and published on ESS-DIVE for long term preservation. The files on GitHub and ESS-DIVE are up-to-date. Per the Watershed Function SFA's data usage policy, we ask that you cite this data from the ESS-DIVE repository.

Cite this data as follows:

>  Chen, H. (2025). *Machine learning–upscaled catchment resistivity model and ParFlow-CLM hydrologic simulations for the Trail Creek Catchment, East-Taylor River Watershed, Colorado* [Dataset]. Watershed Function SFA, ESS-DIVE repository. https://doi.org/10.15485/3012485

## Manuscript Reference

This repository contains the data and code associated with the paper:

> Chen, H., Thibaut, R., Chou, C., Xiong, C., & Wu, Y. (2026). A ModEx Framework for Watershed Subsurface Investigation With Limited Geophysical Data Using Machine Learning and Hydrologic Modeling. *Geophysical Research Letters*, 53(2). https://doi.org/10.1029/2025GL119953

## Contents

- **Res_predict.ipynb**: Trains a `RandomForestWithUncertainty` model to predict subsurface electrical resistivity from topographic attributes (elevation, slope, aspect, plan curvature) at 12 depth layers (0.5–5.0 m). Uncertainty is estimated by combining ensemble variance across trees with a nearest-neighbor local variance term. 

- **parflow.ipynb**: ParFlow-CLM hydrologic model setup and configuration for integrated surface–subsurface flow simulation over the Trail Creek Catchment (322 × 205 × 14 grid cells).

- **ModEx_Framework_Analysis.ipynb**: Core ModEx analysis pipeline. Loads resistivity predictions and three-scenario ParFlow ensemble saturation outputs, fits Archie's law (log ρ = a·log S + b) point-by-point via ODR, computes the Investigation Interest Index (III) from geophysical and hydrologic variability, delineates hydrogeologic zones via K-means clustering, and maps priority survey locations by overlaying uncertainty tiers on the zonation and III outputs.

- **terrain_attributes_clean.csv**: Topographic attributes (elevation, slope, aspect, plan curvature) derived from LiDAR digital elevation models. This is the input to the resistivity prediction workflow.

- **terrain_with_resistivity_predictions_cleaned.csv**: Full-grid predictions including resistivity, uncertainty standard deviation, coefficient of variation for each depth layer

- **test2/, test4/, test5/**: ParFlow‐CLM simulation outputs (saturation, pressure, porosity, permeability, mask files in `.pfb` format) for three model scenarios (S1, S2, S3) used to quantify subsurface uncertainty.


## Usage

1. **Resistivity Prediction**: Run `Res_predict.ipynb` to train the Random Forest model on EMI survey points and predict resistivity with uncertainty across the full spatial grid. The output `terrain_with_resistivity_predictions_cleaned.csv` includes the catchment index and is used by all downstream notebooks.

2. **Hydrologic Modeling**: Use `parflow.ipynb` to configure and run ParFlow‐CLM simulations with EMI-informed subsurface parameterization.

3. **ModEx Analysis**: Run `ModEx_Framework_Analysis.ipynb` to perform the integrated geophysical–hydrologic analysis, including Archie's law fitting, watershed zonation, investigation interest mapping, and priority survey location identification.


## References

This work uses the following source data sets:

- **EMI data:** Thibaut, R. (2026). *Electromagnetic Induction (EMI) Data, 2024, Trail Creek, Colorado* [Dataset]. Watershed Function SFA, ESS-DIVE repository. https://doi.org/10.15485/3364094
- **LiDAR-derived products (terrain attributes):** Falco, N., et al. (2024). *LiDAR-derived products, East River, CO* [Dataset]. Watershed Function SFA, ESS-DIVE repository. https://doi.org/10.15485/1602034
- **Hydrologic model (ParFlow-CLM v3.13.0):** parflow/parflow (2024). *ParFlow Version 3.13.0* [Software]. Zenodo. https://doi.org/10.5281/zenodo.10989198
- **Meteorological data:** NOAA National Operational Hydrologic Remote Sensing Center. https://ftp.nohrsc.noaa.gov/interactive/html/map.html
- **Streamflow data:** U.S. Geological Survey, National Water Information System, monitoring location USGS-09106800. https://waterdata.usgs.gov/monitoring-location/USGS-09106800


## Contact and Corresponding Author

Hang Chen  
School of Earth, Environment, and Sustainability, University of Iowa 
Email: hang-chen-1@uiowa.edu

## Acknowledgements

This work was supported by the Watershed Function Science Focus Area at Lawrence Berkeley National Laboratory funded by the U.S. Department of Energy, Office of Science, Biological and Environmental Research under Contract No. DE-AC02-05CH11231.

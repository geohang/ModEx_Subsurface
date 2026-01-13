<<<<<<< HEAD
# ModEx Framework for Subsurface Characterization

A Model-Experiment (ModEx) framework for integrating machine learning-based geophysical predictions with hydrologic modeling to guide optimal subsurface survey design. This code accompanies the paper on subsurface characterization at Trail Creek, Idaho.

## Overview

This repository implements a workflow that:
1. **Predicts subsurface resistivity** from terrain attributes using Random Forest with uncertainty quantification
2. **Integrates geophysical predictions with ParFlow hydrologic model simulations** to derive saturation data
3. **Calculates Archie's law parameters** relating resistivity to saturation using Orthogonal Distance Regression (ODR)
4. **Identifies high-priority survey locations** using an Investigation Interest Index (III)

## Study Site

Trail Creek catchment- a mountainous headwater catchment where subsurface characterization is critical for understanding water storage and flow dynamics.

## Repository Contents

### Jupyter Notebooks

| Notebook | Description |
|----------|-------------|
| `Res_predict.ipynb` | Main resistivity prediction using Random Forest with uncertainty quantification. Predicts resistivity at multiple depth layers (0.5m - 5.0m) from terrain attributes (elevation, slope, aspect, curvature). |
| `Res_predict_validation.ipynb` | Model validation using Random K-Fold Cross-Validation and bootstrap confidence intervals for R², RMSE, and Nash-Sutcliffe Efficiency (NSE). |
| `ModEx_Framework_Analysis.ipynb` | Core ModEx framework implementation: integrates resistivity predictions with ParFlow saturation data, calculates Archie's law parameters, computes Investigation Interest Index (III), and performs catchment zonation via K-means clustering. |
| `plot_uncertainty_CV.ipynb` | Visualizes resistivity uncertainty using Coefficient of Variation (CV) across depth layers. |
| `parflow.ipynb` | ParFlow hydrologic model setup and configuration. Defines computational grid, subsurface layering (14 variable-thickness layers), permeability fields for 6 soil/rock units, topographic slopes, and boundary conditions for the Trail Creek domain. |
| `properites.ipynb` | Exploratory analysis of ParFlow output properties (porosity, permeability). |

### Data Files

| File | Description |
|------|-------------|
| `terrain_attributes_clean.csv` | Input terrain attributes (x, y, elevation, slope, aspect, plan_curvature) with measured resistivity at 12 depth layers |
| `terrain_with_predicted_resistivity.csv` | Output file with predicted resistivity values and uncertainties |

### ParFlow Simulation Data

Three ensemble members with different subsurface parameterizations:
- `test2/` - Saturation, pressure, porosity, and permeability outputs
- `test4/` - Saturation and pressure outputs  
- `test5/` - Saturation, pressure, and porosity outputs

Each folder contains ParFlow Binary Format (`.pfb`) files for:
- Saturation fields (`*.satur.*.pfb`)
- Pressure fields (`*.press.*.pfb`)
- Porosity (`*.porosity.pfb`)
- Permeability (`*.perm_x.pfb`)
- Domain mask (`*.mask.pfb`)

## Methodology

### 1. Resistivity Prediction with Uncertainty
- **Model**: Random Forest Regressor with 100 trees
- **Features**: Terrain attributes (elevation, slope, aspect, plan curvature)
- **Target**: Log-transformed resistivity at each depth layer
- **Uncertainty**: Combined ensemble variance and nearest-neighbor methods

### 2. Archie's Law Parameter Estimation
Relates electrical resistivity (ρ) to saturation (S):
```
log(ρ) = a·log(S) + b
```
Parameters estimated using ODR with uncertainty propagation.

### 3. Investigation Interest Index (III)
Identifies optimal survey locations by combining:
- **Geophysical uncertainty**: CV of resistivity predictions
- **Hydrologic variability**: CV of saturation across ensemble members
- **Model fit quality**: Archie's law correlation coefficient

```
III = Normalize[(CV_ρ + CV_S) + (1-|r|)] × 100
```

### 4. Catchment Zonation
K-means clustering (k=3) on terrain attributes and resistivity layers to identify hydrogeologically similar zones.

## Requirements

```
numpy
pandas
matplotlib
scikit-learn
scipy
parflow-tools
```

## Usage

1. **Run resistivity prediction**:
   ```
   Open and execute Res_predict.ipynb
   ```

2. **Validate model performance**:
   ```
   Open and execute Res_predict_validation.ipynb
   ```

3. **Run full ModEx framework analysis**:
   ```
   Open and execute ModEx_Framework_Analysis.ipynb
   ```

## Key Outputs

- **Predicted resistivity maps** at 12 depth layers with uncertainty bounds
- **Archie's law parameter maps** showing spatial variability of rock-water relationships
- **Investigation Interest Index map** highlighting priority locations for future surveys
- **Catchment zonation map** delineating hydrogeologically distinct zones

## Citation

If using this code, please cite the associated paper:


## License

See LICENSE file for details.

## Contact

For questions about this code, please open an issue or contact the repository maintainers.
=======
# ModEx Framework for Subsurface Characterization

A Model-Experiment (ModEx) framework for integrating machine learning-based geophysical predictions with hydrologic modeling to guide optimal subsurface survey design. This code accompanies the paper on subsurface characterization at Trail Creek, Idaho.

## Overview

This repository implements a workflow that:
1. **Predicts subsurface resistivity** from terrain attributes using Random Forest with uncertainty quantification
2. **Integrates geophysical predictions with ParFlow hydrologic model simulations** to derive saturation data
3. **Calculates Archie's law parameters** relating resistivity to saturation using Orthogonal Distance Regression (ODR)
4. **Identifies high-priority survey locations** using an Investigation Interest Index (III)

## Study Site

Trail Creek catchment- a mountainous headwater catchment where subsurface characterization is critical for understanding water storage and flow dynamics.

## Repository Contents

### Jupyter Notebooks

| Notebook | Description |
|----------|-------------|
| `Res_predict.ipynb` | Main resistivity prediction using Random Forest with uncertainty quantification. Predicts resistivity at multiple depth layers (0.5m - 5.0m) from terrain attributes (elevation, slope, aspect, curvature). |
| `Res_predict_validation.ipynb` | Model validation using Random K-Fold Cross-Validation and bootstrap confidence intervals for R², RMSE, and Nash-Sutcliffe Efficiency (NSE). |
| `ModEx_Framework_Analysis.ipynb` | Core ModEx framework implementation: integrates resistivity predictions with ParFlow saturation data, calculates Archie's law parameters, computes Investigation Interest Index (III), and performs catchment zonation via K-means clustering. |
| `plot_uncertainty_CV.ipynb` | Visualizes resistivity uncertainty using Coefficient of Variation (CV) across depth layers. |
| `parflow.ipynb` | ParFlow hydrologic model setup and configuration. Defines computational grid, subsurface layering (14 variable-thickness layers), permeability fields for 6 soil/rock units, topographic slopes, and boundary conditions for the Trail Creek domain. |
| `properites.ipynb` | Exploratory analysis of ParFlow output properties (porosity, permeability). |

### Data Files

| File | Description |
|------|-------------|
| `terrain_attributes_clean.csv` | Input terrain attributes (x, y, elevation, slope, aspect, plan_curvature) with measured resistivity at 12 depth layers |
| `terrain_with_predicted_resistivity.csv` | Output file with predicted resistivity values and uncertainties |

### ParFlow Simulation Data

Three ensemble members with different subsurface parameterizations:
- `test2/` - Saturation, pressure, porosity, and permeability outputs
- `test4/` - Saturation and pressure outputs  
- `test5/` - Saturation, pressure, and porosity outputs

Each folder contains ParFlow Binary Format (`.pfb`) files for:
- Saturation fields (`*.satur.*.pfb`)
- Pressure fields (`*.press.*.pfb`)
- Porosity (`*.porosity.pfb`)
- Permeability (`*.perm_x.pfb`)
- Domain mask (`*.mask.pfb`)

## Methodology

### 1. Resistivity Prediction with Uncertainty
- **Model**: Random Forest Regressor with 100 trees
- **Features**: Terrain attributes (elevation, slope, aspect, plan curvature)
- **Target**: Log-transformed resistivity at each depth layer
- **Uncertainty**: Combined ensemble variance and nearest-neighbor methods

### 2. Archie's Law Parameter Estimation
Relates electrical resistivity (ρ) to saturation (S):
```
log(ρ) = a·log(S) + b
```
Parameters estimated using ODR with uncertainty propagation.

### 3. Investigation Interest Index (III)
Identifies optimal survey locations by combining:
- **Geophysical uncertainty**: CV of resistivity predictions
- **Hydrologic variability**: CV of saturation across ensemble members
- **Model fit quality**: Archie's law correlation coefficient

```
III = Normalize[(CV_ρ + CV_S) + (1-|r|)] × 100
```

### 4. Catchment Zonation
K-means clustering (k=3) on terrain attributes and resistivity layers to identify hydrogeologically similar zones.

## Requirements

```
numpy
pandas
matplotlib
scikit-learn
scipy
parflow-tools
```

## Usage

1. **Run resistivity prediction**:
   ```
   Open and execute Res_predict.ipynb
   ```

2. **Validate model performance**:
   ```
   Open and execute Res_predict_validation.ipynb
   ```

3. **Run full ModEx framework analysis**:
   ```
   Open and execute ModEx_Framework_Analysis.ipynb
   ```

## Key Outputs

- **Predicted resistivity maps** at 12 depth layers with uncertainty bounds
- **Archie's law parameter maps** showing spatial variability of rock-water relationships
- **Investigation Interest Index map** highlighting priority locations for future surveys
- **Catchment zonation map** delineating hydrogeologically distinct zones

## Citation

If using this code, please cite the associated paper:


## License

See LICENSE file for details.

## Contact

For questions about this code, please open an issue or contact the repository maintainers.
>>>>>>> 2de18126f533a94b2fa68437ecec473e72cdfc67

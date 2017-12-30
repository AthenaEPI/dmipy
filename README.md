# MIPY: Microstructure Imaging in Python

The Mipy software package facilitates state-of-the-art diffusion MRI Microstructure Imaging using a "Building Block" philosophy. In this philosophy, any combination of biophysical models, typically representing intra- and/or extra-axonal tissue compartments, can be easy combined into a multi-compartment Microstructure Model in just a few lines of code. The created model can be used to both simulate and fit data for any PGSE-based dMRI acquisition, including single shell, multi-shell and qtau-dMRI acquisition schemes.

Model fitting algorithms are also implemented modularly. Currently, any created model can be fitted using either gradient-descent based fitting, or more recent Microstructure Imaging of Crossings (MIX) fitting.

Mipy stands apart from other packages is the complete freedom the user has to design and use a microstructure model. Mipy allows the user to do Microstructure Imaging research at the highest level, while the package automatically takes care of all the coding architecture that is needed to fit a designed model to a data set.

*For Dipy users*: Mipy is completely compatible, complementary and similar in usage to Dipy!
- Dipy gradient tables can be directly used when setting up Mipy models.
- Dipy models can be used to give initial conditions for Mipy models, e.g. using DTI or CSD to give a starting point for orientations or diffusivities for NODDI or NODDIx.
- Setting up and fitting Mipy models is similar as in Dipy models - using fit=model.fit(data) and then exploring the fitted model parameters from the fit object.
- Fitted models that estimate dispersion/orientations can generate Fiber Orientation Densities (FODs) and peaks for every fitted voxel in the same format as Dipy, which can be directly used for fiber tractography!
- Fitted models also provide easy functions to study fitting errors in terms of MSE and $R^2$ error.

## Installation
- clone repository
- python setup.py install

## Dependencies
- numpy >= 1.13
- scipy
- dipy
- pathos (optional for multi-core processing)
- cvxpy (optional for MIX optimization)
- numba (optional for faster functions)

## Getting Started:
To get a feeling for how to use Mipy, we provide a few tutorial notebooks:
- [Setting up an acquisition scheme](https://github.com/AthenaEPI/microstruktur/blob/master/examples/tutorial_setting_up_acquisition_scheme.ipynb)
- [Simulating and fitting data using a simple Stick model](https://github.com/AthenaEPI/microstruktur/blob/master/examples/tutorial_simulating_and_fitting_using_a_simple_model.ipynb)
- [Combining biophysical models into a Microstructure model](https://github.com/AthenaEPI/microstruktur/blob/master/examples/tutorial_combining_biophysical_models_into_microstructure_model.ipynb)
- [Imposing parameter links when combining biophysical models](https://github.com/AthenaEPI/microstruktur/blob/master/examples/tutorial_imposing_parameter_links.ipynb)
- [Using varying initial parameter settings when fitting larger data sets](https://github.com/AthenaEPI/microstruktur/blob/master/examples/tutorial_varying_intial_parameter_settings_for_larger_data_sets.ipynb)
- [Visualization Fiber Orientation Distributions (FODs) of dispersed models](https://github.com/AthenaEPI/microstruktur/blob/master/examples/tutorial_visualizing_fods_from_dispersed_models.ipynb)

## Explanations and Illustrations of Biophysical Models in the Toolbox
- [Intra-axonal Cylinder models](https://github.com/AthenaEPI/microstruktur/blob/master/examples/example_intra_axonal_cylinder_models.ipynb)
- [Extra-axonal Gaussian models](https://github.com/AthenaEPI/microstruktur/blob/master/examples/example_extra_axonal_gaussian_models.ipynb)
- [Axon dispersion models](https://github.com/AthenaEPI/microstruktur/blob/master/examples/example_watson_bingham.ipynb)
- [Spherical Mean Models](https://github.com/AthenaEPI/microstruktur/blob/master/examples/example_spherical_mean_models.ipynb)

## Mipy implementations of Microstructure Models in Literature
### Single Bundle Microstructure Models
- [Ball and Stick [Behrens et al. 2003]](https://github.com/AthenaEPI/microstruktur/blob/master/examples/example_ball_and_stick.ipynb)
- [Ball and Racket [Sotiropoulos et al. 2012]](https://github.com/AthenaEPI/microstruktur/blob/master/examples/example_ball_and_racket.ipynb)
- [NODDI-Watson [Zhang et al. 2012]](https://github.com/AthenaEPI/microstruktur/blob/master/examples/example_noddi_watson.ipynb)
- [NODDI-Bingham [Tariq et al. 2016]](https://github.com/AthenaEPI/microstruktur/blob/master/examples/example_noddi_bingham.ipynb)
- ActiveAx [Alexander et al. 2010]
- [AxCaliber [Assaf et al. 2008]](https://github.com/AthenaEPI/microstruktur/blob/master/examples/example_axcaliber.ipynb)
### Crossing Bundle Microstructure Models
- Using Brute Force optimization
- Using Dipy's CSD as crossing orientation initial guess
- [Using Microstructure Imaging of Crossing (MIX) [Farooq et al. 2016]](https://github.com/AthenaEPI/microstruktur/blob/master/examples/example_mix_microstructure_imaging_in_crossings.ipynb)
- [Multi-Compartment Spherical Mean Technique [Kaden et al. 2016]](https://github.com/AthenaEPI/microstruktur/blob/master/examples/example_multi_compartment_spherical_mean_technique.ipynb)
### Other Models
- [VERDICT tumor model [Panagiotaki et al. 2014]](https://github.com/AthenaEPI/mipy/blob/master/examples/example_verdict.ipynb)

### Examples not to be included in first version
- AxCaliber in presence of axon dispersion [unpublished but similar to Zhang at al. 2010]
- AxCaliber with restricted extra-axonal diffusion [Burcaw et al. 2015]
- MC-SMT with restricted extra-axonal diffusion [unpublished]

When public, use http://htmlpreview.github.io/ to replace ipynb with html files for better rendering.

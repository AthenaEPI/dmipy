# dMIPY: diffusion Microstructure Imaging in Python
NOTE: PACKAGE NOT FUNCTIONAL YET

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

See solutions to [common issues](https://github.com/AthenaEPI/mipy/blob/master/common_issues.md)
## Dependencies
Recommended to use Anaconda Python distribution.
- numpy >= 1.13
- scipy
- [dipy](http://nipy.org/dipy/)
- boto (optional for HCP-AWS interface)
- [pathos](https://pypi.python.org/pypi/pathos) (optional for multi-core processing)
- [numba](https://numba.pydata.org/) (optional for faster functions)

## Getting Started
To get a feeling for how to use Mipy, we provide a few tutorial notebooks:
- [Setting up an acquisition scheme](https://github.com/AthenaEPI/mipy/blob/master/examples/tutorial_setting_up_acquisition_scheme.ipynb)
- [Simulating and fitting data using a simple Stick model](https://github.com/AthenaEPI/mipy/blob/master/examples/tutorial_simulating_and_fitting_using_a_simple_model.ipynb)
- [Combining biophysical models into a Microstructure model](https://github.com/AthenaEPI/mipy/blob/master/examples/tutorial_combining_biophysical_models_into_microstructure_model.ipynb)
- [Creating a dispersed axon bundle representation and imposing parameter links](https://github.com/AthenaEPI/mipy/blob/master/examples/tutorial_imposing_parameter_links.ipynb)
- [Parameter Cascading: Using a simple model to initialize a complex one](https://github.com/AthenaEPI/mipy/blob/master/examples/tutorial_parameter_cascading_and_simulating_nd_datasets.ipynb)

## Explanations and Illustrations of Mipy Contents
### Biophysical Models and Distributions
- [Cylinder models (Axons)](https://github.com/AthenaEPI/mipy/blob/master/examples/example_cylinder_models.ipynb)
- [Sphere Models (Tumor cells)](https://github.com/AthenaEPI/mipy/blob/master/examples/example_sphere_models.ipynb)
- [Parameter distribution models (Gamma)](https://github.com/AthenaEPI/mipy/blob/master/examples/example_diameter_distributions.ipynb)
- [Gaussian models (Extra-axonal)](https://github.com/AthenaEPI/mipy/blob/master/examples/example_gaussian_models.ipynb)
- [Spherical Distribution models (Axon Dispersion)](https://github.com/AthenaEPI/mipy/blob/master/examples/example_watson_bingham.ipynb)
- [Spherical Mean Models](https://github.com/AthenaEPI/mipy/blob/master/examples/example_spherical_mean_models.ipynb)
### Global Optimizers
- Brute Force (Brute2Fine)
- Stochastic (MIX)
## Mipy implementations of Microstructure Models in Literature
Mipy uses HCP data to illustrate microstructure model examples. To reproduce these examples, Mipy provides a direct way to download HCP data (using your own AWS credentials) in the [HCP tutorial](https://github.com/AthenaEPI/mipy/blob/master/examples/tutorial_human_connectome_project_aws.ipynb).
### Single Bundle Microstructure Models
- [Ball and Stick [Behrens et al. 2003]](https://github.com/AthenaEPI/mipy/blob/master/examples/example_ball_and_stick.ipynb)
- [Ball and Racket [Sotiropoulos et al. 2012]](https://github.com/AthenaEPI/mipy/blob/master/examples/example_ball_and_racket.ipynb)
- [NODDI-Watson [Zhang et al. 2012]](https://github.com/AthenaEPI/mipy/blob/master/examples/example_noddi_watson.ipynb)
- [NODDI-Bingham [Tariq et al. 2016]](https://github.com/AthenaEPI/mipy/blob/master/examples/example_noddi_bingham.ipynb)
- [AxCaliber [Assaf et al. 2008]](https://github.com/AthenaEPI/mipy/blob/master/examples/example_axcaliber.ipynb)

### Crossing Bundle Microstructure Models
- [Ball and Sticks [Behrens et al. 2003]](https://github.com/AthenaEPI/mipy/blob/master/examples/example_ball_and_sticks.ipynb)
- [NODDI in crossings (NODDIx) [Farooq et al. 2016]](https://github.com/AthenaEPI/mipy/blob/master/examples/example_mix_microstructure_imaging_in_crossings.ipynb)
### Spherical Mean-Based Microstructure Models
- Spherical Mean Technique [Kaden et al. 2015]
- [Multi-Compartment Microscopic Diffusion Imaging [Kaden et al. 2016]](https://github.com/AthenaEPI/mipy/blob/master/examples/example_multi_compartment_spherical_mean_technique.ipynb)

## How to contribute to Mipy
Mipy's design is completely modular and can easily be extended with new models, distributions or optimizers. To contribute, view our contribution guidelines[link] and wanted additions[link].
## How to cite Mipy
Fick, Rutger. *Advanced dMRI signal modeling for tissue microstructure characterization*. Diss. Université Côte d'Azur; Inria Sophia Antipolis, 2017.

NOTE: PACKAGE NOT FUNCTIONAL YET

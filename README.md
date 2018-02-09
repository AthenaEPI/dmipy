# Dmipy: Diffusion Microstructure Imaging in Python

The Dmipy software package facilitates the estimation of diffusion MRI-based microstructure features by taking a completely modular approach to Microstructure Imaging. Using Dmipy you can design, fit, and recover the parameters of any multi-compartment microstructure model in usually less than 10 lines of code. Created models can be used to simulate and fit data for any PGSE-based dMRI acquisition, including including single shell, multi-shell, multi-diffusion time and multi-TE acquisition schemes. Dmipy's main features include the following:

**Complete Freedom in Model Design and Optimization**
- Any combination of tissue models (e.g. Gaussian, Cylinder, Sphere) can be combined into a multi-compartment model.
- Any model can be distributed with Watson/Bingham or Gamma distributions.
- Any predefined or custom parameter constraints or relations can be imposed.
- Free choice of global optimizer to fit your model to the data (Brute-Force or Stochastic)
- Ability to also fit the spherical mean of any multi-compartment model to the spherical mean of the data.

**Human Connectome Interface**
Dmipy enables you to directly download any HCP subject data using your own credentials.

**Fast, Multi-Core processing**
Despite its python-based modular design, Dmipy is still reasonably fast by using *pathos* multi-core processing and *numba* function compilation.

**Extensive documentation on Tissue and Microstructure Models**
We include documentation and illustrations of all tissue models and parameter distributions, as well as example implementations and results on HCP data for Ball and Stick, Ball and Racket, NODDI-Watson/Bingham, AxCaliber, Spherical Mean models and more.

**Dipy Compatibility**
Dmipy is designed to be complementary for Dipy users. Dipy gradient tables can be directly used in Dmipy models, Dipy models can be used to give initial parameter guesses for Dmipy optimization, and Dmipy models that estimate Fiber Orientation Distributions (FODs) can be visualized and used for tractography in Dipy.

Dmipy allows the user to do Microstructure Imaging research at the highest level, while the package automatically takes care of all the coding architecture that is needed to fit a designed model to a data set. The Dmipy documentation can be found at http://dmipy.readthedocs.io/. If you use Dmipy for your research publications, we kindly request you cite this package at the reference at the bottom of this page.

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
To get a feeling for how to use Dmipy, we provide a few tutorial notebooks:
- [Setting up an acquisition scheme](http://nbviewer.jupyter.org/github/AthenaEPI/mipy/blob/master/examples/tutorial_setting_up_acquisition_scheme.ipynb)
- [Simulating and fitting data using a simple Stick model](http://nbviewer.jupyter.org/github/AthenaEPI/mipy/blob/master/examples/tutorial_simulating_and_fitting_using_a_simple_model.ipynb)
- [Combining biophysical models into a Microstructure model](http://nbviewer.jupyter.org/github/AthenaEPI/mipy/blob/master/examples/tutorial_combining_biophysical_models_into_microstructure_model.ipynb)
- [Creating a dispersed axon bundle representation and imposing parameter links](http://nbviewer.jupyter.org/github/AthenaEPI/mipy/blob/master/examples/tutorial_imposing_parameter_links.ipynb)
- [Parameter Cascading: Using a simple model to initialize a complex one](http://nbviewer.jupyter.org/github/AthenaEPI/mipy/blob/master/examples/tutorial_parameter_cascading_and_simulating_nd_datasets.ipynb)

## Explanations and Illustrations of Dmipy Contents
### Biophysical Models and Distributions
- [Cylinder Models (Axons, e.g. [Assaf et al. 2004])](http://nbviewer.jupyter.org/github/AthenaEPI/mipy/blob/master/examples/example_cylinder_models.ipynb)
- [Sphere Models (Tumor cells, e.g. [Panagiotaki et al. 2014])](http://nbviewer.jupyter.org/github/AthenaEPI/mipy/blob/master/examples/example_sphere_models.ipynb)
- [Parameter Distribution Models (Axon Diameter Distribution, e.g. [Assaf et al. 2008])](http://nbviewer.jupyter.org/github/AthenaEPI/mipy/blob/master/examples/example_diameter_distributions.ipynb)
- [Gaussian Models (Extra-axonal, e.g. [Behrens et al. 2003])](http://nbviewer.jupyter.org/github/AthenaEPI/mipy/blob/master/examples/example_gaussian_models.ipynb)
- [Spherical Distribution Models (Axon Dispersion, e.g. [Kaden et al. 2007])](http://nbviewer.jupyter.org/github/AthenaEPI/mipy/blob/master/examples/example_watson_bingham.ipynb)
- [Spherical Mean of any Compartment Model](http://nbviewer.jupyter.org/github/AthenaEPI/mipy/blob/master/examples/example_spherical_mean_models.ipynb)
### Global Optimizers
- [Brute Force (Brute2Fine)](http://nbviewer.jupyter.org/github/AthenaEPI/dmipy/blob/master/examples/example_brute_force_optimization.ipynb)
- [Stochastic (MIX) [Farooq et al. 2016]](http://nbviewer.jupyter.org/github/AthenaEPI/dmipy/blob/master/examples/example_stochastic_mix_optimization.ipynb)
## Dmipy implementations of Microstructure Models in Literature
Dmipy uses HCP data to illustrate microstructure model examples. To reproduce these examples, dmipy provides a direct way to download HCP data (using your own AWS credentials) in the [HCP tutorial](http://nbviewer.jupyter.org/github/AthenaEPI/mipy/blob/master/examples/tutorial_human_connectome_project_aws.ipynb).
### Single Bundle Microstructure Models
- [Ball and Stick [Behrens et al. 2003]](http://nbviewer.jupyter.org/github/AthenaEPI/mipy/blob/master/examples/example_ball_and_stick.ipynb)
- [Ball and Racket [Sotiropoulos et al. 2012]](http://nbviewer.jupyter.org/github/AthenaEPI/mipy/blob/master/examples/example_ball_and_racket.ipynb)
- [NODDI-Watson [Zhang et al. 2012]](http://nbviewer.jupyter.org/github/AthenaEPI/mipy/blob/master/examples/example_noddi_watson.ipynb)
- [NODDI-Bingham [Tariq et al. 2016]](http://nbviewer.jupyter.org/github/AthenaEPI/mipy/blob/master/examples/example_noddi_bingham.ipynb)
- [AxCaliber [Assaf et al. 2008]](http://nbviewer.jupyter.org/github/AthenaEPI/mipy/blob/master/examples/example_axcaliber.ipynb)

### Crossing Bundle Microstructure Models
- [Ball and Sticks [Behrens et al. 2003]](http://nbviewer.jupyter.org/github/AthenaEPI/mipy/blob/master/examples/example_ball_and_sticks.ipynb)
- [NODDI in Crossings (NODDIx) [Farooq et al. 2016]](http://nbviewer.jupyter.org/github/AthenaEPI/mipy/blob/master/examples/example_mix_microstructure_imaging_in_crossings.ipynb)
### Spherical Mean-Based Microstructure Models
- [Spherical Mean Technique [Kaden et al. 2015]](http://nbviewer.jupyter.org/github/AthenaEPI/dmipy/blob/master/examples/example_spherical_mean_technique.ipynb)
- [Multi-Compartment Microscopic Diffusion Imaging [Kaden et al. 2016]](http://nbviewer.jupyter.org/github/AthenaEPI/mipy/blob/master/examples/example_multi_compartment_spherical_mean_technique.ipynb)

## How to contribute to Dmipy
Dmipy's design is completely modular and can easily be extended with new models, distributions or optimizers. To contribute, view our [contribution guidelines](https://github.com/AthenaEPI/dmipy/blob/master/contribution_guidelines.rst).
## How to cite Dmipy
Please temporarily cite the following PhD thesis: 
Fick, Rutger. *Advanced dMRI signal modeling for tissue microstructure characterization*. Diss. Université Côte d'Azur; Inria Sophia Antipolis, 2017.

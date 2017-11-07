# Microstructure Imaging in Python

The Mipy software package facilitates state-of-the-art diffusion MRI Microstructure Imaging using a "Building Block" philosophy. In this philosophy, any combination of biophysical models, typically representing intra- and/or extra-axonal tissue compartments, can be easy combined into a multi-compartment Microstructure Model in just a few lines of code. The created model can be used to both simulate and fit data for any PGSE-based dMRI acquisition, including single shell, multi-shell and qtau-dMRI acquisition schemes.

What sets Mipy apart from other packages is the complete freedom it gives the user to design and use a microstructure model. Mipy allows the user to do Microstructure Imaging research at the highest level, while the package automatically takes care of all the coding architecture that is needed to fit a designed model to a data set.

For Dipy users: Mipy is completely compatible and complementary to Dipy. Dipy gradient tables can be converted to Mipy acquisition schemes simpy using gtab_dipy2mipy(dipy_gradient_table), after which data sets can be fitted similarly as you were used to before. 

Mipy is now equiped with MIX optimization, giving global minimum for models with many compartments. Sources: [Research paper](https://escholarship.org/uc/item/9mr5b7ww).

## Installation
- clone repository
- python setup.py install

## Dependencies
- numpy >= 1.13
- scipy
- dipy

## Getting Started:
To get a feeling for how to use Mipy, we provide a few tutorial notebooks:
- [Setting up an acquisition scheme](https://github.com/AthenaEPI/microstruktur/blob/master/examples/tutorial_setting_up_acquisition_scheme.ipynb)
- [Simulating and fitting data using a simple Stick model](https://github.com/AthenaEPI/microstruktur/blob/master/examples/tutorial_simulating_and_fitting_using_a_simple_model.ipynb)
- [Combining biophysical models into a Microstructure model](https://github.com/AthenaEPI/microstruktur/blob/master/examples/tutorial_combining_biophysical_models_into_microstructure_model.ipynb)
- [Imposing parameter links when combining biophysical models](https://github.com/AthenaEPI/microstruktur/blob/master/examples/tutorial_imposing_parameter_links.ipynb)
- [Using varying initial parameter settings when fitting larger data sets](https://github.com/AthenaEPI/microstruktur/blob/master/examples/tutorial_varying_intial_parameter_settings_for_larger_data_sets.ipynb)
- Fitting real data with a custom model...

## Explanations and Illustrations of Biophysical Models in the Toolbox
- [Intra-axonal Cylinder models](https://github.com/AthenaEPI/microstruktur/blob/master/examples/example_intra_axonal_cylinder_models.ipynb)
- [Extra-axonal Gaussian models](https://github.com/AthenaEPI/microstruktur/blob/master/examples/example_extra_axonal_gaussian_models.ipynb)
- [Axon dispersion models](https://github.com/AthenaEPI/microstruktur/blob/master/examples/example_watson_bingham.ipynb)
- [Spherical Mean Models](https://github.com/AthenaEPI/microstruktur/blob/master/examples/example_spherical_mean_models.ipynb)

## Examples of how to implement Microstructure Models in Literature (not updated)
- [Ball and Stick [Behrens et al. 2003]](https://github.com/AthenaEPI/microstruktur/blob/master/examples/example_ball_and_stick.ipynb)
- [Ball and Racket [Sotiropoulos et al. 2012]](https://github.com/AthenaEPI/microstruktur/blob/master/examples/example_ball_and_racket.ipynb)
- [NODDI-Watson [Zhang et al. 2012]](https://github.com/AthenaEPI/microstruktur/blob/master/examples/example_noddi_watson.ipynb)
- NODDI-Bingham [Tariq et al. 2016]
- [Multi-Compartment Spherical Mean Technique [Kaden et al. 2016]](https://github.com/AthenaEPI/microstruktur/blob/master/examples/example_multi_compartment_spherical_mean_technique.ipynb)
- AxCaliber [Assaf et al. 2008]
- AxCaliber with restricted extra-axonal diffusion [Burcaw et al. 2015]
- ActiveAx [Alexander et al. 2010]

## Idea Box (not updated)
### Estimating extra-axonal restriction using spherical mean of restricted zeppelin
The idea is to modify the multi-compartment spherical mean technique to have a rotation-, dispersion-, and crossing-invariant method of estimating extra-axonal restriction. The technique modifies the spherical mean of the zeppelin (E4) to use the spherical mean of the restricted zeppelin (E5). If acquisition shells with multiple b-values, big delta and small delta are acquired, then the perpendicular diffusivity can be disentangled from the extra-axonal restriction parameter using the spherical mean framework. Proof of working [here](https://github.com/AthenaEPI/microstruktur/blob/master/examples/example_extra-axonal_restriction_estimation_using_spherical_mean.ipynb). The basis article manuscript with the relevant equations is [here](https://www.overleaf.com/9889990sjksnvyktkqc).
### Using a cascade style estimation framework to obtain robust measures (no degeneracy) in estimated microstructure parameters.
- First use spherical mean technique to get estimates of perpendicular en parallel diffusivity and volume fraction.
- input obtained parameters into NODDIDA to obtain dispersion.
- put combination of parameters in dispersed axcaliber model to obtain axon diameter.

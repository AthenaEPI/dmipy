# Universal Diffusion Microstructure Toolbox (UDMT)
other possible names... Microstructure Imaging in Python (Mipy)...

This toolbox unifies state-of-the-art diffusion MRI Microstructure Imaging using a "Building Block" philosophy. In this philosophy, any combination of biophysical models can be combined into a Microstructure Model, and can be used to both simulate and fit data for any PGSE-based dMRI acquisition, including single shell, multi-shell and qtau-dMRI acquisition schemes.

## Installation
- clone repository
- python setup.py install

## How to Get Started (Done)
-  [Basic usage tutorial](https://github.com/AthenaEPI/microstruktur/blob/master/examples/example_basic_usage.ipynb).

## Explanations and Illustrations of Biophysical Models in the Toolbox (under construction)
- [Intra-axonal Cylinder models](https://github.com/AthenaEPI/microstruktur/blob/master/examples/example_intra_axonal_cylinder_models.ipynb)
- [Extra-axonal Gaussian models](https://github.com/AthenaEPI/microstruktur/blob/master/examples/example_extra_axonal_gaussian_models.ipynb)
- [Axon dispersion models](https://github.com/AthenaEPI/microstruktur/blob/master/examples/example_watson_bingham.ipynb)

## Examples of how to implement Microstructure Models in Literature (not updated)
- [Ball and Stick [Behrens et al. 2003]](https://github.com/AthenaEPI/microstruktur/blob/master/examples/example_ball_and_stick.ipynb)
- [Ball and Racket [Sotiropoulos et al. 2012]](https://github.com/AthenaEPI/microstruktur/blob/master/examples/example_ball_and_racket.ipynb)
- [NODDI-Watson [Zhang et al. 2012]](https://github.com/AthenaEPI/microstruktur/blob/master/examples/example_noddi_watson.ipynb)
- NODDI-Bingham [Tariq et al. 2016]
- [Multi-Compartment Spherical Mean Technique [Kaden et al. 2016]](https://github.com/AthenaEPI/microstruktur/blob/master/examples/example_multi_compartment_spherical_mean_technique.ipynb)
- AxCaliber [Assaf et al. 2008]
- AxCaliber with restricted extra-axonal diffusion [Burcaw et al. 2015]
- ActiveAx [Alexander et al. 2010]

## Idea Box (not update)
### Estimating extra-axonal restriction using spherical mean of restricted zeppelin
The idea is to modify the multi-compartment spherical mean technique to have a rotation-, dispersion-, and crossing-invariant method of estimating extra-axonal restriction. The technique modifies the spherical mean of the zeppelin (E4) to use the spherical mean of the restricted zeppelin (E5). If acquisition shells with multiple b-values, big delta and small delta are acquired, then the perpendicular diffusivity can be disentangled from the extra-axonal restriction parameter using the spherical mean framework. Proof of working [here](https://github.com/AthenaEPI/microstruktur/blob/master/examples/example_extra-axonal_restriction_estimation_using_spherical_mean.ipynb). The basis article manuscript with the relevant equations is [here](https://www.overleaf.com/9889990sjksnvyktkqc).
### Using a cascade style estimation framework to obtain robust measures (no degeneracy) in estimated microstructure parameters.
- First use spherical mean technique to get estimates of perpendicular en parallel diffusivity and volume fraction.
- input obtained parameters into NODDIDA to obtain dispersion.
- put combination of parameters in dispersed axcaliber model to obtain axon diameter.

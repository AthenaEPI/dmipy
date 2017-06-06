# microstruktur Toolbox
To recover braintissue-specific biomarkers sensitive and specific to axon diameter, axon dispersion and volume fractions, diffusion MRI (dMRI)-based Microstructure Imaging uses combinations of specific mathematical representations (biophysical models) to represent diffusion in different tissue types. A combination of biophysical models is called a Microstructure Model. Over time, various microstructure models have been proposed for similar tissue types, consisting of different or overlapping combinations of biophysical models. Yet, comparison of these proposed models on similar data sets is scarsely found.

This microstructure toolbox unifies most state-of-the-art Microstructure Imaging approaches dMRI using a "Building Block" philosophy. In this philosophy, any combination of biophysical models can be combined into a Microstructure Model, and can both be used to simulate dMRI data for any acquisition scheme, and be fitted to any data set to estimate relevant model parameters. We provide examples to produce already proposed microstructure models from the simples Ball & Stick to the more recently proposed NODDI-Bingham.

We also provide a common data set that includes the effects of axon diameter, axon dispersion, varying diffusivities and intra-axonal volume fractions, which we also used to produce results for our journal paper, which has been submitted:
editeable journal article file: https://www.overleaf.com/7710782nxbwpxvyxkwq

## Installation
- clone repository
- python setup.py install

## Getting Started

## Explanations and Illustrations of Biophysical Models in the Toolbox
- Intra-axonal Cylinder models
- Extra-axonal Gaussian models
- [Axon dispersion models](https://github.com/AthenaEPI/microstruktur/blob/master/examples/example_watson_bingham.ipynb)

## Examples of how to implement Microstructure Models in Literature
- [Ball and Stick [Behrens et al. 2003]](https://github.com/AthenaEPI/microstruktur/blob/master/examples/example_ball_and_stick.ipynb)
- [Ball and Racket [Sotiropoulos et al. 2012]](https://github.com/AthenaEPI/microstruktur/blob/master/examples/example_ball_and_racket.ipynb)
- [NODDI-Watson [Zhang et al. 2012]](https://github.com/AthenaEPI/microstruktur/blob/master/examples/example_noddi_watson.ipynb)
- NODDI-Bingham [Tariq et al. 2016]
- [Multi-Compartment Spherical Mean Technique [Kaden et al. 2016]](https://github.com/AthenaEPI/microstruktur/blob/master/examples/example_multi_compartment_spherical_mean_technique.ipynb)
- AxCaliber [Assaf et al. 2008]
- AxCaliber with restricted extra-axonal diffusion [Burcaw et al. 2015]
- ActiveAx [Alexander et al. 2010]

## Idea Box
### Estimating extra-axonal restriction using spherical mean of restricted zeppelin
The idea is to modify the multi-compartment spherical mean technique to have a rotation-, dispersion-, and crossing-invariant method of estimating extra-axonal restriction. The technique modifies the spherical mean of the zeppelin (E4) to use the spherical mean of the restricted zeppelin (E5). If acquisition shells with multiple b-values, big delta and small delta are acquired, then the perpendicular diffusivity can be disentangled from the extra-axonal restriction parameter using the spherical mean framework. Proof of working [here](https://github.com/AthenaEPI/microstruktur/blob/master/examples/example_extra-axonal_restriction_estimation_using_spherical_mean.ipynb). The basis article manuscript with the relevant equations is [here](https://www.overleaf.com/9889990sjksnvyktkqc).
### Using a cascade style estimation framework to obtain robust measures (no degeneracy) in estimated microstructure parameters.
- First use spherical mean technique to get estimates of perpendicular en parallel diffusivity and volume fraction.
- input obtained parameters into NODDIDA to obtain dispersion.
- put combination of parameters in dispersed axcaliber model to obtain axon diameter.

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
- Ball and Racket [Sotiropoulos et al. 2012]
- [NODDI-Watson [Zhang et al. 2012]](https://github.com/AthenaEPI/microstruktur/blob/master/examples/example_noddi_watson.ipynb)
- NODDI-Bingham [Tariq et al. 2016]
- [Multi-Compartment Spherical Mean Technique [Kaden et al. 2016]](https://github.com/AthenaEPI/microstruktur/blob/master/examples/example_multi_compartment_spherical_mean_technique.ipynb)
- AxCaliber [Assaf et al. 2008]
- AxCaliber with restricted extra-axonal diffusion [Burcaw et al. 2015]
- ActiveAx [Alexander et al. 2010]

from os.path import join
import pkg_resources
import nibabel as nib
import numpy as np
DATA_PATH = pkg_resources.resource_filename(
    'microstruktur', 'data/'
)


def wu_minn_hcp_coronal_slice():
    data_name = 'wu_minn_hcp_coronal_slice.nii.gz'
    return nib.load(join(DATA_PATH, data_name)).get_data()


def synthetic_camino_data_parallel():
    fractions_1_7 = np.loadtxt(DATA_PATH + 'fractions_camino_D1_7.txt')
    fractions_2_0 = np.loadtxt(DATA_PATH + 'fractions_camino_D2_0.txt')
    fractions_2_3 = np.loadtxt(DATA_PATH + 'fractions_camino_D2_3.txt')

    data_1_7 = np.loadtxt(join(DATA_PATH, 'data_camino_D1_7.txt'))
    data_2_0 = np.loadtxt(join(DATA_PATH, 'data_camino_D2_0.txt'))
    data_2_3 = np.loadtxt(join(DATA_PATH, 'data_camino_D2_3.txt'))

    fractions = np.r_[fractions_1_7, fractions_2_0, fractions_2_3]
    data = np.r_[data_1_7, data_2_0, data_2_3]
    diffusivity = np.r_[np.tile(1.7e-9, len(fractions_1_7)),
                        np.tile(2e-9, len(fractions_2_0)),
                        np.tile(2.3e-9, len(fractions_2_3))]

    class CaminoData:
        def __init__(self):
            self.fractions = fractions
            self.diffusivities = diffusivity
            self.signal_attenuation = data
    return CaminoData()


def synthetic_camino_data_dispersed():
    data_1_7_dispersed = np.loadtxt(
        join(DATA_PATH, 'data_camino_dispersed_D1_7.txt'))
    data_2_0_dispersed = np.loadtxt(
        join(DATA_PATH, 'data_camino_dispersed_D2_0.txt'))
    data_2_3_dispersed = np.loadtxt(
        join(DATA_PATH, 'data_camino_dispersed_D2_3.txt'))
    data = np.r_[
        data_1_7_dispersed, data_2_0_dispersed, data_2_3_dispersed]

    parameters_1_7_dispersed = np.loadtxt(
        join(DATA_PATH, 'parameters_camino_dispersed_D1_7.txt'))
    parameters_2_0_dispersed = np.loadtxt(
        join(DATA_PATH, 'parameters_camino_dispersed_D2_0.txt'))
    parameters_2_3_dispersed = np.loadtxt(
        join(DATA_PATH, 'parameters_camino_dispersed_D2_3.txt'))

    fractions = np.r_[
        parameters_1_7_dispersed[:, 0],
        parameters_2_0_dispersed[:, 0],
        parameters_2_3_dispersed[:, 0]
    ]

    kappas = np.r_[
        parameters_1_7_dispersed[:, 1],
        parameters_2_0_dispersed[:, 1],
        parameters_2_3_dispersed[:, 1]
    ]

    betas = np.r_[
        parameters_1_7_dispersed[:, 2],
        parameters_2_0_dispersed[:, 2],
        parameters_2_3_dispersed[:, 2]
    ]

    diffusivity = np.r_[np.tile(1.7e-9, len(parameters_1_7_dispersed)),
                        np.tile(2e-9, len(parameters_2_0_dispersed)),
                        np.tile(2.3e-9, len(parameters_2_3_dispersed))]

    class DispersedCaminoData:
        def __init__(self):
            self.fractions = fractions
            self.diffusivities = diffusivity
            self.signal_attenuation = data
            self.kappa = kappas
            self.beta = betas
    return DispersedCaminoData()

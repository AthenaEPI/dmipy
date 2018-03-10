from os.path import join
import os
import pkg_resources
import nibabel as nib
import numpy as np
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
import zipfile
from . import saved_acquisition_schemes

try:
    from urllib2 import urlopen
except ImportError:
    from urllib.request import urlopen

DATA_PATH = pkg_resources.resource_filename(
    'dmipy', 'data'
)

__all__ = [
    'wu_minn_hcp_coronal_slice',
    'duval_cat_spinal_cord_2d',
    'synthetic_camino_data_parallel',
    'synthetic_camino_data_dispersed',
    'visualize_correlation_camino_and_estimated_fractions'
]

def deSantis_camino_data():
    """
    Downloads and returns the 4-shell multi-delta/Delta/G scheme based on acquistion scheme defined in [1]_.
    Note that acquisition parameters in [1]_ used for a STEAM sequence, are used here to generate a PGSE one.

    Returns
    -------
    scheme: DmipyAcquisitionScheme instance,
        acquisition scheme of the generated de Santis data.
    data_genu: array of size (50, 54),
        contains 50 repetitions with added rician noise SNR=30.


    References
    ----------
    .. [1] De Santis, S., Jones, D. K., & Roebroeck, A. (2016). Including 
	diffusion time dependence in the extra-axonal space improves in vivo 
	estimates of axonal diameter and density in human white matter. NeuroImage, 130, 91-103.
    """
    deSantis_data_path = join(DATA_PATH, 'deSantis_camino')

    data = np.loadtxt(join(deSantis_data_path, 'deSantis_signal.txt'), skiprows=2)
	
    scheme = (
        saved_acquisition_schemes.deSantis_generated_acquisition_scheme()
    )    
    return scheme, data


def wu_minn_hcp_coronal_slice():
    "Returns example slice of Wu-Minn HCP data subject 100307."
    data_path = join(
        DATA_PATH, 'hcp', 'hcp_example_slice', 'coronal_slice.nii.gz')
    try:
        data = nib.load(data_path).get_data()
    except IOError:
        msg = "The example HCP data has not been downloaded yet. "
        msg += "Please follow our HCP tutorial where you can use your own AWS "
        msg += "credentials to download the example data and any other HCP "
        msg += "subject data."
        raise ValueError(msg)

    scheme = saved_acquisition_schemes.wu_minn_hcp_acquisition_scheme()

    msg = "This data slice originates from Subject 100307 of the Human "
    msg += "Connectome Project, WU-Minn Consortium (Principal Investigators: "
    msg += "David Van Essen and Kamil Ugurbil; 1U54MH091657) funded by the 16 "
    msg += "NIH Institutes and Centers that support the NIH Blueprint for "
    msg += "Neuroscience Research; and by the McDonnell Center for Systems "
    msg += "Neuroscience at Washington University."
    print(msg)

    return scheme, data


def duval_cat_spinal_cord_2d():
    "Returns 2D multi-diffusion time AxCaliber data of cat spinal cord."
    msg = "This data was used by Duval et al. 'Validation of quantitative MRI "
    msg += "metrics using full slice histology with automatic axon "
    msg += "segmentation', ISMRM 2016. Reference at "
    msg += "Cohen-Adad et al. White Matter Microscopy Database."
    msg += " http://doi.org/10.17605/OSF.IO/YP4QG"
    print(msg)

    data_folder = join(DATA_PATH, "tanguy_cat_spinal_cord")

    class Histology:
        def __init__(self):
            self.h1_axonEquivDiameter = nib.load(
                join(data_folder, '1_axonEquivDiameter.nii')).get_data()
            self.h2_axonEquivDiameter_std = nib.load(
                join(data_folder, '2_axonEquivDiameter_std.nii')).get_data()
            self.h3_axonEquivDiameter_axonvolumeCorrected = nib.load(
                join(data_folder, '3_axonEquivDiameter_axonvolumeCorrected.nii'
                     )).get_data()
            self.h4_fr = nib.load(join(data_folder, '4_fr.nii')).get_data()
            self.h5_MyelinVolumeFraction = nib.load(
                join(data_folder, '5_MyelinVolumeFraction.nii')).get_data()
            self.h6_gRatio = nib.load(
                join(data_folder, '6_gRatio.nii')).get_data()
            self.h7_Number_axons = nib.load(
                join(data_folder, '7_Number_axons.nii')).get_data()

    class DuvalSpinalCordData2D:
        def __init__(self):
            data_name = "tanguy_spinal_cord_2D.nii.gz"
            self.signal = nib.load(join(data_folder, data_name)).get_data()
            self.histology = Histology()
            self.mask = (self.histology.h4_fr > 0)[..., None]

    data = DuvalSpinalCordData2D()
    scheme = (
        saved_acquisition_schemes.duval_cat_spinal_cord_2d_acquisition_scheme()
    )
    return scheme, data


def duval_cat_spinal_cord_3d():
    "Returns 2D multi-diffusion time AxCaliber data of cat spinal cord."
    msg = "This data was used by Duval et al. 'Validation of quantitative MRI "
    msg += "metrics using full slice histology with automatic axon "
    msg += "segmentation', ISMRM 2016. Reference at "
    msg += "Cohen-Adad et al. White Matter Microscopy Database."
    msg += " http://doi.org/10.17605/OSF.IO/YP4QG"
    print(msg)

    data_folder = join(DATA_PATH, "tanguy_cat_spinal_cord")

    class Histology:
        def __init__(self):
            self.h1_axonEquivDiameter = nib.load(
                join(data_folder, '1_axonEquivDiameter.nii')).get_data()
            self.h2_axonEquivDiameter_std = nib.load(
                join(data_folder, '2_axonEquivDiameter_std.nii')).get_data()
            self.h3_axonEquivDiameter_axonvolumeCorrected = nib.load(
                join(data_folder, '3_axonEquivDiameter_axonvolumeCorrected.nii'
                     )).get_data()
            self.h4_fr = nib.load(join(data_folder, '4_fr.nii')).get_data()
            self.h5_MyelinVolumeFraction = nib.load(
                join(data_folder, '5_MyelinVolumeFraction.nii')).get_data()
            self.h6_gRatio = nib.load(
                join(data_folder, '6_gRatio.nii')).get_data()
            self.h7_Number_axons = nib.load(
                join(data_folder, '7_Number_axons.nii')).get_data()

    class DuvalSpinalCordData3D:
        def __init__(self):
            data_name = "tanguy_spinal_cord_3D.nii.gz"
            self.signal = nib.load(join(data_folder, data_name)).get_data()
            self.histology = Histology()
            self.mask = (self.histology.h4_fr > 0)[..., None]

    data = DuvalSpinalCordData3D()
    scheme = (
        saved_acquisition_schemes.duval_cat_spinal_cord_3d_acquisition_scheme()
    )
    return scheme, data


def isbi2015_white_matter_challenge():
    """
    Downloads and returns the 35-shell multi-delta/Delta/G scheme and data for
    the fornix and genu data that was used for the ISBI 2015 white matter
    challenge [1]_.

    Returns
    -------
    scheme: DmipyAcquisitionScheme instance,
        acquisition scheme of the challenge data.
    data_genu: array of size (3612, 6),
        contains the DWIs for 6 genu voxels.
    data_fornix: array of size (3612, 6),
        contains the DWIs for 6 fornix voxels.

    References
    ----------
    .. [1] Ferizi, Uran, et al. "Diffusion MRI microstructure models with in
        vivo human brain Connectome data: results from a multi-group
        comparison." NMR in Biomedicine 30.9 (2017)
    """
    isbi_data_path = join(DATA_PATH, 'isbi2015_white_matter_challenge')

    if not os.path.exists(isbi_data_path):
        os.makedirs(isbi_data_path)

    path_genu = (
        "http://cmic.cs.ucl.ac.uk/wmmchallenge/ISBIdata/seenSignal.txt")
    path_fornix = (
        "http://cmic.cs.ucl.ac.uk/wmmchallenge/ISBIdata/seenSignaX.txt")

    filenames = ['genu.txt', 'fornix.txt']
    paths = [path_genu, path_fornix]

    for filename, path in zip(filenames, paths):
        response = urlopen(path)
        data = response.read()
        file_ = open(join(isbi_data_path, filename), 'wb')
        file_.write(data)
        file_.close()

    data_genu = np.loadtxt(join(isbi_data_path, 'genu.txt'), skiprows=1)
    data_fornix = np.loadtxt(join(isbi_data_path, 'fornix.txt'), skiprows=1)
    scheme = (
        saved_acquisition_schemes.isbi2015_white_matter_challenge_scheme()
    )
    return scheme, data_genu.T, data_fornix.T


def panagiotaki_verdict():
    """
    Downloads and returns the example VERDICT acquisition scheme and data that
    is available at the UCL website. The data is an example of [1]_.

    Returns
    -------
    scheme: DmipyAcquisitionScheme instance,
        acquisition scheme of the challenge data.
    data_verdict: array,
        contains the DWIs for a single tumor voxel.

    References
    ----------
    .. [1] Panagiotaki, Eletheria, et al. "Noninvasive quantification of solid
        tumor microstructure using VERDICT MRI." Cancer research 74.7 (2014):
        1902-1912.
    """
    verdict_data_path = join(DATA_PATH, 'panagiotaki_verdict')
    if not os.path.exists(verdict_data_path):
        os.makedirs(verdict_data_path)

    url = "http://camino.cs.ucl.ac.uk/uploads/Tutorials/"
    filename = "LSDTIDWtut.Bfloat.zip"
    response = urlopen(join(url, filename))
    with open(join(verdict_data_path, filename), 'wb') as f:
        f.write(response.read())
    with zipfile.ZipFile(join(verdict_data_path, filename)) as zip:
        zip.extract("LSDTIDWtut.Bfloat", path=verdict_data_path)

    data_verdict = np.fromfile(join(verdict_data_path, "LSDTIDWtut.Bfloat"),
                               dtype='>f')
    scheme = saved_acquisition_schemes.panagiotaki_verdict_acquisition_scheme()
    return scheme, data_verdict


def synthetic_camino_data_parallel():
    """The parallel data was generated using the Camino Monte-Carlo
    Diffusion Simulator. See http://camino.cs.ucl.ac.uk/.
    """
    fractions_1_7 = np.loadtxt(
        join(DATA_PATH, 'camino', 'fractions_camino_D1_7.txt'))
    fractions_2_0 = np.loadtxt(
        join(DATA_PATH, 'camino', 'fractions_camino_D2_0.txt'))
    fractions_2_3 = np.loadtxt(
        join(DATA_PATH, 'camino', 'fractions_camino_D2_3.txt'))

    data_1_7 = np.loadtxt(join(DATA_PATH, 'camino', 'data_camino_D1_7.txt'))
    data_2_0 = np.loadtxt(join(DATA_PATH, 'camino', 'data_camino_D2_0.txt'))
    data_2_3 = np.loadtxt(join(DATA_PATH, 'camino', 'data_camino_D2_3.txt'))

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

    scheme = saved_acquisition_schemes.wu_minn_hcp_acquisition_scheme()
    return scheme, CaminoData()


def synthetic_camino_data_dispersed():
    """The dispersed data was generated by using the parallel Camino data as
    an described above, and then dispersing it using Watson and Bingham
    distributions.
    """
    data_1_7_dispersed = np.loadtxt(
        join(DATA_PATH, 'camino', 'data_camino_dispersed_D1_7.txt'))
    data_2_0_dispersed = np.loadtxt(
        join(DATA_PATH, 'camino', 'data_camino_dispersed_D2_0.txt'))
    data_2_3_dispersed = np.loadtxt(
        join(DATA_PATH, 'camino', 'data_camino_dispersed_D2_3.txt'))
    data = np.r_[
        data_1_7_dispersed, data_2_0_dispersed, data_2_3_dispersed]

    parameters_1_7_dispersed = np.loadtxt(
        join(DATA_PATH, 'camino', 'parameters_camino_dispersed_D1_7.txt'))
    parameters_2_0_dispersed = np.loadtxt(
        join(DATA_PATH, 'camino', 'parameters_camino_dispersed_D2_0.txt'))
    parameters_2_3_dispersed = np.loadtxt(
        join(DATA_PATH, 'camino', 'parameters_camino_dispersed_D2_3.txt'))

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

    scheme = saved_acquisition_schemes.wu_minn_hcp_acquisition_scheme()
    return scheme, DispersedCaminoData()


def visualize_correlation_camino_and_estimated_fractions(
        estim_fractions_parallel, estim_fractions_dispersed):
    "Function that visualizes Camino estimated results versus ground truth."

    data_parallel = synthetic_camino_data_parallel()
    data_dispersed = synthetic_camino_data_dispersed()

    mask_par_17 = data_parallel.diffusivities == 1.7e-9
    mask_disp_17 = data_dispersed.diffusivities == 1.7e-9

    fractions_par_17 = data_parallel.fractions[mask_par_17]
    fractions_disp_17 = data_dispersed.fractions[mask_disp_17]

    estim_fractions_par_17 = estim_fractions_parallel[mask_par_17]
    estim_fractions_disp_17 = estim_fractions_dispersed[mask_disp_17]

    pr = pearsonr(estim_fractions_par_17, fractions_par_17)
    pr_dispersed = pearsonr(estim_fractions_disp_17, fractions_disp_17)
    pr_multidif = pearsonr(estim_fractions_parallel, data_parallel.fractions)
    pr_multidif_dispersed = pearsonr(
        estim_fractions_dispersed, data_dispersed.fractions)

    f, ((ax1, ax2), (ax3, ax4)) = plt.subplots(
        2, 2, sharex='col', sharey='row')
    ax1.scatter(fractions_par_17, estim_fractions_par_17)
    ax2.scatter(data_parallel.fractions, estim_fractions_parallel)
    ax3.scatter(fractions_disp_17, estim_fractions_disp_17)
    ax4.scatter(data_dispersed.fractions, estim_fractions_dispersed)

    ax1.text(.216, .817, 'pearsonR= ' +
             str(np.round(pr[0], 3)), fontsize=10,
             bbox=dict(facecolor='white', alpha=1))
    ax2.text(.216, .817, 'pearsonR= ' +
             str(np.round(pr_multidif[0], 3)), fontsize=10,
             bbox=dict(facecolor='white', alpha=1))
    ax3.text(.216, .817, 'pearsonR= ' +
             str(np.round(pr_dispersed[0], 3)), fontsize=10,
             bbox=dict(facecolor='white', alpha=1))
    ax4.text(.216, .817, 'pearsonR= ' + str(np.round(
        pr_multidif_dispersed[0], 3)), fontsize=10,
        bbox=dict(facecolor='white', alpha=1))

    ax1.set_title('Static Diffusivity')
    ax3.set_xlabel('Ground Truth')
    ax2.set_title('Varying Diffusivity')
    ax1.set_ylabel('Estimated intra-vf')
    ax4.set_xlabel('Ground Truth')
    ax3.set_ylabel('Estimated intra-vf')

    ax1.plot([0, 1], [0, 1], ls='--', c='k', lw=3)
    ax2.plot([0, 1], [0, 1], ls='--', c='k', lw=3)
    ax3.plot([0, 1], [0, 1], ls='--', c='k', lw=3)
    ax4.plot([0, 1], [0, 1], ls='--', c='k', lw=3)
    ax1.set_ylim(0.2, .9)
    ax1.set_xlim(0.2, .8)
    ax4.set_ylim(0.2, .9)
    ax4.set_xlim(0.2, .8)

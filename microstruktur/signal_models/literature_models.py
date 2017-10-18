from microstruktur.signal_models import (
    three_dimensional_models,
    dispersed_models
)
from microstruktur.signal_models.utils import (
    T1_tortuosity, parameter_equality
)

# ball and stick
_ball = three_dimensional_models.E3Ball()
_stick = three_dimensional_models.I1Stick()

ball_and_stick = (
    three_dimensional_models.PartialVolumeCombinedMicrostrukturModel(
        models=[_stick, _ball],
        parameter_links=[],
        optimise_partial_volumes=True)
)

# ball and racket
_ball = three_dimensional_models.E3Ball()
_bingham_stick = dispersed_models.SD2I1BinghamDispersedStick()

ball_and_racket = (
    three_dimensional_models.PartialVolumeCombinedMicrostrukturModel(
        [_bingham_stick, _ball],
        parameter_links=[],
        optimise_partial_volumes=True)
)

# NODDI Watson with Dpar = 1.7e-9


def T1_tortuosity_with_preset_lambda_par(f_intra, lambda_par=1.7):
    lambda_perp = (1 - f_intra) * lambda_par
    return lambda_perp


_dispersed_stick = dispersed_models.SD3I1WatsonDispersedStick()
_dispersed_zeppelin = dispersed_models.SD3E4WatsonDispersedZeppelin()

_parameter_links = [
    (  # parallel stick diffusivity to 1.7
        _dispersed_stick, 'lambda_par',
        lambda: 1.7, []
    ),
    (  # tortuosity assumption
        _dispersed_zeppelin, 'lambda_perp',
        T1_tortuosity_with_preset_lambda_par, [
            (None, 'partial_volume_0')
        ]
    ),
    (  # parallel zeppelin diffusivity to 1.7
        _dispersed_zeppelin, 'lambda_par',
        lambda: 1.7, []
    ),
    (  # kappa equality
        _dispersed_stick, 'kappa',
        parameter_equality, [
            (_dispersed_zeppelin, 'kappa')
        ]
    ),
    (  # mu equality
        _dispersed_stick, 'mu',
        parameter_equality, [
            (_dispersed_zeppelin, 'mu')
        ]
    )
]

noddi_watson = (
    three_dimensional_models.PartialVolumeCombinedMicrostrukturModel(
        models=[_dispersed_stick, _dispersed_zeppelin],
        parameter_links=_parameter_links,
        optimise_partial_volumes=True)
)

# multicompartment spherical mean technique
_stick_spherical_mean = three_dimensional_models.I1StickSphericalMean()
_zeppelin_spherical_mean = three_dimensional_models.E4ZeppelinSphericalMean()

_parameter_links_smt = [
    (  # tortuosity assumption
        _zeppelin_spherical_mean, 'lambda_perp',
        T1_tortuosity, [
            (None, 'partial_volume_0'),
            (_stick_spherical_mean, 'lambda_par')
        ]
    ),
    (  # equal parallel diffusivities
        _zeppelin_spherical_mean, 'lambda_par',
        parameter_equality, [
            (_stick_spherical_mean, 'lambda_par')
        ]
    )
]

multi_compartment_spherical_mean_technique = (
    three_dimensional_models.PartialVolumeCombinedMicrostrukturModel(
        models=[_stick_spherical_mean, _zeppelin_spherical_mean],
        parameter_links=_parameter_links_smt,
        optimise_partial_volumes=True)
)

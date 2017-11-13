import numpy as np
from microstruktur.core.acquisition_scheme import (
    acquisition_scheme_from_bvalues,
    acquisition_scheme_from_qvalues,
    acquisition_scheme_from_gradient_strengths,
    calculate_shell_bvalues_and_indices,
    gtab_dipy2mipy, gtab_mipy2dipy)
from dipy.core.gradients import gradient_table
from numpy.testing import (
    assert_raises, assert_equal, assert_array_equal)


def test_catch_negative_bvalues(Nsamples=10):
    bvalues = np.tile(-1, Nsamples)
    bvecs = np.tile(np.r_[1., 0., 0.], (Nsamples, 1))
    delta = np.ones(Nsamples)
    Delta = np.ones(Nsamples)
    assert_raises(ValueError, acquisition_scheme_from_bvalues,
                  bvalues, bvecs, delta, Delta)


def test_catch_different_length_bvals_bvecs(Nsamples=10):
    bvalues = np.tile(1, Nsamples - 1)
    bvecs = np.tile(np.r_[1., 0., 0.], (Nsamples, 1))
    delta = np.ones(Nsamples)
    Delta = np.ones(Nsamples)
    assert_raises(ValueError, acquisition_scheme_from_bvalues,
                  bvalues, bvecs, delta, Delta)


def test_catch_2d_bvals_bvecs(Nsamples=10):
    bvalues = np.ones((Nsamples, 2))
    bvecs = np.tile(np.r_[1., 0., 0.], (Nsamples, 1))
    delta = np.ones(Nsamples)
    Delta = np.ones(Nsamples)
    assert_raises(ValueError, acquisition_scheme_from_bvalues,
                  bvalues, bvecs, delta, Delta)


def test_catch_different_shape_bvals_delta_Delta(Nsamples=10):
    bvalues = np.tile(1, Nsamples)
    bvecs = np.tile(np.r_[1., 0., 0.], (Nsamples, 1))
    delta = np.ones(Nsamples - 1)
    Delta = np.ones(Nsamples)
    assert_raises(ValueError, acquisition_scheme_from_bvalues,
                  bvalues, bvecs, delta, Delta)

    delta = np.ones(Nsamples)
    Delta = np.ones(Nsamples - 1)
    assert_raises(ValueError, acquisition_scheme_from_bvalues,
                  bvalues, bvecs, delta, Delta)


def test_catch_2d_delta_Delta(Nsamples=10):
    bvalues = np.tile(1, Nsamples)
    bvecs = np.tile(np.r_[1., 0., 0.], (Nsamples, 1))
    delta = np.ones((Nsamples, 2))
    Delta = np.ones(Nsamples)
    assert_raises(ValueError, acquisition_scheme_from_bvalues,
                  bvalues, bvecs, delta, Delta)

    delta = np.ones(Nsamples)
    Delta = np.ones((Nsamples, 2))
    assert_raises(ValueError, acquisition_scheme_from_bvalues,
                  bvalues, bvecs, delta, Delta)


def test_catch_negative_delta_Delta(Nsamples=10):
    bvalues = np.tile(1, Nsamples)
    bvecs = np.tile(np.r_[1., 0., 0.], (Nsamples, 1))
    delta = -np.ones(Nsamples)
    Delta = np.ones(Nsamples)
    assert_raises(ValueError, acquisition_scheme_from_bvalues,
                  bvalues, bvecs, delta, Delta)

    delta = np.ones(Nsamples)
    Delta = -np.ones(Nsamples)
    assert_raises(ValueError, acquisition_scheme_from_bvalues,
                  bvalues, bvecs, delta, Delta)


def test_catch_wrong_shape_bvecs(Nsamples=10):
    bvalues = np.tile(1, Nsamples)
    bvecs = np.tile(np.r_[1., 0., 0.], (Nsamples, 1, 1))
    delta = np.ones(Nsamples)
    Delta = np.ones(Nsamples)
    assert_raises(ValueError, acquisition_scheme_from_bvalues,
                  bvalues, bvecs, delta, Delta)


def test_catch_non_unit_vector_bvecs(Nsamples=10):
    bvalues = np.tile(1, Nsamples)
    bvecs = np.tile(np.r_[1., 0., 0.], (Nsamples, 1)) + 1.
    delta = np.ones(Nsamples)
    Delta = np.ones(Nsamples)
    assert_raises(ValueError, acquisition_scheme_from_bvalues,
                  bvalues, bvecs, delta, Delta)


def test_equivalent_scheme_bvals_and_bvecs(Nsamples=10):
    bvalues = np.tile(1, Nsamples)
    bvecs = np.tile(np.r_[1., 0., 0.], (Nsamples, 1))
    delta = np.ones(Nsamples)
    Delta = np.ones(Nsamples)
    scheme_from_bvals = acquisition_scheme_from_bvalues(
        bvalues, bvecs, delta, Delta)
    qvalues = scheme_from_bvals.qvalues
    scheme_from_qvals = acquisition_scheme_from_qvalues(
        qvalues, bvecs, delta, Delta)
    bvalues_from_qvalues = scheme_from_qvals.bvalues
    assert_array_equal(bvalues, bvalues_from_qvalues)


def test_equivalent_scheme_bvals_and_gradient_strength(Nsamples=10):
    bvalues = np.tile(1, Nsamples)
    bvecs = np.tile(np.r_[1., 0., 0.], (Nsamples, 1))
    delta = np.ones(Nsamples)
    Delta = np.ones(Nsamples)
    scheme_from_bvals = acquisition_scheme_from_bvalues(
        bvalues, bvecs, delta, Delta)
    gradient_strengths = scheme_from_bvals.gradient_strengths
    scheme_from_gradient_strengths = (
        acquisition_scheme_from_gradient_strengths(
            gradient_strengths, bvecs, delta, Delta))
    bvalues_from_gradient_strengths = (
        scheme_from_gradient_strengths.bvalues)
    assert_array_equal(bvalues, bvalues_from_gradient_strengths)


def test_estimate_shell_indices():
    bvalues = np.arange(10)
    max_distance = 1
    shell_indices, shell_bvalues = (
        calculate_shell_bvalues_and_indices(
            bvalues, max_distance=max_distance))
    assert_equal(int(np.unique(shell_indices)), 0)
    assert_equal(float(shell_bvalues), np.mean(bvalues))

    max_distance = 0.5
    shell_indices, shell_bvalues = (
        calculate_shell_bvalues_and_indices(
            bvalues, max_distance=max_distance))
    assert_array_equal(shell_indices, bvalues)


def test_shell_indices_with_vayring_diffusion_times(Nsamples=10):
    # tests whether measurements with the same bvalue but different diffusion
    # time are correctly classified in different shells
    bvalues = np.tile(1e9, Nsamples)
    delta = 0.01
    Delta = np.hstack([np.tile(0.01, len(bvalues) / 2),
                       np.tile(0.03, len(bvalues) / 2)])
    gradient_directions = np.tile(np.r_[1., 0., 0.], (Nsamples, 1))
    scheme = acquisition_scheme_from_bvalues(
        bvalues, gradient_directions, delta, Delta)
    assert_equal(len(np.unique(scheme.shell_indices)), 2)


def test_dipy2mipy_acquisition_converter(Nsamples=10):
    bvals = np.tile(1e3, Nsamples)
    bvecs = np.tile(np.r_[1., 0., 0.], (Nsamples, 1))
    big_delta = 0.03
    small_delta = 0.01
    gtab_dipy = gradient_table(
        bvals=bvals, bvecs=bvecs, small_delta=small_delta, big_delta=big_delta)
    gtab_mipy = gtab_dipy2mipy(gtab_dipy)
    assert_array_equal(gtab_mipy.bvalues / 1e6, gtab_dipy.bvals)
    assert_array_equal(gtab_mipy.gradient_directions, gtab_dipy.bvecs)
    assert_equal(np.unique(gtab_mipy.Delta), gtab_dipy.big_delta)
    assert_equal(np.unique(gtab_mipy.delta), gtab_dipy.small_delta)


def test_mipy2dipy_acquisition_converter(Nsamples=10):
    bvals = np.tile(1e9, Nsamples)
    bvecs = np.tile(np.r_[1., 0., 0.], (Nsamples, 1))
    big_delta = 0.03
    small_delta = 0.01
    gtab_mipy = acquisition_scheme_from_bvalues(
        bvalues=bvals, gradient_directions=bvecs,
        delta=small_delta, Delta=big_delta)
    gtab_dipy = gtab_mipy2dipy(gtab_mipy)
    assert_array_equal(gtab_mipy.bvalues / 1e6, gtab_dipy.bvals)
    assert_array_equal(gtab_mipy.gradient_directions, gtab_dipy.bvecs)
    assert_equal(gtab_mipy.Delta, gtab_dipy.big_delta)
    assert_equal(gtab_mipy.delta, gtab_dipy.small_delta)

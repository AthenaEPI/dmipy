import numpy as np
from microstruktur.acquisition_scheme.acquisition_scheme import (
    acquisition_scheme_from_bvalues,
    acquisition_scheme_from_qvalues,
    acquisition_scheme_from_gradient_strengths,
    calculate_shell_bvalues_and_indices)
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

# -*- coding: utf-8 -*-
"""Tests for whitening.transforms.Whitening.

Master invariant: for all methods, np.cov(fit_transform(X).T) ≈ I.
"""

import numpy as np
import pytest
from numpy.testing import assert_allclose
from sklearn.base import clone
from sklearn.exceptions import NotFittedError

from whitening import Whitening

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

ALL_METHODS = ['zca', 'pca', 'cholesky', 'zca_cor', 'pca_cor']


def _correlated_data(n=500, p=5, seed=42):
    """Return (n, p) array from a correlated multivariate normal with
    non-zero mean and features on very different scales."""
    rng = np.random.default_rng(seed)
    A = rng.standard_normal((p, p))
    Sigma = A @ A.T + np.eye(p)          # guaranteed positive-definite
    mu = rng.standard_normal(p) * 5      # non-zero mean tests centering
    X = rng.multivariate_normal(mu, Sigma, size=n)
    # Give features different scales to stress the cor-based methods.
    scale = np.array([0.1, 1.0, 10.0, 100.0, 0.5][:p])
    return X * scale


def _check_cov_identity(X_w, atol=1e-4):
    C = np.cov(X_w.T)
    assert_allclose(C, np.eye(X_w.shape[1]), atol=atol,
                    err_msg="Covariance of whitened output is not ≈ I")


# ---------------------------------------------------------------------------
# Core invariant: Cov(X_w) ≈ I
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("method", ALL_METHODS)
def test_identity_covariance_z_scoring_true(method):
    X = _correlated_data()
    X_w = Whitening(method=method, z_scoring=True).fit_transform(X)
    _check_cov_identity(X_w)


@pytest.mark.parametrize("method", ALL_METHODS)
def test_identity_covariance_z_scoring_false(method):
    X = _correlated_data()
    X_w = Whitening(method=method, z_scoring=False).fit_transform(X)
    _check_cov_identity(X_w)


@pytest.mark.parametrize("method", ALL_METHODS)
def test_fit_transform_equals_fit_then_transform(method):
    X = _correlated_data()
    w1, w2 = Whitening(method=method), Whitening(method=method)
    assert_allclose(w1.fit_transform(X), w2.fit(X).transform(X), atol=1e-12)


# ---------------------------------------------------------------------------
# Bug 1: scaler fitted once in fit(), reused in transform()
# ---------------------------------------------------------------------------

def test_scaler_not_refit_in_transform():
    """transform() must reuse the scaler fitted on X_train, not refit."""
    rng = np.random.default_rng(7)
    scale = np.array([1, 10, 100, 0.1])
    X_train = rng.standard_normal((200, 4)) * scale
    X_test = rng.standard_normal((50, 4)) * scale + 5

    w = Whitening(method='zca', z_scoring=True).fit(X_train)

    # Two calls to transform with the same input must be identical.
    assert_allclose(w.transform(X_test.copy()), w.transform(X_test.copy()),
                    atol=1e-12, err_msg="transform() is not deterministic")

    # The saved scaler must carry training-set statistics.
    # X_train mean ≈ 0, X_test mean ≈ 5 → after applying the training scaler,
    # X_test should NOT have mean ≈ 0.
    X_test_scaled = w.scaler_.transform(X_test - w.mean_)
    assert abs(X_test_scaled.mean()) > 0.1, (
        "Scaler seems to have been fit on test data (mean ≈ 0 after transform)"
    )


def test_scaler_is_none_when_z_scoring_false():
    X = _correlated_data()
    w = Whitening(z_scoring=False).fit(X)
    assert w.scaler_ is None


def test_cor_methods_have_no_separate_scaler():
    """zca_cor/pca_cor embed variance scaling in W_; no extra scaler."""
    X = _correlated_data()
    for method in ('zca_cor', 'pca_cor'):
        w = Whitening(method=method).fit(X)
        assert w.scaler_ is None, f"{method} should have scaler_=None"


# ---------------------------------------------------------------------------
# Bugs 2 & 3: V_sqrt correctness in zca_cor / pca_cor
# ---------------------------------------------------------------------------

def test_zca_cor_rows_not_uniform_when_features_have_different_scales():
    """If V_sqrt were computed from already z-scored X (bug 3), all
    feature stds would be ≈ 1 and W_ rows would have identical norms."""
    X = _correlated_data()  # features intentionally have different scales
    w = Whitening(method='zca_cor').fit(X)
    row_norms = np.linalg.norm(w.W_, axis=1)
    assert row_norms.std() > 1e-3, (
        "W_ rows all have equal norm — V_sqrt was likely computed from "
        "z-scored X (bug 3 not fixed)."
    )


def test_zca_cor_cov_identity_high_correlation():
    """High-correlation, mixed-scale data stresses cor vs cov distinction."""
    rng = np.random.default_rng(0)
    n = 1000
    base = rng.standard_normal(n)
    X = np.column_stack([
        base * 10 + rng.standard_normal(n) * 0.1,
        base * 0.5 + rng.standard_normal(n) * 0.05,
        base * 3 + rng.standard_normal(n) * 0.3,
        rng.standard_normal(n),
    ])
    X_w = Whitening(method='zca_cor', eps=1e-6).fit_transform(X)
    _check_cov_identity(X_w, atol=1e-3)


def test_cor_whitening_matrix_satisfies_identity_condition():
    """Directly verify W @ Sigma @ W.T ≈ I for zca_cor and pca_cor."""
    X = _correlated_data(n=1000)
    for method in ('zca_cor', 'pca_cor'):
        w = Whitening(method=method, z_scoring=False).fit(X)
        X_c = X - w.mean_
        n = len(X_c)
        Sigma = X_c.T @ X_c / (n - 1)   # unbiased, matches fit()
        result = w.W_ @ Sigma @ w.W_.T
        # atol=1e-5 accounts for the eps regularisation shifting eigenvalues.
        assert_allclose(result, np.eye(X.shape[1]), atol=1e-5,
                        err_msg=f"{method}: W @ Sigma @ W.T ≠ I")


# ---------------------------------------------------------------------------
# Bug 5: check_is_fitted guard
# ---------------------------------------------------------------------------

def test_transform_raises_not_fitted():
    with pytest.raises(NotFittedError):
        Whitening().transform(_correlated_data())


# ---------------------------------------------------------------------------
# Invalid method
# ---------------------------------------------------------------------------

def test_invalid_method_raises():
    with pytest.raises(ValueError, match="Unknown method"):
        Whitening(method='bad').fit(_correlated_data())


# ---------------------------------------------------------------------------
# scikit-learn compatibility
# ---------------------------------------------------------------------------

def test_clone_has_no_fitted_attributes():
    w = Whitening(method='pca').fit(_correlated_data())
    w_clone = clone(w)
    with pytest.raises((AttributeError, NotFittedError)):
        _ = w_clone.W_


def test_get_set_params():
    w = Whitening(eps=1e-4, method='pca')
    assert w.get_params()['eps'] == 1e-4
    assert w.get_params()['method'] == 'pca'
    w.set_params(eps=1e-3, method='zca')
    assert w.eps == 1e-3
    assert w.method == 'zca'


# ---------------------------------------------------------------------------
# Determinism and copy=False
# ---------------------------------------------------------------------------

def test_deterministic_fit():
    X = _correlated_data()
    w1 = Whitening(method='zca_cor').fit(X)
    w2 = Whitening(method='zca_cor').fit(X)
    assert_allclose(w1.W_, w2.W_, atol=1e-14)


@pytest.mark.parametrize("method", ['zca', 'zca_cor'])
def test_copy_false_still_correct(method):
    X = _correlated_data()
    w = Whitening(method=method, copy=False)
    X_w = w.fit(X.copy()).transform(X.copy())
    _check_cov_identity(X_w)

# -*- coding: utf-8 -*-
"""
Whitening (sphering) transformations as a scikit-learn compatible transformer.

Implements ZCA, PCA, Cholesky, ZCA-cor, and PCA-cor whitening as described in:

    Kessy, Lewin & Strimmer (2018). "Optimal whitening and decorrelation."
    The American Statistician, 72(4), 309–314.
    https://arxiv.org/abs/1512.00809

The ZCA whitening (Mahalanobis whitening) ensures that the average covariance
between whitened and original variables is maximal.  ZCA-cor whitening leads
to whitened variables that are maximally correlated on average with the
original variables.
"""

__author__ = 'Severin Langberg'
__contact__ = 'langberg91@gmail.com'

import numpy as np
from scipy import linalg
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler
from sklearn.utils.validation import check_array, check_is_fitted


class Whitening(TransformerMixin, BaseEstimator):
    """Whiten (sphere) a data matrix so that the output has identity covariance.

    Five whitening methods are supported.  All produce output ``X_w = X @ W_.T``
    such that ``Cov(X_w) ≈ I``.

    For ``zca_cor`` and ``pca_cor`` the whitening matrix is derived from the
    *correlation* matrix of the input (Kessy et al. 2018).  These methods
    embed the per-feature variance scaling inside ``W_`` and therefore always
    operate on mean-centred (not variance-scaled) data regardless of the
    ``z_scoring`` setting.

    For ``zca``, ``pca``, and ``cholesky`` the ``z_scoring`` parameter controls
    whether the input is additionally standardised before the whitening matrix
    is computed.

    Parameters
    ----------
    eps : float, default=1e-6
        Regularisation constant added to eigenvalues before inversion to
        prevent division by zero.
    method : {'zca', 'pca', 'cholesky', 'zca_cor', 'pca_cor'}, default='zca_cor'
        Whitening method.
    z_scoring : bool, default=True
        For ``zca``, ``pca``, and ``cholesky`` only: if True, standardise
        (divide by per-feature std) the mean-centred input before computing
        the whitening matrix.  Has no effect for the ``_cor`` methods, which
        always embed variance scaling in the whitening matrix itself.
    copy : bool, default=True
        If False, avoid copying input arrays where possible.

    Attributes
    ----------
    W_ : ndarray of shape (n_features, n_features)
        The fitted whitening matrix.  ``transform(X)`` returns ``X @ W_.T``.
    mean_ : ndarray of shape (n_features,)
        Per-feature means of the training data, used to centre inputs in
        ``transform``.
    scaler_ : StandardScaler or None
        Fitted variance scaler used by ``zca``/``pca``/``cholesky`` when
        ``z_scoring=True``.  None otherwise.
    """

    _cor_methods = frozenset({'zca_cor', 'pca_cor'})
    _valid_methods = frozenset({'zca', 'pca', 'cholesky', 'zca_cor', 'pca_cor'})

    def __init__(self, eps=1e-6, method='zca_cor', z_scoring=True, copy=True):
        self.eps = eps
        self.method = method
        self.z_scoring = z_scoring
        self.copy = copy

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(self, X, y=None):
        """Fit the whitening matrix.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
        y : ignored

        Returns
        -------
        self
        """
        if self.method not in self._valid_methods:
            raise ValueError(
                f"Unknown method {self.method!r}. "
                f"Choose from {sorted(self._valid_methods)}."
            )

        X = check_array(X, copy=self.copy, dtype=np.float64)

        # Always mean-centre.  The cor-based methods need the centred (but
        # NOT variance-scaled) data; other methods may additionally scale.
        self.mean_ = X.mean(axis=0)
        X_centred = X - self.mean_

        if self.method in self._cor_methods:
            # Sigma and std are both derived from mean-centred (unscaled) X.
            # Variance information lives in V^{1/2} = diag(std), which is
            # embedded directly in W_.  No separate scaler is needed.
            self.scaler_ = None
            n = X_centred.shape[0]
            Sigma = X_centred.T @ X_centred / (n - 1)

            if self.method == 'zca_cor':
                self.W_ = self._zca_cor(Sigma, X_centred)
            else:
                self.W_ = self._pca_cor(Sigma, X_centred)

        else:
            # Optionally variance-scale before computing Sigma.
            if self.z_scoring:
                self.scaler_ = StandardScaler(with_mean=False, copy=False)
                X_for_sigma = self.scaler_.fit_transform(X_centred)
            else:
                self.scaler_ = None
                X_for_sigma = X_centred

            n = X_for_sigma.shape[0]
            Sigma = X_for_sigma.T @ X_for_sigma / (n - 1)

            if self.method == 'cholesky':
                self.W_ = self._cholesky(Sigma)
            elif self.method == 'zca':
                self.W_ = self._zca(Sigma)
            else:  # pca
                self.W_ = self._pca(Sigma)

        return self

    def transform(self, X):
        """Apply the whitening transform.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)

        Returns
        -------
        X_w : ndarray of shape (n_samples, n_features)
            Whitened data.  ``np.cov(X_w.T)`` is close to the identity matrix.
        """
        check_is_fitted(self, ['W_', 'mean_'])
        X = check_array(X, copy=self.copy, dtype=np.float64)

        X = X - self.mean_

        if self.scaler_ is not None:
            X = self.scaler_.transform(X)

        return X @ self.W_.T

    # ------------------------------------------------------------------
    # Private whitening-matrix constructors
    # ------------------------------------------------------------------

    def _cholesky(self, Sigma):
        # scipy cholesky (lower=False default) returns upper-triangular R with
        # R.T @ R = Sigma_inv.  We need W s.t. W @ Sigma @ W.T = I.
        # R @ Sigma @ R.T = I  ✓  so return R; transform applies X @ R.T.
        U, Lambda, _ = np.linalg.svd(Sigma)
        return linalg.cholesky(
            U @ np.diag(1.0 / (Lambda + self.eps)) @ U.T
        )

    def _zca(self, Sigma):
        U, Lambda, _ = linalg.svd(Sigma)
        return U @ np.diag(1.0 / np.sqrt(Lambda + self.eps)) @ U.T

    def _pca(self, Sigma):
        U, Lambda, _ = linalg.svd(Sigma)
        return np.diag(1.0 / np.sqrt(Lambda + self.eps)) @ U.T

    def _zca_cor(self, Sigma, X_centred):
        """ZCA-cor whitening matrix (Kessy et al. 2018).

        W = G @ Θ^{-1/2} @ G^T @ V^{-1/2}

        where  P = V^{-1/2} @ Σ @ V^{-1/2}  is the correlation matrix,
               G @ Θ @ G^T = SVD(P),
        and    V^{1/2} = diag(std of mean-centred X).

        Applied to mean-centred X: Cov(X @ W^T) = I.
        """
        std = X_centred.std(axis=0)
        V_inv = np.diag(1.0 / std)                 # V^{-1/2}
        P = V_inv @ Sigma @ V_inv                   # correlation matrix

        G, Theta, _ = linalg.svd(P)
        return G @ np.diag(1.0 / np.sqrt(Theta + self.eps)) @ G.T @ V_inv

    def _pca_cor(self, Sigma, X_centred):
        """PCA-cor whitening matrix (Kessy et al. 2018).

        W = Θ^{-1/2} @ G^T @ V^{-1/2}

        Applied to mean-centred X: Cov(X @ W^T) = I.
        """
        std = X_centred.std(axis=0)
        V_inv = np.diag(1.0 / std)
        P = V_inv @ Sigma @ V_inv

        G, Theta, _ = linalg.svd(P)
        return np.diag(1.0 / np.sqrt(Theta + self.eps)) @ G.T @ V_inv

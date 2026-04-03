# whitening

Scikit-learn compatible whitening (sphering) transformations.

Based on:

> Kessy, Lewin & Strimmer (2018). *Optimal whitening and decorrelation.*  
> The American Statistician, 72(4), 309–314.  
> https://arxiv.org/abs/1512.00809

## What is whitening?

Whitening transforms a data matrix **X** into **X_w = X @ W^T** such that
`Cov(X_w) = I` — the features of the output are uncorrelated and have unit
variance.  Different whitening matrices **W** produce outputs with different
geometric properties.

## Methods

| Method | Whitening matrix W | Properties |
|---|---|---|
| `zca` | **U** Λ^{-1/2} **U**^T | Closest to identity; maximises avg. correlation with original |
| `pca` | Λ^{-1/2} **U**^T | Rotates to principal-component axes |
| `cholesky` | chol(Σ^{-1})^T | Lower-triangular; computationally efficient |
| `zca_cor` | **G** Θ^{-1/2} **G**^T **V**^{-1/2} | ZCA in correlation space |
| `pca_cor` | Θ^{-1/2} **G**^T **V**^{-1/2} | PCA in correlation space |

**Notation:** Σ = sample covariance, **U** Λ **U**^T = SVD(Σ),
**V**^{1/2} = diag(σ₁,…,σ_p), P = **V**^{-1/2} Σ **V**^{-1/2} (correlation matrix),
**G** Θ **G**^T = SVD(P).

The `_cor` variants derive **W** from the correlation matrix, embedding the
per-feature variance scaling directly in the whitening matrix.  They operate
on mean-centred data and do not require a separate standardisation step.

## Installation

```bash
pip install .
```

For development (includes pytest):

```bash
pip install -e ".[dev]"
```

## Quick start

```python
import numpy as np
from whitening import Whitening

rng = np.random.default_rng(0)
X_train = rng.standard_normal((300, 5)) @ rng.standard_normal((5, 5))
X_test  = rng.standard_normal((100, 5)) @ rng.standard_normal((5, 5))

w = Whitening(method='zca_cor')        # default
X_train_w = w.fit_transform(X_train)
X_test_w  = w.transform(X_test)        # reuses statistics from fit()

print(np.cov(X_train_w.T).round(3))    # ≈ identity matrix
```

Choosing a method:

```python
# ZCA — output stays in the original feature space
w = Whitening(method='zca')

# PCA — output aligned to principal components
w = Whitening(method='pca')

# Correlation-based ZCA — good when features have very different scales
w = Whitening(method='zca_cor')

# Disable additional variance scaling (z_scoring only affects zca/pca/cholesky)
w = Whitening(method='zca', z_scoring=False)
```

## scikit-learn compatibility

`Whitening` follows the `fit` / `transform` / `fit_transform` API and inherits
`get_params` / `set_params` / `clone` from `BaseEstimator`.  It can be used
inside a `Pipeline`:

```python
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

pipe = Pipeline([
    ('whiten', Whitening(method='zca_cor')),
    ('clf',    LogisticRegression()),
])
pipe.fit(X_train, y_train)
```

## Running tests

```bash
pytest
```

## Parameters

| Parameter | Type | Default | Description |
|---|---|---|---|
| `method` | str | `'zca_cor'` | Whitening method (see table above) |
| `eps` | float | `1e-6` | Regularisation added to eigenvalues to avoid division by zero |
| `z_scoring` | bool | `True` | For `zca`/`pca`/`cholesky`: standardise by per-feature std before computing **W** |
| `copy` | bool | `True` | Copy input arrays; set to `False` to transform in-place where possible |

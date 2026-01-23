# NaNs appearence in PCA process problem solution report

## Project
- Dataset:
- Method:
- Target:

## Problem
### 1. NaNs errors happen in `pca().fit()`.
#### Error description
> ValueError: array must not contain infs or NaNs

- Error location: `pca = PCA(n_components=n_components, whiten=True).fit(X_train)`

- Unusual phenomena: Errors occur randomly.

####  Reason
##### Whitening

When you set `whiten=True` in PCA, you're asking it to scale each principal component to have unit variance. This is done by dividing the components by the square root of the explained variance (which is related to the singular values from the SVD).

If a principal component has a very small, near-zero variance, you end up dividing by a very small number. This can lead to floating-point instability, resulting in inf (infinity) or NaN (Not a Number) values.

##### Intermittent: svd_sover

By default (svd_solver='auto'), for datasets of a certain size, PCA uses a randomized SVD solver (svd_solver='randomized'). This is a faster approximation of the exact SVD. Because it has a random element, the results can vary slightly from run to run. In some runs, the randomized algorithm might produce a singular value that is small enough to cause the division-by-near-zero issue during whitening. In other runs, the singular values might be just large enough to avoid it.

#### Solution
Use the Full SVD Solver: This is more computationally intensive but deterministic, so it won't produce different results on different runs.
```python
pca = PCA(n_components=n_components, whiten=True, svd_solver='full').fit(X_train)
```

### 2. NaNs errors happen after `pca.transform()`
#### Error description

> ValueError: NaN detected in X_train_pca after PCA!
> ValueError: NaN detected in X_test_pca after PCA!
The NaNs in X_train_pca or X_test_pca will influence the following modules.

- Error location:
```python
X_train_pca = pca.transform(X_train)
X_test_pca = pca.transform(X_test)
```

- Unusual phenomena: Errors occur randomly in two varibales.

#### Reason
The problem is still division by a near-zero number. The fit() method calculates the principal components and their explained_variance_. Some of these variance values are extremely small (numerically close to zero).

#### Solution
Filter Out Zero-Variance Features: pre-process your data to remove features (pixels) that don't vary at all across the samples.

```python
# Before the PCA block, add/uncomment this:
from sklearn.feature_selection import VarianceThreshold

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

print("Original number of features:", X_train.shape[1])
vt = VarianceThreshold(threshold=1e-4) # Using a small threshold
X_train = vt.fit_transform(X_train)
X_test = vt.transform(X_test)
print("Number of features after variance thresholding:", X_train.shape[1])
```


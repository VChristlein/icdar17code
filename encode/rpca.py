import numpy as np
from scipy import linalg
# note: in sklearn <= 0.15 it is array2d instead of check_array
try: 
    from sklearn.utils import check_array
    def array2d(x):
        return check_array(x)
except ImportError:
    from sklearn.utils import array2d

from sklearn.utils import as_float_array
from sklearn.utils.extmath import fast_logdet, randomized_svd
from sklearn.base import TransformerMixin, BaseEstimator

# let's use the formulation of the scikit-learn master branch
# instead of 15.x (otherwise we could just use that PCA as base class)
class RegularizedPCA(BaseEstimator, TransformerMixin):
    def __init__(self, n_components=None, copy=True, whiten=False,
                 regularization=10**-5, start_component=0):
        self.n_components = n_components
        self.copy = copy
        self.whiten = whiten
        self.regularization = regularization
        self.start_c = start_component

    def fit(self, X):
        X = array2d(X)
        n_samples, n_features = X.shape
        X = as_float_array(X, copy=self.copy)
        # Center data
        self.mean_ = np.mean(X, axis=0)
        X -= self.mean_
        U, S, V = linalg.svd(X, full_matrices=False)
        explained_variance_ = (S ** 2) / n_samples
        explained_variance_ratio_ = (explained_variance_ /
                                     explained_variance_.sum())
        components_ = V
        n_components = self.n_components
        if n_components is None:
            n_components = n_features

        # store n_samples to revert whitening when getting covariance
        self.n_samples_ = n_samples
        self.components_ = components_[self.start_c:self.start_c+n_components]
        self.explained_variance_ = explained_variance_[self.start_c:self.start_c+n_components]
        self.explained_variance_ratio_ = explained_variance_ratio_[self.start_c:self.start_c+n_components]
        self.n_components_ = n_components
        return self

    def transform(self, X):
        """Apply the dimensionality reduction on X.
        X is projected on the first principal components previous extracted
        from a training set.
        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
        New data, where n_samples is the number of samples
        and n_features is the number of features.
        Returns
        -------
        X_new : array-like, shape (n_samples, n_components)
        """
        X = array2d(X)
        if self.mean_ is not None:
            X -= self.mean_
       

        # Note: fast_dot and np.dot don't work anymore with my 
        # multiprocessing ^^
        X_transformed = np.dot(X, self.components_.T)
        if self.whiten:
            X_transformed /= np.sqrt(self.explained_variance_ +
                                     self.regularization)
        return X_transformed



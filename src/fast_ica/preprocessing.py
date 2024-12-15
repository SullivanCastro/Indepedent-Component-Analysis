import numpy as np
from numpy.typing import ArrayLike
import warnings


class Preprocessing:

    @staticmethod
    def _centering(X: ArrayLike) -> ArrayLike:
        """
        Subtract the mean of each dimension from the data matrix X.

        Parameters
        ----------
        X : ArrayLike
            Data matrix of shape (N, M).

        Returns
        -------
        ArrayLike
            Centered data matrix of shape (N, M).
        """
        XT = X.T
        return XT - XT.mean(axis=-1, keepdims=True)

    @staticmethod
    def _whitening_eigh(X: ArrayLike, n_components=1) -> ArrayLike:
        """
        Whitening of the data matrix X such that X_whitened @ X_whitened.T = I.

        Parameters
        ----------
        X : ArrayLike
            Data matrix of shape (N, M).

        Returns
        -------
        ArrayLike
            Whitened data matrix of shape (N, M).
        """
        # Faster when num_samples >> n_features
        _, n_samples = X.shape
        d, u = np.linalg.eigh(X.dot(X.T))
        sort_indices = np.argsort(d)[::-1]
        eps = np.finfo(d.dtype).eps * 10
        degenerate_idx = d < eps
        if np.any(degenerate_idx):
            warnings.warn(
                "There are some small singular values, using "
                "whiten_solver = 'svd' might lead to more "
                "accurate results."
            )
        d[degenerate_idx] = eps  # For numerical issues
        np.sqrt(d, out=d)
        d, u = d[sort_indices], u[:, sort_indices]

        u *= np.sign(u[0])
        K = (u / d).T[:n_components]
        X_whitened = np.dot(K, X)
        X_whitened *= np.sqrt(n_samples)
        return X_whitened
    
    
    @staticmethod
    def _whitening_svd(X: ArrayLike, n_components=1) -> ArrayLike:
        """
        
        Parameters
        ----------
        X : ArrayLike
            Data matrix of shape (n_samples, n_features).

        Returns
        -------
        ArrayLike
            Whitened data matrix of shape (n_components, n_samples
        """
        _, n_samples = X.shape
        u, d = np.linalg.svd(X, full_matrices=False)[:2]

        u *= np.sign(u[0])
        K = (u / d).T[:n_components]
        X_whitened = np.dot(K, X)
        X_whitened *= np.sqrt(n_samples)
        return X_whitened
    

    @staticmethod
    def _whitening(X, method="svd", n_components=1):
        if method == "eigh":
            return Preprocessing._whitening_eigh(X, n_components)
        elif method == "svd":
            return Preprocessing._whitening_svd(X, n_components)
        else:
            raise ValueError("Invalid method. Choose between 'eigh' and 'svd'.")
    

    @staticmethod
    def preprocessing(X, method="svd", n_components=1):

        # Centering
        XT = Preprocessing._centering(X)

        # Whitening
        XT = Preprocessing._whitening(XT, method=method, n_components=n_components)
        return XT

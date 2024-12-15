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
            Data matrix of shape (n_components, n_samples).

        Returns
        -------
        ArrayLike
            Centered data matrix of shape (n_components, n_samples).
        """
        return X - X.mean(axis=-1, keepdims=True)

    @staticmethod
    def _whitening_eigh(X: ArrayLike) -> ArrayLike:
        """
        Whitening of the data matrix X such that X_whitened @ X_whitened.T = I.

        Parameters
        ----------
        X : ArrayLike
            Data matrix of shape (n_components, n_samples).

        Returns
        -------
        ArrayLike
            Whitened data matrix of shape (n_componentss, n_samples).
        """
        # Compute the diagonalisation of the covariance matrix
        eigvals, eigvecs = np.linalg.eigh(X.dot(X.T))

        # Sort the eigenvalues in descending order
        sort_eigvals = np.argsort(eigvals)[::-1]

        # Handle the degenerate eigenvalues
        eps = np.finfo(eigvals.dtype).eps * 10
        degenerate_idx = eigvals < eps
        if np.any(degenerate_idx):
            warnings.warn(
                "There are some small singular values, using "
                "whiten_solver = 'svd' might lead to more "
                "accurate results."
            )
        eigvals[degenerate_idx] = eps  # For numerical issues

        # Sort and compute the square root of the eigenvalues
        eigvals = np.sqrt(eigvals)
        eigvals, eigvecs = eigvals[sort_eigvals], eigvecs[:, sort_eigvals]

        return eigvals, eigvecs
    
    
    @staticmethod
    def _whitening_svd(X: ArrayLike) -> ArrayLike:
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
        # Compute the SVD of the data matrix
        sigvecs, sigvals = np.linalg.svd(X, full_matrices=False)[:2]

        return sigvals, sigvecs
    

    @staticmethod
    def _whitening(X, whiten_method="svd"):
        if whiten_method == "eigh":
            return Preprocessing._whitening_eigh(X)
        elif whiten_method == "svd":
            return Preprocessing._whitening_svd(X)
        else:
            raise ValueError("Invalid method. Choose between 'eigh' and 'svd'.")
    

    @staticmethod
    def preprocessing(X, whiten_method="eigh"):

        # Centering
        X = Preprocessing._centering(X)

        # Whitening
        vals, vecs = Preprocessing._whitening(X, whiten_method=whiten_method)

        # The projection is up to the sign of the vector
        vecs *= np.sign(vecs[0])

        # Creating the projection whitening matrix
        K = (vecs / vals).T[:X.shape[0]]

        # Projecting the data
        X_whitened = K @ X
        X_whitened *= np.sqrt(X.shape[1])

        return X_whitened

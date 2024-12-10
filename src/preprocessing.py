import numpy as np
from numpy.typing import ArrayLike


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
        return X - X.mean(axis=1, keepdims=True)

    @staticmethod
    def _whitening(X: ArrayLike) -> ArrayLike:
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
        # Compute covariance matrix
        E = np.cov(X)

        # Compute the EVD decomposition and handle non positive eigenvalues
        eigenvalues, eigenvectors = np.linalg.eigh(E)
        eigenvalues_cleaned = np.where(eigenvalues < 1e-25, 1e-25, eigenvalues)

        # Compute the whitened data matrix
        X_whitened = (
            eigenvectors @ np.diag(eigenvalues_cleaned**-0.5) @ eigenvectors.T @ X
        )

        # Check if the whitening was successful
        if np.linalg.norm(np.cov(X_whitened) - np.eye(X.shape[0])) > 1e-10:
            print(
                f"[WARNING] Whitening failed, norm of covariance matrix: {np.linalg.norm(np.cov(X_whitened))}"
            )

        return X_whitened

    @staticmethod
    def preprocessing(X):
        # Centering
        X = Preprocessing._centering(X)

        # Whitening
        X = Preprocessing._whitening(X)
        return X

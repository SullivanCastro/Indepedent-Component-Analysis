import numpy as np
from numpy.typing import ArrayLike
from preprocessing import Preprocessing


class Fast_ICA:

    def __init__(
        self, n_components: int, max_iter: int = 200, tol: float = 1e-4
    ) -> None:
        """
        Initialize the Fast_ICA object.

        Parameters
        ----------
        n_components : int
            Number of components to extract.
        max_iter : int
            Maximum number of iterations.
        tol : float
            Tolerance for convergence

        Returns
        -------
        None
        """
        self._max_iter = max_iter
        self._tol = tol
        self._n_components = n_components
        self._g = lambda X: X**3
        self._dg = lambda X: 3 * X**2

    def _compute_new_weights(self, X: ArrayLike, w: ArrayLike) -> ArrayLike:
        """
        Compute the new weights for the Fast_ICA algorithm according to the update rule.

        Parameters
        ----------
        X : ArrayLike
            Data matrix of shape (N, M).
        w : ArrayLike
            Current weights of shape (N, M).

        Returns
        -------
        ArrayLike
            New weights of shape (N, M).
        """
        # Compute each part of the update rule
        Y = w.T @ X
        cov_g = (X * self._g(Y)).mean(axis=1)
        cov_dg = np.mean(self._dg(Y)) * w

        # Update weights
        w_new = cov_g - cov_dg

        # Return the new weights
        return w_new

    def _orthogonalisation(self, w: ArrayLike, previous_w: ArrayLike) -> ArrayLike:
        """
        Orthogonalize the new weight vector w with respect to the previous weight vectors W (according to the Gram-Schmidt method).

        Parameters
        ----------
        w_new : ArrayLike
            New weight vector of shape (N,).
        previous_w : ArrayLike
            Previous weight vectors of shape (K, N).

        Returns
        -------
        ArrayLike
            Orthogonalized weight vector of shape (N,).
        """
        w_normalized = w.copy()
        w_normalized -= previous_w.T @ (previous_w @ w)
        w_normalized /= np.linalg.norm(w_normalized, ord=2)

        return w_normalized

    def fit(self, X: ArrayLike) -> None:
        """
        Fit the Fast_ICA model to the data matrix X.

        Parameters
        ----------
        X : ArrayLike
            Data matrix of shape (N, M).

        Returns
        -------
        None
        """
        # Preprocessing of X
        N, _ = X.shape
        X = Preprocessing.preprocessing(X)

        # Initialize random weights
        self.weights = np.random.random((self._n_components, N))
        for i in range(self._n_components):
            self.weights[i] = self._orthogonalisation(self.weights[i], self.weights[:i])

        # FastICA algorithm
        for p in range(self._n_components):

            # Initialize weights and old wieghts
            w_p = self.weights[p].copy()
            w_p_old = np.zeros_like(w_p)
            cpt = 0

            # Check convergence
            while cpt < self._max_iter and abs(np.inner(w_p, w_p_old)) < (
                1 - self._tol
            ):

                w_new = self._compute_new_weights(X, w_p)

                # Orthogonalization
                w_new = self._orthogonalisation(w_new, self.weights[:p])

                # Convergence check
                w_temp = w_p.copy()
                w_p, w_p_old = w_new.copy(), w_temp.copy()

                cpt += 1

            self.weights[p] = w_p

    def transform(self, X):
        """
        Project the data matrix X into the independent components space.

        Parameters
        ----------
        X : ArrayLike
            Data matrix of shape (N, M).

        Returns
        -------
        ArrayLike
            Data matrix projected into the independent components space of shape (N, M).
        """
        # Preprocessing of X
        X_whitened = Preprocessing.preprocessing(X)

        # Return the projection of X into the independent components space
        return self.weights @ X_whitened

    def fit_transform(self, X):
        """
        Apply the Fast_ICA algorithm to the data matrix X and return the projected data matrix into the independent components space.

        Parameters
        ----------
        X : ArrayLike
            Data matrix of shape (N, M).

        Returns
        -------
        ArrayLike
            Data matrix projected into the independent components space of shape (N, M).
        """
        # Fit the Fast_ICA model to the data matrix X
        self.fit(X)

        # Return the projection of X into the independent components space
        return self.transform(X)

    @property
    def W(self):
        """
        Return the unmixing matrix W.

        Returns
        -------
        ArrayLike
            Unmixing matrix of shape (N, N).
        """
        if hasattr(self, "weights"):
            return self.weights

    @property
    def A(self):
        """
        Return the mixing matrix A=W**-1.

        Returns
        -------
        ArrayLike
            Mixing matrix of shape (N, N).
        """
        if hasattr(self, "weights"):
            return np.linalg.pinv(self.weights)

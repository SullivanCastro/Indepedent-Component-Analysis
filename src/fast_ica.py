import numpy as np
from numpy import exp, diag
from numpy.linalg import eigh, pinv, norm
from numpy.typing import ArrayLike
from preprocessing import Preprocessing


class Fast_ICA:

    def __init__(self, n_components: int, func: str = "cubic", max_iter: int = 200, tol: float = 1e-4, a: float = 1) -> None:
        """
        Initialize the Fast_ICA object.

        Parameters
        ----------
        n_components : int
            Number of components to extract.
        max_iter : int, optional
            Maximum number of iterations (default is 200).
        tol : float, optional
            Tolerance for convergence (default is 1e-20).
        func : str, optional
            The non-linearity to use ('cubic', 'exp', or 'tanh').
        a : float, optional
            Scaling factor for 'tanh' non-linearity (default is 1).

        Returns
        -------
        None
        """
        self._max_iter = max_iter
        self._tol = tol
        self._n_components = n_components
        if func == "cubic":
            self._g = lambda X: X ** 3
            self._dg = lambda X: 3 * X ** 2
        elif func == "exp":
            self._g = lambda X: X * exp(-X**2 / 2)
            self._dg = lambda X: (1 - X**2) * exp(-X**2 / 2)
        elif func == "tanh":
            assert 1 <= a <= 2, "a must be in the range [1, 2]"
            self._g = lambda X: np.tanh(a * X)
            self._dg = lambda X: a * (1 - np.tanh(a * X) ** 2)


    def _compute_new_weights(self, X: ArrayLike, w: ArrayLike) -> ArrayLike:
        """
        Compute the new weights for the Fast_ICA algorithm using Newton's Method.

        Parameters
        ----------
        X : ArrayLike
            Data matrix of shape (M, T), where M is the number of features, and T is the number of samples.
        w : ArrayLike
            Current weight vector of shape (M,).

        Returns
        -------
        ArrayLike
            Updated weight vector of shape (M,).
        """
        Y = w @ X  # Projection of X onto w
        cov_g = np.mean(X * self._g(Y), axis=1)  # Shape: (M,)
        cov_dg = np.mean(self._dg(Y)) * w  # Shape: (M,)
        w_new = cov_g - cov_dg
        return w_new / norm(w_new, ord=2)


    def _orthogonalisation_iterative(self, w: ArrayLike, previous_w: ArrayLike) -> ArrayLike:
        """
        Orthogonalize the new weight vector w with respect to the previous weight vectors using Gram-Schmidt.

        Parameters
        ----------
        w : ArrayLike
            New weight vector of shape (M,).
        previous_w : ArrayLike
            Previous weight vectors of shape (K, M), where K < M is the number of already computed components.

        Returns
        -------
        ArrayLike
            Orthogonalized weight vector of shape (M,).
        """
        w -= previous_w.T @ (previous_w @ w)
        return w / norm(w, ord=2)


    def decorrelation(self, W: ArrayLike) -> ArrayLike:
        """
        Perform minimum distance unitary mapping for decorrelation.

        Parameters
        ----------
        W : ArrayLike
            Weight matrix of shape (N, M).

        Returns
        -------
        ArrayLike
            Decorrelated weight matrix of shape (N, M).
        """
        eigvals, eigvecs = eigh(W @ W.T)
        return eigvecs @ diag(1.0 / np.sqrt(eigvals)) @ eigvecs.T @ W
    

    def fit_newton(self, X: ArrayLike) -> None:
        """
        Fit the Fast_ICA model to the data matrix X using Newton's Method.

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
            self.weights[i] = self._orthogonalisation_iterative(self.weights[i], self.weights[:i])

        # FastICA algorithm
        for p in range(self._n_components):

            # Initialize weights and old weights
            w_p = self.weights[p].copy()
            w_p_old = np.zeros_like(w_p)
            cpt = 0

            # Check convergence
            while cpt < self._max_iter and abs(np.inner(w_p, w_p_old)) < (1 - self._tol):
                # Update weights
                w_new = self._compute_new_weights(X, w_p)

                # Orthogonalization
                w_new = self._orthogonalisation_iterative(w_new, self.weights[:p])

                # Convergence check
                w_temp = w_p.copy()
                w_p, w_p_old = w_new.copy(), w_temp.copy()

                cpt += 1

            self.weights[p] = w_p


    def fit_parallel(self, X: ArrayLike) -> None:
        """
        Fit the Fast_ICA model to the data matrix X using the parallel FastICA algorithm.

        Parameters
        ----------
        X : ArrayLike
            Data matrix of shape (M, T).

        Returns
        -------
        None
        """
        # Preprocessing of X
        X = Preprocessing.preprocessing(X)
        M, _ = X.shape
        W = np.random.random((self._n_components, M))  # Shape: (n_components, M)

        for _ in range(self._max_iter):
            W1 = np.array([
                self._compute_new_weights(X, w) for w in W
            ])  # Update all weights in parallel
            W1 = self.decorrelation(W1)  # Decorrelate weights

            # Check convergence
            if np.max(np.abs(np.abs(np.diag(W1 @ W.T)) - 1)) < self._tol:
                break
            W = W1

        self.weights = W


    def transform(self, X: ArrayLike) -> ArrayLike:
        """
        Project the data matrix X into the independent components space.

        Parameters
        ----------
        X : ArrayLike
            Data matrix of shape (M, T).

        Returns
        -------
        ArrayLike
            Data matrix projected into the independent components space of shape (N, T).
        """
        X = Preprocessing.preprocessing(X)
        return self.weights @ X


    def fit_transform(self, X: ArrayLike, method: str = 'parallel') -> ArrayLike:
        """
        Apply the Fast_ICA algorithm to the data matrix X and return the projected data matrix.

        Parameters
        ----------
        X : ArrayLike
            Data matrix of shape (M, T).
        method : str, optional
            Method to use for fitting ('iterative' for Newton, 'parallel' for parallel FastICA).

        Returns
        -------
        ArrayLike
            Data matrix projected into the independent components space of shape (N, T).
        """
        if method == 'iterative':
            self.fit_newton(X)
        else:
            self.fit_parallel(X)

        return self.transform(X)

    @property
    def W(self) -> ArrayLike:
        """
        Return the unmixing matrix W.

        Returns
        -------
        ArrayLike
            Unmixing matrix of shape (N, M).
        """
        return self.weights


    @property
    def A(self) -> ArrayLike:
        """
        Return the mixing matrix A, the pseudo-inverse of W.

        Returns
        -------
        ArrayLike
            Mixing matrix of shape (M, N).
        """
        return pinv(self.weights)

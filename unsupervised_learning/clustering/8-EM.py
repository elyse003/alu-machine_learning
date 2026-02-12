#!/usr/bin/env python3
"""
Expectation Maximization for a Gaussian Mixture Model
"""

import numpy as np
initialize = __import__('4-initialize').initialize
expectation = __import__('6-expectation').expectation
maximization = __import__('7-maximization').maximization


def expectation_maximization(X, k, iterations=1000, tol=1e-5, verbose=False):
    """
    Performs expectation maximization for a GMM

    Parameters
    ----------
    X : numpy.ndarray of shape (n, d)
        Dataset
    k : int
        Number of clusters
    iterations : int, optional
        Maximum number of iterations
    tol : float, optional
        Tolerance for log likelihood change
    verbose : bool, optional
        If True, prints log likelihood progress

    Returns
    -------
    pi : numpy.ndarray of shape (k,)
        Priors for each cluster
    m : numpy.ndarray of shape (k, d)
        Centroid means
    S : numpy.ndarray of shape (k, d, d)
        Covariance matrices
    g : numpy.ndarray of shape (k, n)
        Posterior probabilities
    l : float
        Log likelihood of the model
    """
    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None, None, None, None, None
    if not isinstance(k, int) or k <= 0:
        return None, None, None, None, None
    if not isinstance(iterations, int) or iterations <= 0:
        return None, None, None, None, None
    if not isinstance(tol, float) or tol < 0:
        return None, None, None, None, None
    if not isinstance(verbose, bool):
        return None, None, None, None, None

     pi, m, S = initialize(X, k)
    g, l = expectation(X, pi, m, S)
    prev_like = i = 0
    msg = "Log Likelihood after {} iterations: {}"

    for i in range(iterations):
        if verbose and i % 10 == 0:
            print(msg.format(i, total_log_like.round(5)))
        pi, m, S = maximization(X, g)
        g, total_log_like = expectation(X, pi, m, S)
        if abs(prev_like - total_log_like) <= tol:
            break
        prev_like = total_log_like

    if verbose:
        print(msg.format(i + 1, total_log_like.round(5)))

    return pi, m, S, g, log_like

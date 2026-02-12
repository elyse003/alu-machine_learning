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

    # Initialize parameters
    pi, m, S = initialize(X, k)
    g, log_like = expectation(X, pi, m, S)
    prev_like = log_like

    msg = "Log Likelihood after {} iterations: {}"

    for i in range(iterations):
        # Maximization step
        pi, m, S = maximization(X, g)
        # Expectation step
        g, log_like = expectation(X, pi, m, S)

        # Verbose printing
        if verbose and (i % 10 == 0 or i == iterations - 1):
            print(msg.format(i + 1, round(log_like, 5)))

        # Check for convergence
        if abs(log_like - prev_like) <= tol:
            break
        prev_like = log_like

    return pi, m, S, g, log_like

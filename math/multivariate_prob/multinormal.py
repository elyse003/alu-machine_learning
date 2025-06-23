#!/usr/bin/env python3
"""Represents a multivariate normal distribution and calculates PDF"""


import numpy as np


class MultiNormal:
    """
    Represents a multivariate normal distribution and PDF calculation

    Parameters:
    data(numpy.ndarray): Array of the shape(d, n) containing the dataset:
    - n: number of data points
    - d: number of dimensions in each data point
    """

    def __init__(self, data):
        if type(data) != np.ndarray:
            raise TypeError("data must be a 2D numpy.ndarray")
        if len(data.shape) != 2:
            raise TypeError("data must be a 2D numpy.ndarray")
        d, n = data.shape
        if n < 2:
            raise ValueError("data must contain multiple data points")

        self.mean = data.mean(axis=1).reshape(d, 1)
        centered_data = data - self.mean
        self.cov = (centered_data @ centered_data.T) / (n - 1)

    def pdf(self, x):
        """
        Calculates the PDF at a datapoint

        Parameters:
            x is a numpy.ndarray of shape (d,) containing the data point
                whose PDF should be calculated
                d is the number of dimensions of the Multinomial instance

        Returns:
            the PDF at x
        """
        if type(x) is not np.ndarray:
            raise TypeError("x must be a numpy.ndarray")
        d = self.cov.shape[0]
        if len(x.shape) != 2
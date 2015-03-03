#!/usr/bin/env python
# -*- coding: utf-8 -*-
r"""
Fama-McBeth regressions
=======================

This module provides two estimators of risk premia in Fama-McBeth regressions:
2-step OLS and GMM.

The inspiration for this code comes from Chapters 27.2-3 of
Kevin Sheppard's book "Python for Econometrics":
<http://www.kevinsheppard.com/images/0/09/Python_introduction.pdf>

The data with Fama-French risk factors:
<http://www.kevinsheppard.com/images/0/0b/FamaFrench.zip>

"""
from __future__ import print_function, division

import numpy as np
from scipy.stats import chi2

from mygmm import GMM

__author__ = "Stanislav Khrapov"
__email__ = "khrapovs@gmail.com"
__status__ = "Development"

__all__ = ['FamaMcBeth', 'convert_theta_to1d']


class FamaMcBeth(object):

    r"""Fama-McBeth regressions.

    Time series regression:
    :math:`E\left[R_{it} - \beta_i * f_t | f_t\right] = 0`
    and
    Cross-sectional regression:
    :math:`E\left[R_{it} - \beta_i * \gamma\right] = 0`

    Attributes
    ----------
    factors : (dim_t, dim_k) array
        Explanatory factors in the regression
    excess_ret : (dim_t, dim_n) array
        Portfolio excess returns that we are trying to explain

    Methods
    -------
    two_step_ols
        Two-step OLS estimator
    compute_theta_var
        Estimate variance of the 2-step OLS estimator
    gamma_tstat
        T-statistics for risk premia estimates
    jtest
        J-test for misspecification of the model

    """

    def __init__(self, factors, excess_ret):
        """Initialize the class.

        Parameters
        ----------
        factors : (dim_t, dim_k) array
            Explanatory factors in the regression
        excess_ret : (dim_t, dim_n) array
            Portfolio excess returns that we are trying to explain

        """
        # Store data internally.
        self.factors, self.excess_ret = factors, excess_ret

    def __get_dimensions(self):
        """Get essential dimentions of the data.

        Returns
        -------
        dim_t : int
            Time
        dim_n : int
            Number of portfolio returns to be explained
        dim_k : int
            Number of explanatory factors

        """
        dim_t, dim_n = self.excess_ret.shape
        dim_k = self.factors.shape[1]
        return dim_t, dim_n, dim_k

    def two_step_ols(self):
        """Two-step OLS estimator.

        Returns
        -------
        gamma : (dim_k, ) array
            Risk premia
        gamma_stde : (dim_k, ) array
            Standard errors
        gamma_rsq : (1, ) array
            R-squared for one cross-sectional regression
        theta : (dim_k, dim_n) array
            Risk exposures [alpha, beta]
        theta_stde : (dim_k, dim_n) array
            Standard errors
        theta_rsq : (dim_n, ) array
            R-squared for each time series regression

        """
        dim_t, dim_n, dim_k = self.__get_dimensions()
        # Time series regressions
        out = np.linalg.lstsq(self.factors, self.excess_ret)
        theta = out[0]
        theta_rmse = (out[1] / dim_t) ** .5
        theta_rsq = theta_rmse**2 / self.excess_ret.var(0)

        beta = theta[1:]
        mean_excess_ret = self.excess_ret.mean(0)
        # Cross-section regression
        out = np.linalg.lstsq(beta.T, mean_excess_ret.T)
        gamma = out[0]

        gamma_rmse = (out[1] / dim_n) ** .5
        gamma_rsq = gamma_rmse**2 / mean_excess_ret.var()

        return (gamma, gamma_rsq * 100, gamma_rmse,
                theta, theta_rsq * 100, theta_rmse)

    def gamma_tstat(self, gamma, theta_var):
        """T-statistics for risk premia estimates.

        Parameters
        ----------
        gamma : (dim_k,) array
            Risk premia
        theta_var : (dim_n*(dim_k+1), dim_n*(dim_k+1)) array
            Variance matrix of all parameters

        Returns
        -------
        (dim_k,) array

        """
        dim_n, dim_k = self.__get_dimensions()[1:]
        return gamma / np.diag(theta_var[dim_n*dim_k:, dim_n*dim_k:])**.5

    def param_stde(self, theta_var):
        """Standard errors for parameter estimates.

        Parameters
        ----------
        theta_var : (dim_n*(dim_k+1), dim_n*(dim_k+1)) array
            Variance matrix of all parameters

        Returns
        -------
        (dim_n*(dim_k+1), ) array

        """
        return np.diag(theta_var)**.5

    def param_tstat(self, theta, theta_var):
        """T-statistics for parameter estimates.

        Parameters
        ----------
        theta_var : (dim_n*(dim_k+1), dim_n*(dim_k+1)) array
            Variance matrix of all parameters

        Returns
        -------
        (dim_n*(dim_k+1), ) array

        """
        return theta / self.param_stde(theta_var)

    def gamma_theta_stde(self, gamma, theta):
        """Standard errors for parameter estimates.

        Parameters
        ----------
        gamma : (dim_k,) array
            Risk premia
        beta : (dim_k, dim_n) array
            Risk exposures

        Returns
        -------
        (dim_k, dim_n) array
            Stde for risk exposures
        (dim_k,) array
            Stde for risk premia

        """
        param_var = self.compute_theta_var(gamma, theta)
        stde = self.param_stde(param_var)
        dim_n, dim_k = self.__get_dimensions()[1:]
        beta_stde = np.reshape(stde[:dim_n*dim_k], (dim_n, dim_k)).T
        gamma_stde = stde[dim_n*dim_k:]
        return beta_stde, gamma_stde

    def jtest(self, theta, theta_var):
        """J-test for misspecification of the model.

        Parameters
        ----------
        theta : (dim_k, dim_n) array
            Risk exposures
        theta_var : (dim_n*(dim_k+1), dim_n*(dim_k+1)) array
            Variance matrix of all parameters

        Returns
        -------
        jstat : int
            J-statistic
        jpval : int
            Corresponding p-value of the test

        """

        dim_n, dim_k = self.__get_dimensions()[1:]
        alpha = theta[0]
        alpha_var = theta_var[0:dim_n*dim_k:dim_k, 0:dim_n*dim_k:dim_k]
        inv_var = np.linalg.inv(alpha_var)
        jstat = alpha.dot(inv_var).dot(alpha[np.newaxis, :].T)[0]
        jpval = 1 - chi2(dim_n).cdf(jstat)
        return jstat, jpval*100

    def convert_theta_to2d(self, theta):
        """Convert parameter vector to matrices.

        Parameters
        ----------
        theta : (dim_n*(dim_k+1), ) array

        Returns
        -------
        alpha : (dim_n, ) array
            Intercepts in time series regressions
        beta : (dim_k, dim_n) array
            Risk exposures
        gamma : (dim_k, ) array
            Risk premia

        """
        dim_n, dim_k = self.__get_dimensions()[1:]
        alpha = theta[:dim_n]
        beta = np.reshape(theta[dim_n:dim_n*dim_k], (dim_n, dim_k-1))
        gamma = np.reshape(theta[dim_n*dim_k:], (dim_k-1, 1))
        return alpha, beta, gamma

    def momcond(self, theta, **kwargs):
        """Moment restrictions and avergae of its derivatives.

        Parameters
        ----------
        theta : (dim_n*(dim_k+1),) array

        Returns
        -------
        moments : (dim_t, dim_n*(dim_k+1)) array
            Moment restrictions
        dmoments : (dim_k*(dim_n+1), dim_n*(dim_k+1)) array
            Average derivative of the moment restrictions

        """

        dim_t, dim_n, dim_k = self.__get_dimensions()
        alpha, beta, gamma = self.convert_theta_to2d(theta)

        errors1 = self.excess_ret - alpha - self.factors[:, 1:].dot(beta.T)
        moments1 = errors1[:, :, np.newaxis] * self.factors[:, np.newaxis, :]
        moments1 = moments1.reshape(dim_t, dim_n*dim_k)

        errors2 = self.excess_ret - beta.dot(gamma).T
        moments2 = errors2.dot(beta)

        moments = np.hstack((moments1, moments2))

        dmoments = np.zeros(((dim_n+1)*dim_k-1, (dim_n+1)*dim_k-1))
        factor_var = self.factors.T.dot(self.factors) / dim_t
        dmoments[:dim_n*dim_k, :dim_n*dim_k] = np.kron(np.eye(dim_n), factor_var)
        dmoments[dim_n*dim_k:, dim_n*dim_k:] = -beta.T.dot(beta)

        for i in range(dim_n):
            temp = np.zeros((dim_k-1, dim_k))
            values = np.mean(errors2[:, i]) - beta.T[:, i] * gamma
            temp[:, 1:] = np.diag(values)
            dmoments[dim_n*dim_k:, i*dim_k:(i+1)*dim_k] = temp

        return moments, dmoments.T

    def compute_theta_var(self, gamma, theta, kernel='SU'):
        """Estimate variance of the 2-step OLS estimator.

        Parameters
        ----------
        gamma : (dim_k,) array
            Risk premia
        theta : (dim_k, dim_n) array
            Risk exposures

        Returns
        -------
        ((dim_n+1)*dim_k-1, (dim_n+1)*dim_k-1) array
            Variance matrix of the estimator

        """
        estimator = GMM(self.momcond)
        theta = convert_theta_to1d(theta[0], theta[1:], gamma)
        return estimator.varest(theta, kernel=kernel)

    def gmmest(self, theta, **kwargs):
        """Estimate model parameters using GMM.

        """
        estimator = GMM(self.momcond)
        return estimator.gmmest(theta, **kwargs)

    def callback(self, theta):
        """Callback function to run after each optimization iteration.

        Parameters
        ----------
        theta : (dim_n*(dim_k+1),) array

        """
        pass


def convert_theta_to1d(alpha, beta, gamma):
    """Convert parameter matrices to 1d vector.

    Parameters
    ----------
    alpha : (dim_n, ) array
        Intercepts in time series regressions
    beta : (dim_k, dim_n) array
        Risk exposures
    gamma : (dim_k,) array
        Risk premia

    Returns
    -------
    (dim_n*(dim_k+1),) array

    """
    return np.concatenate((alpha, beta.flatten(), gamma))


if __name__ == '__main__':
    pass

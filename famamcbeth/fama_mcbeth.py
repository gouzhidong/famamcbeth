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

import warnings

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
        Explanatory factors in the regression,
        including constant in the first place
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
            Explanatory factors in the regression,
            including constant in the first place
        excess_ret : (dim_t, dim_n) array
            Portfolio excess returns that we are trying to explain

        """
        # Store data internally.
        self.factors = factors
        self.excess_ret = excess_ret

    def __get_dimensions(self):
        """Get essential dimentions of the data.

        Returns
        -------
        dim_t : int
            Time
        dim_n : int
            Number of portfolio returns to be explained
        dim_k : int
            Number of explanatory factors, including constant

        """
        dim_t, dim_n = self.excess_ret.shape
        dim_k = self.factors.shape[1]
        return dim_t, dim_n, dim_k

    def two_step_ols(self):
        """Two-step OLS estimator.

        Returns
        -------
        theta : (dim_k*(dim_n+1)-1, ) array
            Parameter vector
        gamma_stde : (dim_k, ) array
            Standard errors
        gamma_rsq : (1, ) array
            R-squared for one cross-sectional regression
        theta_stde : (dim_k, dim_n) array
            Standard errors
        theta_rsq : (dim_n, ) array
            R-squared for each time series regression

        """
        dim_t, dim_n, dim_k = self.__get_dimensions()
        # Time series regressions
        # (dim_k, dim_n) array. This theta includes intercepts alpha
        theta, resid = np.linalg.lstsq(self.factors, self.excess_ret)[:2]
        # float
        theta_rmse = (resid / dim_t) ** .5
        # float
        theta_rsq = 100 * (1 - theta_rmse**2 / self.excess_ret.var(0))

        # (dim_n, ) array
        alpha = theta[0]
        # (dim_k-1, dim_n) array
        beta = theta[1:]
        # (dim_n, ) array
        mean_excess_ret = self.excess_ret.mean(0)
        # Cross-section regression
        # (dim_k-1, ) array
        gamma, resid = np.linalg.lstsq(beta.T, mean_excess_ret.T)[:2]
        # float
        gamma_rmse = (resid / dim_n) ** .5
        # float
        gamma_rsq = 1 - gamma_rmse**2 / mean_excess_ret.var()
#        gamma_rsq = 1 - (1 - gamma_rsq) * (dim_n - 1) / (dim_n - dim_k - 1)
        gamma_rsq *= 100

        param = convert_theta_to1d(alpha, beta, gamma)

        return param, gamma_rsq, gamma_rmse, theta_rsq, theta_rmse

    def param_stde(self, theta, **kwargs):
        """Standard errors for parameter estimates.

        Parameters
        ----------
        theta : (dim_k*(dim_n+1)-1, ) array
            Parameter vector

        Returns
        -------
        (dim_k*(dim_n+1)-1, ) array

        """
        var = self.compute_theta_var(theta, **kwargs)
        return np.abs(np.diag(var))**.5

    def param_tstat(self, theta, **kwargs):
        """T-statistics for parameter estimates.

        Parameters
        ----------
        theta : (dim_k*(dim_n+1)-1, ) array
            Parameter vector

        Returns
        -------
        (dim_k*(dim_n+1)-1, ) array

        """
        return theta / self.param_stde(theta, **kwargs)

    def alpha_beta_gamma_stde(self, theta, **kwargs):
        """Standard errors for parameter estimates.

        Parameters
        ----------
        theta : (dim_k*(dim_n+1)-1, ) array
            Parameter vector

        Returns
        -------
        alpha_stde : (dim_n, ) array
            Intercepts in time series regressions
        beta_stde : (dim_k-1, dim_n) array
            Risk exposures
        gamma_stde : (dim_k-1, ) array
            Risk premia

        """
        stde = self.param_stde(theta, **kwargs)
        return self.convert_theta_to2d(stde)

    def alpha_beta_gamma_tstat(self, theta, **kwargs):
        """Standard errors for parameter estimates.

        Parameters
        ----------
        theta : (dim_k*(dim_n+1)-1, ) array
            Parameter vector

        Returns
        -------
        alpha_tstat : (dim_n, ) array
            Intercepts in time series regressions
        beta_tstat : (dim_k-1, dim_n) array
            Risk exposures
        gamma_tstat : (dim_k-1, ) array
            Risk premia

        """
        tstat = self.param_tstat(theta, **kwargs)
        return self.convert_theta_to2d(tstat)

    def jtest(self, theta, **kwargs):
        """J-test for misspecification of the model.

        Tests whether all intercepts alphas are simultaneously zero.

        Parameters
        ----------
        theta : (dim_k*(dim_n+1)-1, ) array
            Parameter vector

        Returns
        -------
        jstat : int
            J-statistic
        jpval : int
            Corresponding p-value of the test, percent

        """

        dim_n, dim_k = self.__get_dimensions()[1:]
        param_var = self.compute_theta_var(theta, **kwargs)
        alpha_var = param_var[0:dim_n*dim_k:dim_k, 0:dim_n*dim_k:dim_k]
        eig = np.linalg.eigvalsh(alpha_var).min()
        if eig <= 0:
            alpha_var -= np.eye(dim_n) * eig * 1.1
        inv_var = np.linalg.pinv(alpha_var)
        try:
            np.linalg.cholesky(inv_var)
        except np.linalg.LinAlgError:
            warnings.warn('Inverse of alpha variance is not P.D.!')
        alpha = self.convert_theta_to2d(theta)[0]
        jstat = (alpha.dot(inv_var) * alpha).sum()
        jpval = 1 - chi2(dim_n).cdf(jstat)
        return jstat, jpval*100

    def convert_theta_to2d(self, theta):
        """Convert parameter vector to matrices.

        Parameters
        ----------
        theta : (dim_k*(dim_n+1)-1, ) array

        Returns
        -------
        alpha : (dim_n, ) array
            Intercepts in time series regressions
        beta : (dim_k-1, dim_n) array
            Risk exposures
        gamma : (dim_k-1, ) array
            Risk premia

        """
        dim_n, dim_k = self.__get_dimensions()[1:]
        temp = np.reshape(theta[:dim_n*dim_k], (dim_n, dim_k)).T
        alpha = temp[0]
        beta = temp[1:]
        gamma = theta[dim_n*dim_k:]
        return alpha, beta, gamma

    def momcond(self, theta, **kwargs):
        """Moment restrictions and avergae of its derivatives.

        Parameters
        ----------
        theta : (dim_k*(dim_n+1)-1, ) array

        Returns
        -------
        moments : (dim_t, dim_n*(dim_k+1)) array
            Moment restrictions
        dmoments : (dim_k*(dim_n+1), dim_n*(dim_k+1)) array
            Average derivative of the moment restrictions

        """

        dim_t, dim_n, dim_k = self.__get_dimensions()
        alpha, beta, gamma = self.convert_theta_to2d(theta)

        errors1 = self.excess_ret - alpha - self.factors[:, 1:].dot(beta)
        moments1 = errors1[:, :, np.newaxis] * self.factors[:, np.newaxis, :]
        # (dim_t, dim_n*dim_k) array
        moments1 = moments1.reshape(dim_t, dim_n*dim_k)

        # (dim_t, dim_n) array
        errors2 = self.excess_ret - beta.T.dot(gamma)
        # (dim_t, dim_k-1) array
        moments2 = errors2.dot(beta.T)

        # (dim_t, (dim_n+1)*dim_k-1) array
        moments = np.hstack((moments1, moments2))

        dmoments = np.zeros(((dim_n+1)*dim_k-1, (dim_n+1)*dim_k-1))
        # (dim_k, dim_k) array
        factor_var = self.factors.T.dot(self.factors) / dim_t
        eye = np.eye(dim_n)
        # (dim_n*dim_k, dim_n*dim_k) array
        dmoments[:dim_n*dim_k, :dim_n*dim_k] = np.kron(eye, factor_var)
        # (dim_k-1, dim_k-1) array
        dmoments[dim_n*dim_k:, dim_n*dim_k:] = -beta.dot(beta.T)

        for i in range(dim_n):
            temp = np.zeros((dim_k-1, dim_k))
            values = np.mean(errors2[:, i]) - beta[:, i] * gamma
            temp[:, 1:] = np.diag(values)
            dmoments[dim_n*dim_k:, i*dim_k:(i+1)*dim_k] = temp

        return moments, dmoments.T

    def compute_theta_var(self, theta, **kwargs):
        """Estimate variance of the estimator using GMM variance matrix.

        Parameters
        ----------
        theta : (dim_k*(dim_n+1)-1, ) array
            Risk exposures

        Returns
        -------
        (dim_k*(dim_n+1)-1, dim_k*(dim_n+1)-1) array
            Variance matrix of the estimator

        """
        estimator = GMM(self.momcond)
        return estimator.varest(theta, **kwargs)

    def gmmest(self, theta, **kwargs):
        """Estimate model parameters using GMM.

        """
        estimator = GMM(self.momcond)
        return estimator.gmmest(theta, **kwargs)

    def get_realized_ret(self):
        """Estimate variance of the estimator using GMM variance matrix.

        Returns
        -------
        (dim_n, ) array
            Realized average (across time) returns

        """
        return self.excess_ret.mean(0)

    def get_predicted_ret(self, param):
        """Estimate variance of the estimator using GMM variance matrix.

        Parameters
        ----------
        param : (dim_k*(dim_n+1)-1, ) array
            Model parameters

        Returns
        -------
        (dim_n, ) array
            Predicted average (across time) returns

        """
        beta, gamma = self.convert_theta_to2d(param)[1:]
        return beta.T.dot(gamma)


def convert_theta_to1d(alpha, beta, gamma):
    """Convert parameter matrices to 1d vector.

    Parameters
    ----------
    alpha : (dim_n, ) array
        Intercepts in time series regressions
    beta : (dim_k-1, dim_n) array
        Risk exposures
    gamma : (dim_k,) array
        Risk premia

    Returns
    -------
    (dim_k*(dim_n+1)-1, ) array

    """
    beta = np.vstack((alpha, beta)).T
    return np.concatenate((beta.flatten(), gamma))


if __name__ == '__main__':
    pass

#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Fama-McBeth regressions.

This module provides two estimators of risk premia in Fama-McBeth regressions:
2-step OLS and GMM.

The inspiration for this code comes from Chapters 27.2-3 of
Kevin Sheppard's book "Python for Econometrics"
http://www.kevinsheppard.com/images/0/09/Python_introduction.pdf

The data with Fama-French risk factors:
http://www.kevinsheppard.com/images/0/0b/FamaFrench.zip

GMM estimator is located here:
https://github.com/khrapovs/MyGMM

"""

from __future__ import print_function, division

import numpy as np
from scipy.stats import chi2

from MyGMM.gmm import GMM

__author__ = "Stanislav Khrapov"
__email__ = "khrapovs@gmail.com"
__status__ = "Development"

class FamaMcBeth(GMM):
    """Fama-McBeth regressions.
    
    It is assumed that:
    Time series regression: E[R_it - beta_i * f_t | f_t] = 0
    and
    Cross-sectional regression: E[R_it - beta_i * gamma] = 0
    
    """
    
    def __init__(self, factors, excess_ret):
        super(FamaMcBeth, self).__init__()
        self.factors, self.excess_ret = factors, excess_ret
    
    def __get_dimensions(self):
        T, N = self.excess_ret.shape
        K = self.factors.shape[1]
        return T, N, K
        
    def two_step_ols(self):
        T, N, K = self.__get_dimensions()
        # Time series regressions
        out = np.linalg.lstsq(self.factors, self.excess_ret)
        theta = out[0]
        beta = theta[1:]
        mean_excess_ret = self.excess_ret.mean(0)
        # Cross-section regression
        gamma = np.linalg.lstsq(beta.T, mean_excess_ret.T)[0]
        
        return gamma, theta
    
    def compute_theta_var(self, gamma, theta):
        T, N, K = self.__get_dimensions()
        beta = theta[1:]
        # Moment conditions
        errors1 = self.excess_ret - self.factors.dot(theta)
        moments1 = errors1[:,:,np.newaxis] * self.factors[:,np.newaxis,:]
        moments1 = moments1.reshape(T,N*K)
        
        errors2 = self.excess_ret - gamma.T.dot(beta)
        moments2 = errors2.dot(beta.T)
        # Score covariance
        S = np.cov(np.hstack((moments1, moments2)).T)
        # Jacobian
        G = np.zeros_like(S)
        SigmaX = self.factors.T.dot(self.factors) / T
        
        G[:N*K, :N*K] = np.kron(np.eye(N), SigmaX)
        G[N*K:, N*K:] = -beta.dot(beta.T)

        for i in range(N):
            temp = np.zeros((K-1, K))
            values = np.mean(errors2[:, i]) - beta[:, i] * gamma
            temp[:, 1:] = np.diag(values)
            G[N*K:, i*K:(i+1)*K] = temp
        
        invG = np.linalg.inv(G)
        vcv = invG.T.dot(S).dot(invG) / T
        return vcv
        
    def gamma_tstat(self, gamma, theta_var):
        N, K = self.__get_dimensions()[1:]
        return gamma / np.diag(theta_var[N*K:, N*K:])**.5
    
    def jtest(self, theta, theta_var):
        T, N, K = self.__get_dimensions()
        alpha = theta[0]
        alpha_var = theta_var[0:N*K:K, 0:N*K:K]
        inv_var = np.linalg.inv(alpha_var)
        Jstat = alpha.dot(inv_var).dot(alpha[np.newaxis, :].T)[0]
        Jpval = 1 - chi2(N).cdf(Jstat)
        return Jstat, Jpval
    
    def convert_theta_to1d(self, beta, gamma):
        return np.concatenate((beta.flatten(), gamma))
        
    def convert_theta_to2d(self, theta):
        N, K = self.__get_dimensions()[1:]
        K -= 1
        beta = np.reshape(theta[:N*K], (N, K))
        gamma = np.reshape(theta[N*K:], (K, 1))
        return beta, gamma
        
    def moment(self, theta):
        T, N, K = self.__get_dimensions()
        K -= 1
        beta, gamma = self.convert_theta_to2d(theta)
        factors = self.factors[:, 1:]
        errors1 = self.excess_ret - factors.dot(beta.T)
        moments1 = errors1[:, :, np.newaxis] * factors[:, np.newaxis, :]
        moments1 = moments1.reshape(T, N*K)
        moments2 = self.excess_ret - beta.dot(gamma).T
        moments = np.hstack((moments1, moments2))
        
        dmoments = np.zeros((K*(N+1), N*(K+1)))
        factor_var = factors.T.dot(factors) / T
        dmoments[:N*K, :N*K] = np.kron(np.eye(N), factor_var)
        dmoments[:N*K, N*K:] = np.kron(np.eye(N), -gamma)
        dmoments[N*K:, N*K:] = -beta.T
        
        return moments, dmoments.T
    
    def callback(self, theta):
        T, N, K = self.__get_dimensions()
        K -= 1
        # print(theta[-K:])

if __name__ == '__main__':
    import usage_example
    usage_example.test_default()
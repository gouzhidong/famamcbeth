# -*- coding: utf-8 -*-
"""

Chapters 27.2-3 from
http://www.kevinsheppard.com/images/0/09/Python_introduction.pdf

Fama-French risk factors:
http://www.kevinsheppard.com/images/0/0b/FamaFrench.zip

GMM estimator is located here:
https://github.com/khrapovs/MyGMM

"""

import pandas as pd
import numpy as np
import datetime as dt
from scipy.stats import chi2

from MyGMM.gmm import GMM

np.set_printoptions(precision=3, suppress=True)

class FamaMcBeth(GMM):
    
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
        risk_premia = np.linalg.lstsq(beta.T, mean_excess_ret.T)[0]
        
        return risk_premia, theta
    
    def compute_theta_var(self, risk_premia, theta):
        T, N, K = self.__get_dimensions()
        beta = theta[1:]
        # Moment conditions
        errors1 = self.excess_ret - self.factors.dot(theta)
        moments1 = errors1[:,:,np.newaxis] * self.factors[:,np.newaxis,:]
        moments1 = moments1.reshape(T,N*K)
        
        errors2 = self.excess_ret - risk_premia.T.dot(beta)
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
            values = np.mean(errors2[:, i]) - beta[:, i] * risk_premia
            temp[:, 1:] = np.diag(values)
            G[N*K:, i*K:(i+1)*K] = temp
        
        invG = np.linalg.inv(G)
        vcv = invG.T.dot(S).dot(invG) / T
        return vcv
        
    def risk_premia_tstat(self, risk_premia, theta_var):
        N, K = self.__get_dimensions()[1:]
        return risk_premia / np.diag(theta_var[N*K:, N*K:])**.5
    
    def jtest(self, theta, theta_var):
        T, N, K = self.__get_dimensions()
        alpha = theta[0]
        alpha_var = theta_var[0:N*K:K, 0:N*K:K]
        inv_var = np.linalg.inv(alpha_var)
        Jstat = alpha.dot(inv_var).dot(alpha[np.newaxis, :].T)[0]
        Jpval = 1 - chi2(N).cdf(Jstat)
        return Jstat, Jpval
    
    def convert_theta_to1d(self, beta, risk_premia):
        return np.concatenate((beta.flatten(), risk_premia))
        
    def convert_theta_to2d(self, theta):
        N, K = self.__get_dimensions()[1:]
        K -= 1
        beta = np.reshape(theta[:N*K], (N, K))
        risk_premia = np.reshape(theta[N*K:], (K, 1))
        return beta, risk_premia
        
    def moment(self, theta):
        T, N, K = self.__get_dimensions()
        K -= 1
        beta, risk_premia = self.convert_theta_to2d(theta)
        factors = self.factors[:, 1:]
        errors1 = self.excess_ret - factors.dot(beta.T)
        moments1 = errors1[:, :, np.newaxis] * factors[:, np.newaxis, :]
        moments1 = moments1.reshape(T, N*K)
        moments2 = self.excess_ret - beta.dot(risk_premia).T
        moments = np.hstack((moments1, moments2))
        
        dmoments = np.zeros((K*(N+1), N*(K+1)))
        factor_var = factors.T.dot(factors) / T
        dmoments[:N*K, :N*K] = np.kron(np.eye(N), factor_var)
        dmoments[:N*K, N*K:] = np.kron(np.eye(N), -risk_premia)
        dmoments[N*K:, N*K:] = -beta.T
        
        return moments, dmoments.T
    
    def callback(self, theta):
        T, N, K = self.__get_dimensions()
        K -= 1
        # print(theta[-K:])
        
def import_data():
    parse = lambda x: dt.datetime.strptime(x, '%Y%m')
    date_name = 'date'
    factor_names = ['VWMe', 'SMB', 'HML']
    rf_name = 'RF'
    data = pd.read_csv('FamaFrench.csv', index_col=date_name,
                     parse_dates=date_name, date_parser=parse)
    
    riskfree = data[[rf_name]].values
    factors = data[factor_names].values
    # Augment factors with the constant
    factors = np.hstack((np.ones_like(riskfree), factors))
    portfolios = data[data.columns - factor_names - [rf_name]].values
    excess_ret = portfolios - riskfree
    
    return factors, excess_ret

def test_default():
    factors, excess_ret = import_data()
    model = FamaMcBeth(factors, excess_ret)
    risk_premia, beta = model.two_step_ols()
    beta_var = model.compute_theta_var(risk_premia, beta)
    Jstat, Jpval = model.jtest(beta, beta_var)
    tstat = model.risk_premia_tstat(risk_premia, beta_var)
    print(risk_premia)
    print(tstat * 12**.5)
    print('J-stat = %.2f, p-value = %.2f' % (Jstat, Jpval))
    
    theta = model.convert_theta_to1d(beta[1:], risk_premia)
    model.method = 'Powell'
    model.gmmest(theta)
    K = factors.shape[1]
    
    print(model.theta[-K:])
    print(model.tstat[-K:])
    print('J-stat = %.2f, p-value = %.2f' % (model.jstat, model.pval))
    
def import_cay():
    import calendar
    calendar.monthrange(2002,1)
    parse = lambda x: dt.datetime.strptime(x, '%Y\:%q')
    date_name = 'date'

    def parse(value):
        year = int(value[:4])
        month = 3*int(value[5:])
        day = calendar.monthrange(year, month)[1]
        return dt.datetime(year, month, day)
        
    cay = pd.read_csv('cay_q_13Q3.csv', index_col=date_name,
                       parse_dates=date_name, date_parser=parse)[['cay']]
    
    parse = lambda x: dt.datetime.strptime(x, '%Y%m')
    date_name = 'date'
    rf_name = 'RF'
    data = pd.read_csv('FamaFrench.csv', index_col=date_name,
                     parse_dates=date_name, date_parser=parse)
    ff_factors = data.resample('Q')
    data = pd.merge(cay, ff_factors, left_index=True, right_index=True)
    
    factor_names = ['cay', 'VWMe', 'SMB', 'HML']
    riskfree = data[[rf_name]].values
    factors = data[factor_names].values
    # Augment factors with the constant
    factors = np.hstack((np.ones_like(riskfree), factors))
    portfolios = data[data.columns - factor_names - [rf_name]].values
    excess_ret = portfolios - riskfree
    
    return factors, excess_ret

def test_with_cay():
    factors, excess_ret = import_cay()
    model = FamaMcBeth(factors, excess_ret)
    risk_premia, beta = model.two_step_ols()
    beta_var = model.compute_theta_var(risk_premia, beta)
    Jstat, Jpval = model.jtest(beta, beta_var)
    tstat = model.risk_premia_tstat(risk_premia, beta_var)
    print(risk_premia)
    print(tstat)
    print('J-stat = %.2f, p-value = %.2f' % (Jstat, Jpval))
    
    theta = model.convert_theta_to1d(beta[1:], risk_premia)
    model.method = 'Powell'
    model.gmmest(theta)
    K = factors.shape[1]
    
    print(model.theta[-K:])
    print(model.tstat[-K:])
    print('J-stat = %.2f, p-value = %.2f' % (model.jstat, model.pval))

if __name__ == '__main__':
    test_default()
    test_with_cay()
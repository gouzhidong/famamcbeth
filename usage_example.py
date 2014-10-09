#!/usr/bin/env python
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

from FamaMcBeth.fama_mcbeth import FamaMcBeth

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
    tstat = model.gamma_tstat(risk_premia, beta_var)
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
    

if __name__ == '__main__':
    np.set_printoptions(precision=3, suppress=True)
    test_default()
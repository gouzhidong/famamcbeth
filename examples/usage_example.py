#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Usage examples

"""

import pandas as pd
import numpy as np
import datetime as dt

from famamcbeth import FamaMcBeth, convert_theta_to1d


def import_data():
    parse = lambda x: dt.datetime.strptime(x, '%Y%m')
    date_name = 'date'
    factor_names = ['VWMe', 'SMB', 'HML']
    rf_name = 'RF'
    data = pd.read_csv('../data/FamaFrench.csv', index_col=date_name,
                       parse_dates=date_name, date_parser=parse)

    riskfree = data[[rf_name]].values
    factors = data[factor_names].values
    # Augment factors with the constant
    factors = np.hstack((np.ones_like(riskfree), factors))
    portfolios = data[data.columns - factor_names - [rf_name]].values
    excess_ret = portfolios - riskfree

    return factors, excess_ret


def test_default():

    np.set_printoptions(precision=3, suppress=True)

    factors, excess_ret = import_data()
    model = FamaMcBeth(factors, excess_ret)
    risk_premia, beta = model.two_step_ols()
    beta_var = model.compute_theta_var(risk_premia, beta)
    jstat, jpval = model.jtest(beta, beta_var)
    tstat = model.gamma_tstat(risk_premia, beta_var)
    print(risk_premia)
    print(tstat * 12**.5)
    print('J-stat = %.2f, p-value = %.2f' % (jstat, jpval))

    theta = convert_theta_to1d(beta[1:], risk_premia)
    model.method = 'Powell'
    res = model.gmmest(theta)
    K = factors.shape[1]

    print(res.theta[-K:])
    print(res.tstat[-K:])
    print('J-stat = %.2f, p-value = %.2f'
          % (res.jstat, res.jpval))


if __name__ == '__main__':
    test_default()
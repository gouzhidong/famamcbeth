#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Usage examples

"""

import pandas as pd
import numpy as np
import datetime as dt
import matplotlib.pylab as plt
import seaborn as sns

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
    (param, gamma_rsq, gamma_rmse, theta_rsq, theta_rmse) \
        = model.two_step_ols()
    alpha, beta, gamma = model.convert_theta_to2d(param)

    kernel = 'Bartlett'
    band = 3
    jstat, jpval = model.jtest(param, kernel=kernel, band=band)
    tstat = model.alpha_beta_gamma_tstat(param, kernel=kernel, band=band)
    alpha_tstat, beta_tstat, gamma_tstat = tstat

    print('OLS results:')
    print(gamma)
    print(gamma_tstat)
    print('J-stat = %.2f, p-value = %.2f\n' % (jstat, jpval))

    method = 'L-BFGS-B'
    res = model.gmmest(param, kernel=kernel, band=band, method=method)
    param_final = model.convert_theta_to2d(res.theta)
    alpha_final, beta_final, gamma_final = param_final
    tstat_final = model.convert_theta_to2d(res.tstat)
    alpha_tstat, beta_tstat, gamma_tstat = tstat_final

    print('GMM results:')
    print(gamma_final)
    print(gamma_tstat)
    jstat, jpval = model.jtest(res.theta, kernel=kernel, band=band)
    print('J-stat = %.2f, p-value = %.2f' % (res.jstat, res.jpval))
    print('J-stat = %.2f, p-value = %.2f' % (jstat, jpval))

    ret_realized = model.get_realized_ret()
    ret_predicted = model.get_predicted_ret(res.theta)

    plt.scatter(ret_realized, ret_predicted)
    x = np.linspace(*plt.gca().get_xlim())
    plt.gca().plot(x, x)
    plt.xlabel('Realized')
    plt.ylabel('Predicted')
    plt.show()


if __name__ == '__main__':

    sns.set_context('paper')
    test_default()

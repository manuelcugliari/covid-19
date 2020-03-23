"""
    File name: compare.py
    Author: Manuel Cugliari
    Date created: 20/03/2020
    Python Version: 3.7.4
"""
import argparse
import logging
import itertools
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from fastdtw import fastdtw
from scipy.optimize import curve_fit
from sklearn.metrics import mean_squared_error
from src.settings import *
from src.utils import parser, parser_world

logging.basicConfig(level=logging.DEBUG)
mpl_logger = logging.getLogger('matplotlib')
mpl_logger.setLevel(logging.WARNING)


def _get_args():
    parser = argparse.ArgumentParser(
        description='This script explore the 5-parameter Stannard curve for COVID-19 infection dynamic',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('-l', '--compare', type=str, help='compare Italy with another country', default='Italy, Hubei')

    return parser.parse_args()


def func_lw(x_lw, slope, offset):
    return slope * x_lw + offset


def func_lw_inverse(y_lw, slope, offset):
    return (y_lw - offset) / slope


'''
y0, is the lower asymptote = 0
A, is the upper asymptote;
u, is the time of maximum growth;
d, is the growth rate and
v, is a variable which fixes the point of inflection.
'''


def stannard_4(x, A, u, d, v):
    """Richards growth model (equivalent to Stannard).
    Proposed in Zwietering et al., 1990 (PMID: 16348228)
    """
    y = (
            A
            * pow(
        1
        + (
                v * (np.exp(1 + v) * np.exp((u / A) * (1 + v) * (1 + (1 / v)) * (d - x)))
        ),
        -(1 / v),
    )
    )
    return y


def guess_lag(x, y):
    """Given two axes returns a guess of the lag point.
    The lag point is defined as the x point where the difference in y
    with the next point is higher then the mean differences between
    the points plus one standard deviation. If such point is not found
    or x and y have different lengths the function returns zero.
    """
    if len(x) != len(y):
        return 0

    diffs = []
    indexes = range(len(x))

    for i in indexes:
        if i + 1 not in indexes:
            continue
        diffs.append(y[i + 1] - y[i])
    diffs = np.array(diffs)

    flex = x[-1]
    for i in indexes:
        if i + 1 not in indexes:
            continue
        if (y[i + 1] - y[i]) > (diffs.mean() + (diffs.std())):
            flex = x[i]
            break

    return flex


def guess_plateau(x, y):
    """Given two axes returns a guess of the plateau point.
    The plateau point is defined as the x point where the y point
    is near one standard deviation of the differences between the y points to
    the maximum y value. If such point is not found or x and y have
    different lengths the function returns zero.
    """
    if len(x) != len(y):
        return 0

    diffs = []
    indexes = range(len(y))

    for i in indexes:
        if i + 1 not in indexes:
            continue
        diffs.append(y[i + 1] - y[i])
    diffs = np.array(diffs)

    ymax = y[-1]
    for i in indexes:
        if y[i] > (ymax - diffs.std()) and y[i] < (ymax + diffs.std()):
            ymax = y[i]
            break

    return ymax


def fit(function, x, y):
    """Fit the provided function to the x and y values.
    The function parameters and the parameters covariance.
    """
    # Compute guesses for the parameters
    # This is necessary to get significant fits
    # p0 = [guess_plateau(x, y), 4.0, guess_lag(x, y), 0.1, min(y)]
    p0 = [guess_plateau(x, y), 4.0, guess_lag(x, y), 0.1]

    # params, pcov = curve_fit(function, x, y, p0=p0, method='lm')
    params, pcov = curve_fit(function, x, y, method='lm')
    return params, pcov


if __name__ == '__main__':
    args = _get_args()

    dir_italy = ITALY_RAW_DATA_FOLDER
    dir_world = WORLD_RAW_DATA_FOLDER
    dataset_italy = 'dpc-covid19-ita-andamento-nazionale'
    dataset_world = 'covid_19_data'

    df_italy = pd.read_csv(dir_italy / f'{dataset_italy}.csv', parse_dates=[0], squeeze=True, date_parser=parser)
    df_world = pd.read_csv(dir_world / f'{dataset_world}.csv', parse_dates=[1], squeeze=True, date_parser=parser_world)

    LIST = args.compare
    countries = LIST.split(',')
    if countries[0].strip() != 'Italy':
        raise ValueError(f'First country must be Italy')

    if countries[1].strip() not in supported_coutries:
        raise ValueError(f'Supported contexts are: {supported_coutries}')

    country = countries[1].strip()

    df_world = df_world[df_world['Province/State'] == str(countries[1].strip())]

    t_italy = np.arange(df_italy['data'].values.shape[0])
    t_world = np.arange(df_world['ObservationDate'].values.shape[0])
    l_world = df_world['Confirmed'].to_numpy()
    l_italy = df_italy['totale_casi'].to_numpy()

    popt_world, pcov_world = fit(stannard_4, t_world, l_world)
    print(popt_world)
    print(pcov_world)

    """
    Plot section growth rate curve
    """
    fig1 = plt.figure()
    l_world_diff = np.diff(l_world)
    l_italy_diff = np.diff(l_italy)
    # plt.plot(t_world[0:t_world.shape[0] - 1], l_world_diff,
    #          'bo', alpha=0.4, label='Hubei - growth rate totale_casi')
    # plt.plot(t_italy[0:t_italy.shape[0] - 1], l_italy_diff,
    #          'go', alpha=0.4, label='Italy - growth rate totale_casi')
    sorted_growth_italy = -np.sort(-l_italy_diff)
    params_lw, pcov_lw = curve_fit(func_lw, t_italy[t_italy.shape[0] - 4:t_italy.shape[0]],
                                   l_italy[l_italy.shape[0] - 4:l_italy.shape[0]], method='lm')
    ax = fig1.gca()
    plt.grid()
    # plt.suptitle('Hubei vs Italy - Growth rate')
    # plt.legend(loc='best')

    y_star = 0
    delta_init = (y_star - params_lw[1]) / params_lw[0]
    mu_m_init = params_lw[0]

    A = 100000
    u = sorted_growth_italy[0]
    d = delta_init
    v = popt_world[3]
    print('A: ' + str(A))
    print('u: ' + str(u))
    print('d: ' + str(d))
    print('v: ' + str(v))

    A_delta_lower = 0.2 * A
    A_delta_upper = 0.8 * A
    A_grain = 1000
    u_delta_lower = 0
    u_delta_upper = 0.4 * u
    u_grain = 30
    d_delta_lower = 0
    d_delta_upper = 0.4 * d
    d_grain = 30
    v_delta_lower = 0.1 * v
    v_delta_upper = 0.4 * v
    v_grain = 30

    A_range = np.linspace(A - A_delta_lower, A + A_delta_upper, A_grain)
    u_range = np.linspace(u - u_delta_lower, u + u_delta_upper, u_grain)
    d_range = np.linspace(d - d_delta_lower, d + d_delta_upper, d_grain)
    v_range = np.linspace(v - v_delta_lower, v + v_delta_upper, v_grain)
    print('A_range: ' + str(A_range))
    print('u_range: ' + str(u_range))
    print('d_range: ' + str(d_range))
    print('v_range: ' + str(v_range))

    iterables = [A_range, u_range, d_range, v_range]
    distances_dtw = {}
    mses = {}
    distance_dtw_min = 1.00e+10
    distance_mse_min = 1.00e+10
    t_dtw_star = [0, 0, 0, 0]
    t_mse_star = [0, 0, 0, 0]
    for t in itertools.product(*iterables):
        distance, path = fastdtw(l_italy, stannard_4(t_world[0:t_italy.shape[0]], *t))
        mse = mean_squared_error(l_italy, stannard_4(t_world[0:t_italy.shape[0]], *t))

        if distance < distance_dtw_min:
            distance_dtw_min = distance
            t_dtw_star = t

        if mse < distance_mse_min:
            distance_mse_min = mse
            t_mse_star = t

    print('distance_min_dtw: ' + str(distance_dtw_min))
    print('params_dtw: ' + str(t_dtw_star))

    print('distance_min_mse: ' + str(distance_mse_min))
    print('params_mse: ' + str(t_mse_star))

    """
    Plot section growth curve 
    """
    for (dimension_it, dimension_it_values) in df_italy.iteritems():
        if dimension_it == 'totale_casi':
            plt.plot(t_italy, dimension_it_values, 'go', alpha=0.4, linewidth=1, label='Italy - totale_casi')
            for (dimension_wo, dimension_wo_values) in df_world.iteritems():
                if dimension_wo == 'Confirmed':
                    plt.plot(t_world, dimension_wo_values,
                             'bo', alpha=0.4,
                             label=country + ' totale_casi')
            del df_italy[dimension_it]

    plt.plot(t_world, stannard_4(t_world, *popt_world), 'b', linewidth=1, alpha=0.4,
             label='Hubei - Stannard curve fitting')

    # label_dtw = 'A=' + str(round(t_dtw_star[0], 2)) + ' u=' + str(round(t_dtw_star[1], 2)) + ' d=' + str(
    #     round(t_dtw_star[2], 2)) + ' v=' + str(round(t_dtw_star[3], 2))
    # plt.plot(t_world, stannard_4(t_world, *t_dtw_star), 'g--', linewidth=1,
    #          label='DWT - Fitting & Forecast Italy ' + label_dtw)

    label_mse = 'A=' + str(round(t_mse_star[0], 2)) + ' u=' + str(round(t_mse_star[1], 2)) + ' d=' + str(
        round(t_mse_star[2], 2)) + ' v=' + str(round(t_mse_star[3], 2))
    plt.plot(t_world, stannard_4(t_world, *t_mse_star), 'g-', linewidth=1,
             label='Italy - Stannard curve fitting & forecast')

    plt.legend(loc='best')
    plt.show()

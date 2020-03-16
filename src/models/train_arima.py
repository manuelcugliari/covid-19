"""
    File name: train_arima.py
    Author: Manuel Cugliari
    Date created: 15/03/2020
    Python Version: 3.7.4
"""
import logging

import matplotlib.pyplot as plt
import pandas as pd
from statsmodels.tsa.arima_model import ARIMA

from src.settings import header, region_code
from src.utils import parser

logging.basicConfig(level=logging.DEBUG)


def plot_prediction_with_test(X, test, predictions, size, region, dimension):
    label1 = "History"
    label2 = "True future values"
    label3 = "Prediction future values"

    title = 'Predictions v.s. true values '
    if region == 'all':
        title = title + dimension + ' for Italy'
    else:
        title = title + dimension + ' for region ' + ' for region ' + region_code[int(region)]

    plt.suptitle(title)
    plt.plot(range(len(X)), X, linewidth=1, color='g', alpha=0.4, label=label1)
    plt.plot(range(size, size + len(test)), test, "x--b", linewidth=1, label=label2)
    plt.plot(range(size, size + len(predictions)), predictions, "o--y", linewidth=1, label=label3)
    plt.legend(loc='best')
    plt.show()


def plot_prediction(offet, predictions, region, dimension):
    label = "Prediction"

    title = 'Predictions v.s. true values '
    if region == 'all':
        title = title + dimension + ' for Italy'
    else:
        title = title + dimension + ' for region ' + ' for region ' + region_code[int(region)]

    plt.suptitle(title)
    plt.plot(range(offet, offet + len(predictions)), predictions, "x--b", linewidth=1, label=label)
    plt.legend(loc='best')
    plt.show()


def train_ARIMA(dir, dataset, region, dimension, p, d, q, aforward):
    if region == 'all':
        # df_master = pd.read_csv(dir / f'{dataset}.csv', parse_dates=[0], date_parser=parser)
        df_master = pd.read_csv(dir / f'{dataset}.csv', index_col=0, parse_dates=[0], date_parser=parser)
    else:
        df_master = pd.read_csv(dir / f'{dataset}.csv', parse_dates=[0],
                                index_col=0, date_parser=parser,
                                dtype={'codice_regione': 'Int64',
                                       list(header.keys())[0]: 'Int64',
                                       list(header.keys())[1]: 'Int64',
                                       list(header.keys())[2]: 'Int64',
                                       list(header.keys())[3]: 'Int64',
                                       list(header.keys())[4]: 'Int64',
                                       list(header.keys())[5]: 'Int64',
                                       list(header.keys())[6]: 'Int64',
                                       list(header.keys())[7]: 'Int64',
                                       list(header.keys())[8]: 'Int64',
                                       list(header.keys())[9]: 'Int64'})

        df_master = df_master.loc[df_master['codice_regione'] == int(region)]
    if dimension == 'all':
        for key in header.items():
            df_master = df_master.filter(['data', key])
            core_ARIMA(df_master, p, d, q, dimension, region, aforward)
    else:
        df_master = df_master.filter(['data', dimension])
        core_ARIMA(df_master, p, d, q, dimension, region, aforward)


def core_ARIMA(df_master, p, d, q, dimension, region, aforward):

    model = ARIMA(df_master.values, order=(p, d, q))
    fit_model = model.fit(trend='c', full_output=True, disp=True)

    forecast = fit_model.forecast(steps=aforward)
    pred_y = forecast[0].tolist()

    print(pd.DataFrame(pred_y))

    fig = fit_model.plot_predict()
    plt.show()

    title = 'Forecast vs Actual for '
    if region == 'all':
        plt.title(title + 'Italy')
    else:
        plt.title(title + ' region ' + region)

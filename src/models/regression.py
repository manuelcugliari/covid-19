"""
    File name: regression.py
    Author: Manuel Cugliari
    Date created: 14/03/2020
    Python Version: 3.7.4
"""
import logging
from datetime import timedelta

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.neural_network import MLPRegressor

from src.settings import header
from src.utils import parser

logging.basicConfig(level=logging.DEBUG)


def train_REGRESSION(dir, dataset, region, dimension):
    if region == 'all':
        df_master = pd.read_csv(dir / f'{dataset}.csv', parse_dates=[0], date_parser=parser)
    else:
        df_master = pd.read_csv(dir / f'{dataset}.csv', parse_dates=[0], date_parser=parser,
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
            core_REGRESSION(df_master, dimension, region)
    else:
        df_master = df_master.filter(['data', dimension])
        core_REGRESSION(df_master, dimension, region)


def core_REGRESSION(df_master, dimension, region):
    x = np.arange(len(df_master)).reshape(-1, 1)
    y = df_master[dimension].values
    model = MLPRegressor(hidden_layer_sizes=[32, 32, 10], max_iter=50000, alpha=0.0005, random_state=26)
    model.fit(x, y)

    test = np.arange(len(df_master) + 7).reshape(-1, 1)
    pred = model.predict(test)
    prediction = pred.round().astype(int)

    week = [df_master['data'][0] + timedelta(days=i) for i in range(len(prediction))]
    dt_idx = pd.DatetimeIndex(week)

    basic = [df_master['data'][0] + timedelta(days=i) for i in range(len(y))]
    actual_count = pd.Series(y, basic)
    predicted_count = pd.Series(prediction, dt_idx)

    plt.plot(actual_count)
    plt.plot(predicted_count)
    plt.xticks(rotation=40)
    plt.title('Prediction of ' + dimension + ' for ' + region)
    plt.legend(['Observed ' + dimension, 'predicted ' + dimension])
    plt.show()
    plt.pause(10)


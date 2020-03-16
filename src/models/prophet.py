"""
    File name: prophet.py
    Author: Manuel Cugliari
    Date created: 14/03/2020
    Python Version: 3.7.4
"""
import logging

import matplotlib.pyplot as plt
import pandas as pd
from fbprophet import Prophet
from fbprophet.plot import add_changepoints_to_plot

from src.settings import header
from src.utils import parser

logging.basicConfig(level=logging.DEBUG)


def train_PROPHET(dir, dataset, region, dimension, changepoint_prior_scale, periods):
    if region == 'all':
        df_master = pd.read_csv(dir / f'{dataset}.csv', parse_dates=[0], date_parser=parser)
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
            core_PROPHET(df_master, dimension, region, changepoint_prior_scale, periods)
    else:
        df_master = df_master.filter(['data', dimension])
        core_PROPHET(df_master, dimension, region, changepoint_prior_scale, periods)


def core_PROPHET(df_master, dimension, region, changepoint_prior_scale, periods):
    df_master.rename(columns={df_master.columns[0]: 'ds',
                              df_master.columns[1]: 'y'},
                     inplace=True)

    m = Prophet(changepoint_prior_scale=changepoint_prior_scale)
    m.fit(df_master)
    future = m.make_future_dataframe(periods=periods)
    forecast = m.predict(future)

    if region == 'all':
        region = 'Italy'

    fig = m.plot(forecast, xlabel='Date', ylabel=dimension + ' for ' + region)
    add_changepoints_to_plot(fig.gca(), m, forecast)
    m.plot_components(forecast)
    plt.pause(10)

    forecast.rename(columns={'ds': 'data',
                             'yhat': 'predizione',
                             'yhat_lower': 'intervallo_inferiore',
                             'yhat_upper': 'intervallo_superiore'},
                    inplace=True)
    print(forecast[['data', 'predizione', 'intervallo_inferiore', 'intervallo_superiore']].tail(periods))

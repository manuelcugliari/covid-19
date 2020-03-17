"""
    File name: train.py
    Author: Manuel Cugliari
    Date created: 15/03/2020
    Python Version: 3.7.4
"""
import argparse
import logging

from src.models.prophet import train_PROPHET
from src.models.regression import train_REGRESSION
from src.models.train_arima import train_ARIMA
from src.models.train_rnn_seq2seq import train_RNN_seq2seq
from src.settings import *

logging.basicConfig(level=logging.DEBUG)


def _get_args():
    parser = argparse.ArgumentParser(
        description=('This script trains the models for COVID-19 forecast '
                     'estimation.'),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--country', type=str, default='italy',
                        help='context to use, italy or world')

    parser.add_argument('--region', type=str, default='all',
                        help='region code to analyze, use all for national trend')

    parser.add_argument('--dimension', type=str, default='totale_casi',
                        help='dimension to predict, ricoverati_con_sintomi, terapia_intensiva, totale_ospedalizzati, '
                             'isolamento_domiciliare, totale_attualmente_positivi, nuovi_attualmente_positivi, '
                             'dimessi_guariti, deceduti, totale_casi, tamponi')

    parser.add_argument('--model', type=str, default='prophet',
                        help='model to use, arima or RNN_seq2seq or prophet or regression')

    parser.add_argument('--initial_lrate', type=float, default=1E-03,
                        help='initial learning rate')
    parser.add_argument('--epochs', type=int, default=50,
                        help='number of epochs')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='batch size')

    parser.add_argument('--lag_observations', type=float, default=2,
                        help='the number of lag observations included in the models, also called the lag order')
    parser.add_argument('--degree_differencing', type=int, default=2,
                        help='the number of times that the raw observations are differenced, also called the degree '
                             'of differencing.')
    parser.add_argument('--moving_average', type=int, default=0,
                        help='the size of the moving average window, also called the order of moving average')
    parser.add_argument('--aforward', type=int, default=7,
                        help='number of ARIMA forecast days')

    parser.add_argument('--changepoint_prior_scale', type=float, default=0.5,
                        help='adjust the strength of the sparse prior')
    parser.add_argument('--pforward', type=int, default=7,
                        help='number of Prophet forecast days')

    return parser.parse_args()


def _make_hparam_string():
    return (f'{MODEL}'
            f'_lr{INITIAL_LRATE:.0E}_batch{BATCH_SIZE}')


if __name__ == '__main__':
    args = _get_args()

    COUNTRY = args.country
    supported_countries = ('italy', 'world')
    if COUNTRY not in supported_countries:
        raise ValueError(f'Supported contexts are: {supported_countries}')

    REGION = args.region

    MODEL = args.model
    supported_models = ('arima', 'RNN_seq2seq', 'prophet', 'regression')
    if MODEL not in supported_models:
        raise ValueError(f'Supported models are: {supported_models}')

    DIMENSION = args.dimension
    supported_dimensions = ('ricoverati_con_sintomi', 'terapia_intensiva',
                            'totale_ospedalizzati', 'isolamento_domiciliare', 'totale_attualmente_positivi',
                            'nuovi_attualmente_positivi',
                            'dimessi_guariti', 'deceduti', 'totale_casi', 'tamponi')

    if DIMENSION not in supported_dimensions:
        raise ValueError(f'Supported dimensions are: {supported_dimensions}')

    INITIAL_LRATE = args.initial_lrate
    EPOCHS = args.epochs
    BATCH_SIZE = args.batch_size

    LAG_OBSERVATIONS = args.lag_observations
    DEGREE_DIFFERENCING = args.degree_differencing
    MOVING_AVERAGE = args.moving_average
    AFORWARD = args.aforward

    CHANGEPOINTS_PRIOR_SCALE = args.changepoint_prior_scale
    PFORWARD = args.pforward

    MODEL_NAME = f'{_make_hparam_string()}'
    MODELS_FOLDER.mkdir(parents=True, exist_ok=True)

    logging.debug(MODEL_NAME)

    if MODEL == 'arima':
        if COUNTRY == 'italy' and REGION == 'all':
            train_ARIMA(ITALY_RAW_DATA_FOLDER, country_dataset, REGION, DIMENSION,
                        LAG_OBSERVATIONS, DEGREE_DIFFERENCING, MOVING_AVERAGE, AFORWARD)
        else:
            train_ARIMA(ITALY_RAW_DATA_FOLDER, region_dataset, REGION, DIMENSION,
                        LAG_OBSERVATIONS, DEGREE_DIFFERENCING, MOVING_AVERAGE, AFORWARD)

    if MODEL == 'RNN_seq2seq':
        if COUNTRY == 'italy' and REGION == 'all':
            train_RNN_seq2seq(ITALY_RAW_DATA_FOLDER, country_dataset,
                              REGION, DIMENSION)
        else:
            train_RNN_seq2seq(ITALY_RAW_DATA_FOLDER, 'none',
                              REGION, DIMENSION)

    if MODEL == 'prophet':
        if COUNTRY == 'italy' and REGION == 'all':
            train_PROPHET(ITALY_RAW_DATA_FOLDER, country_dataset,
                              REGION, DIMENSION, CHANGEPOINTS_PRIOR_SCALE, PFORWARD)
        else:
            train_PROPHET(ITALY_RAW_DATA_FOLDER, 'none',
                              REGION, DIMENSION, CHANGEPOINTS_PRIOR_SCALE, PFORWARD)

    if MODEL == 'regression':
        if COUNTRY == 'italy' and REGION == 'all':
            train_REGRESSION(ITALY_RAW_DATA_FOLDER, country_dataset,
                              REGION, DIMENSION)
        else:
            train_REGRESSION(ITALY_RAW_DATA_FOLDER, 'none',
                              REGION, DIMENSION)


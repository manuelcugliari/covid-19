"""
    File name: compare.py
    Author: Manuel Cugliari
    Date created: 14/03/2020
    Python Version: 3.7.4
"""
import argparse
import logging
from datetime import timedelta

import matplotlib.pyplot as plt
import pandas as pd

from src.settings import *
from src.utils import parser, parser_world, save_csv

logging.basicConfig(level=logging.DEBUG)


def _get_args():
    parser = argparse.ArgumentParser(
        description='This script compare data for COVID-19 pandemic',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('-l', '--compare', type=str, help='compare Italy with another country', default='Italy, Hubei')

    return parser.parse_args()


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
    dataset = 'covid_19_data_' + country
    save_csv(WORLD_PROCESSED_DATA_FOLDER, dataset, df_world)

    fig1, axs = plt.subplots(2, 1, constrained_layout=True, figsize=(6, 6))
    for nn, ax in enumerate(axs):
        for (dimension_it, dimension_it_values) in df_italy.iteritems():
            if dimension_it == 'deceduti' \
                    or dimension_it == 'totale_casi':

                if dimension_it == 'totale_casi':
                    ax.plot(df_italy['data'], dimension_it_values, "x--r", linewidth=0.5, label='totale_casi Italy')
                    for (dimension_wo, dimension_wo_values) in df_world.iteritems():
                        if dimension_wo == 'Confirmed':
                            ax.plot(df_world['ObservationDate'] + timedelta(days=33), dimension_wo_values, "x--b",
                                    linewidth=0.5,
                                    label='totale_casi ' + country)

                if dimension_it == 'deceduti':
                    ax.plot(df_italy['data'], dimension_it_values, "x--r", linewidth=0.5, label='deceduti Italy')
                    for (dimension_wo, dimension_wo_values) in df_world.iteritems():
                        if dimension_wo == 'Deaths':
                            ax.plot(df_world['ObservationDate'], dimension_wo_values, "x--b", linewidth=0.5,
                                    label='deceduti ' + country)

                if nn == len(axs) - 1:
                    for label in ax.get_xticklabels():
                        label.set_rotation(40)
                        label.set_horizontalalignment('right')
                else:
                    ax.set_xticklabels([])

                ax.legend(loc='best')

                del df_italy[dimension_it]
                break

    plt.suptitle('Country Italy vs ' + countries[1].strip() + ' trend')
    plt.show()

"""
    File name: explore.py
    Author: Manuel Cugliari
    Date created: 14/03/2020
    Python Version: 3.7.4
"""
import argparse
import logging

import matplotlib.pyplot as plt
import pandas as pd
from pandas.plotting import autocorrelation_plot

from src.settings import *
from src.utils import parser, parser_autorrelation

logging.basicConfig(level=logging.DEBUG)


def _get_args():
    parser = argparse.ArgumentParser(
        description='This script explores data for COVID-19 pandemic',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--country', type=str, default='italy',
                        help='country to use, italy or world')

    parser.add_argument('--region', type=str, default='all',
                        help='code region to use, all or specific region code')

    parser.add_argument('-l', '--compare', type=str, help='delimited region code list', default='3')

    return parser.parse_args()


def plot_region(region, master, lang):
    fig1, axs = plt.subplots(5, 1, constrained_layout=True, figsize=(6, 6))
    for nn, ax in enumerate(axs):
        for (dimension_it, dimension_it_values) in master.iteritems():
            if dimension_it == 'terapia_intensiva' \
                    or dimension_it == 'totale_ospedalizzati' \
                    or dimension_it == 'totale_attualmente_positivi' \
                    or dimension_it == 'deceduti' \
                    or dimension_it == 'totale_casi':
                if lang == 'it':
                    ax.plot(master['data'], dimension_it_values, "x--b", linewidth=1, label=dimension_it)
                else:
                    ax.plot(master['data'], dimension_it_values, linewidth=1, label=header[dimension_it])

                if nn == len(axs) - 1:
                    for label in ax.get_xticklabels():
                        label.set_rotation(40)
                        label.set_horizontalalignment('right')
                else:
                    ax.set_xticklabels([])

                ax.legend(loc='best')

                del master[dimension_it]
                break

    plt.suptitle('Region ' + region + ' trend')
    plt.show()


def compare_region(regions, master, lang):
    fig1, axs = plt.subplots(5, 1, constrained_layout=True, figsize=(6, 6))
    for nn, ax in enumerate(axs):
        for item in master:
            for (dimension_it, dimension_it_values) in item.iteritems():
                if dimension_it == 'terapia_intensiva' \
                        or dimension_it == 'totale_ospedalizzati' \
                        or dimension_it == 'totale_attualmente_positivi' \
                        or dimension_it == 'deceduti' \
                        or dimension_it == 'totale_casi':
                    if lang == 'it':
                        ax.plot(item['data'], dimension_it_values, "x--b", linewidth=1, label=dimension_it)
                    else:
                        ax.plot(item['data'], dimension_it_values, linewidth=1, label=header[dimension_it])

                    if nn == len(axs) - 1:
                        for label in ax.get_xticklabels():
                            label.set_rotation(40)
                            label.set_horizontalalignment('right')
                    else:
                        ax.set_xticklabels([])

                    ax.legend(loc='best')

                    del item[dimension_it]
                    break

    plt.suptitle('Regions ' + ' '.join([str(elem) for elem in regions]) + ' comparison')
    plt.show()


def plot_autocorrelation(region, master, lang):
    '''
    Autocorrelation plots are often used for checking randomness in time series. This is done by computing
    autocorrelations for data values at varying time lags. If time series is random, such autocorrelations should be
    near zero for any and all time-lag separations. If time series is non-random then one or more of the
    autocorrelations will be significantly non-zero. The horizontal lines displayed in the plot correspond to 95% and
    99% confidence bands. The dashed line is 99% confidence band.
    '''
    for (dimension_it, dimension_it_values) in master.iteritems():
        df_m = master
        print(dimension_it)
        if dimension_it == 'terapia_intensiva' \
                or dimension_it == 'totale_ospedalizzati' \
                or dimension_it == 'totale_attualmente_positivi' \
                or dimension_it == 'deceduti' \
                or dimension_it == 'totale_casi':
            if lang == 'it':
                autocorrelation_plot(df_m)
            del master[dimension_it]
        else:
            del df_m[dimension_it]

    plt.suptitle('Autocorrelation dimensions for region ' + region)
    plt.show()
    plt.pause(1)


if __name__ == '__main__':
    args = _get_args()

    COUNTRY = args.country
    supported_contexts = ('italy', 'world')
    if COUNTRY not in supported_contexts:
        raise ValueError(f'Supported contexts are: {supported_contexts}')

    REGION = args.region

    dir = ITALY_RAW_DATA_FOLDER
    if COUNTRY == 'italy' and REGION == 'all':
        dataset = 'dpc-covid19-ita-andamento-nazionale'
        df_master = pd.read_csv(dir / f'{dataset}.csv', parse_dates=[0],
                                squeeze=True, date_parser=parser)
        plot_region('Italy', df_master, 'it')

        df_master = pd.read_csv(dir / f'{dataset}.csv', parse_dates=[0],
                                squeeze=True, date_parser=parser)
        plot_autocorrelation('Italy', df_master, 'it')
    else:
        dataset = 'dpc-covid19-ita-regioni'

        df_master = pd.read_csv(dir / f'{dataset}.csv', parse_dates=[0],
                                squeeze=True, date_parser=parser)
        for code, region in region_code.items():
            if code == 12:
                df_region = df_master[df_master['codice_regione'] == code]
                plot_region(region, df_region, 'it')

        df_master = pd.read_csv(dir / f'{dataset}.csv', parse_dates=[0],
                                index_col=0, squeeze=True, date_parser=parser_autorrelation)
        for code, region in region_code.items():
            if code == 12:
                df_region = df_master[df_master['codice_regione'] == code]
                plot_autocorrelation(region, df_region, 'it')

        df_master = pd.read_csv(dir / f'{dataset}.csv', parse_dates=[0],
                                squeeze=True, date_parser=parser)
        LIST = args.compare
        regions = []
        dfs_region = []
        for item in LIST.split(','):
            df_region = df_master[df_master['codice_regione'] == int(item)]
            regions.append(region_code.get(int(item)))
            dfs_region.append(df_region)
        compare_region(regions, dfs_region, 'it')

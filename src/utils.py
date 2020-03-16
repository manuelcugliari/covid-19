"""
    File name: utils.py
    Author: Manuel Cugliari
    Date created: 15/03/2020
    Python Version: 3.7.4
"""
import json
import sys
from datetime import datetime, date
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

PROJECT_ROOT = Path(__file__).resolve().parents[0]
RAW_DATA_FOLDER = PROJECT_ROOT / 'data' / 'raw'
GA_RAW_DATA_FOLDER = PROJECT_ROOT / 'data' / 'raw' / 'italy'
SEO_RAW_DATA_FOLDER = PROJECT_ROOT / 'data' / 'raw' / 'seo'
PROCESSED_DATA_FOLDER = PROJECT_ROOT / 'data' / 'processed'
TRAINING_DATA_FOLDER = PROJECT_ROOT / 'data' / 'training'

sys.path.append(str(PROJECT_ROOT / 'src'))

naming_ga_raw = 'Analytics IT.W.O.1 - it.iqos.com (production)'
naming_ga_aggregated = 'ga_aggregated_training_dataset'


def parser_world(x):
    dt = datetime.strptime(x, '%m/%d/%Y')
    return date(dt.year, dt.month, dt.day)


def parser(x):
    dt = datetime.strptime(x, '%Y-%m-%d %H:%M:%S')
    return date(dt.year, dt.month, dt.day)


def parser_autorrelation(x):
    return datetime.strptime(x, '%Y-%m-%d %H:%M:%S')


def read_csv(dir, dataset):
    if naming_ga_raw in dataset:
        df = pd.read_csv(dir / f'{dataset}', sep=',', skiprows=6)
        return df
    if naming_ga_aggregated in dataset:
        df = pd.read_csv(dir / f'{dataset}.csv', sep=',')
        return df


def save_csv(dir, dataset, df):
    output_file = dir / f'{dataset}.csv'
    print(f'Saving to {output_file}')
    df.to_csv(output_file, index=False)


def save_json(obj, output_path, filename):
    filename = output_path / filename
    filename = filename.with_suffix('.json')
    filename.parent.mkdir(parents=True, exist_ok=True)

    with open(filename, 'w') as json_file:
        json.dump(obj, json_file)


def convert_ga_format_date(time):
    dmy = time.split('/')
    m = str(dmy[0])
    d = str(dmy[1])
    y = str('20' + dmy[2])
    if len(d) == 1:
        d = '0' + d
    if len(m) == 1:
        m = '0' + m
    time = d + '-' + m + '-' + y

    return time


def short_format_date(time):
    dmy = time.split('-')
    m = str(dmy[0])
    d = str(dmy[1])
    if len(d) == 1:
        d = '0' + d
    if len(m) == 1:
        m = '0' + m
    time = d + '-' + m

    return time


def plot_prediction(x, y_true, y_pred, master, train_size):
    """Plots the predictions.

    Arguments
    ---------
    x: Input sequence of shape (input_sequence_length,
        dimension_of_signal)
    y_true: True output sequence of shape (input_sequence_length,
        dimension_of_signal)
    y_pred: Predicted output sequence (input_sequence_length,
        dimension_of_signal)
    """

    plt.figure(figsize=(12, 3))

    offset = train_size

    output_dim = x.shape[-1]
    for j in range(output_dim):
        past = x[:, j]
        true = y_true[:, j]
        pred = y_pred[:, j]

        label1 = "Seen (past) values" if j == 0 else "_nolegend_"
        label2 = "True future values" if j == 0 else "_nolegend_"
        label3 = "Predictions" if j == 0 else "_nolegend_"

        plt.plot(range(offset, offset + len(past)), past, "o--b", linewidth=1, label=label1)
        plt.plot(range(offset + len(past), offset + len(true) + len(past)), true, "x--b", linewidth=1, label=label2)
        plt.plot(range(offset + len(past), offset + len(pred) + len(past)), pred, "o--y", linewidth=1, label=label3)

    label4 = "original"
    plt.plot(range(len(master)), master, label=label4, linewidth=1, color='g', alpha=0.4)

    plt.legend(loc='best')
    plt.title("Predictions v.s. true values")
    plt.show()


def batch_data_generator(dataset, batch_size, steps_per_epoch,
                         input_sequence_length, target_sequence_length, seed, train):
    if dataset.shape[0] - input_sequence_length - target_sequence_length < 0:
        input_sequence_length = int(dataset.shape[0] / 2)
        target_sequence_length = int(dataset.shape[0] / 2)

    num_points = input_sequence_length + target_sequence_length

    # if not train:
    #     if num_points > dataset.shape[0]:
    #         input_sequence_length = round(dataset.shape[0]/4)
    #         target_sequence_length = round(dataset.shape[0]/4)
    #         num_points = input_sequence_length + target_sequence_length

    while True:
        np.random.seed(seed)
        for _ in range(steps_per_epoch):
            train = np.array(dataset)
            signals = np.zeros((batch_size, num_points))
            for x in range(batch_size):
                index = np.random.randint(low=0,
                                          high=dataset.shape[0] - input_sequence_length - target_sequence_length,
                                          size=1)
                start = int(index[0])
                end = int(index[0]) + input_sequence_length + target_sequence_length
                signals[x] = train[start:end]
            signals = np.expand_dims(signals, axis=2)

            encoder_input = signals[:, :input_sequence_length, :]
            decoder_output = signals[:, input_sequence_length:, :]

            # The output of the generator must be ([encoder_input, decoder_input], [decoder_output])
            decoder_input = np.zeros((decoder_output.shape[0], decoder_output.shape[1], 1))
            yield ([encoder_input, decoder_input], decoder_output)


def get_scaler(original_data):
    values = original_data.reshape((len(original_data), 1))
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaler = scaler.fit(values)
    return scaler


def normalize(dataset):
    # prepare data for normalization
    values = dataset.reshape((len(dataset), 1))
    # train the normalization
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaler = scaler.fit(values)
    # print('Min: %f, Max: %f' % (scaler.data_min_, scaler.data_max_))
    # normalize the dataset and print
    normalized = scaler.transform(values)
    # print(normalized)
    # inverse transform and print
    inversed = scaler.inverse_transform(normalized)
    # print(inversed)

    return normalized

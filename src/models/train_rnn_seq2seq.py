"""
    File name: train_rnn_seq2seq.py
    Author: Manuel Cugliari
    Date created: 16/03/2020
    Python Version: 3.7.4
"""
import logging

import keras
import numpy as np
import pandas as pd
from keras.callbacks import TensorBoard, ModelCheckpoint

from src.settings import *
from src.utils import plot_prediction, parser, batch_data_generator, get_scaler

logging.basicConfig(level=logging.DEBUG)


def _make_hparam_string(model, input_sequence_length, target_sequence_length,
                        initial_lrate, batch_size,
                        train_percentage,
                        epochs, decay,
                        num_steps_to_predict, num_layers,
                        steps_per_epoch):
    return (f'{model}_input-length{input_sequence_length}_target-length{target_sequence_length}'
            f'_lr{initial_lrate:.0E}_batch{batch_size}'
            f'_train-percentage{train_percentage}'
            f'_epochs{epochs}_decay{decay}'
            f'_num-steps-to-pred{num_steps_to_predict}_decay{num_layers}'
            f'_step-epoch{steps_per_epoch}')


def _make_hparam_string_light(model, input_sequence_length, target_sequence_length,
                              initial_lrate, batch_size, epochs, decay):
    return (f'{model}_input-length{input_sequence_length}_target-length{target_sequence_length}'
            f'_lr{initial_lrate:.0E}_batch{batch_size}'
            f'_epoch{epochs}'f'_decay{decay}')


def predict(x, encoder_predict_model, decoder_predict_model, num_steps_to_predict, batch_size):
    """Predict time series with encoder-decoder.

    Uses the encoder and decoder RNN_seq2seq previously trained to predict the next
    num_steps_to_predict values of the time series.

    Arguments
    ---------
    x: input time series of shape (batch_size, input_sequence_length, input_dimension).
    encoder_predict_model: The Keras encoder models.
    decoder_predict_model: The Keras decoder models.
    num_steps_to_predict: The number of steps in the future to predict

    Returns
    -------
    y_predicted: output time series for shape (batch_size, target_sequence_length,
        ouput_dimension)
    """
    y_predicted = []

    # Encode the values as a state vector
    states = encoder_predict_model.predict(x)

    # The states must be a list
    if not isinstance(states, list):
        states = [states]

    # Generate first value of the decoder input sequence
    decoder_input = np.zeros((x.shape[0], 1, 1))

    # batch_size = 512
    for _ in range(num_steps_to_predict):
        outputs_and_states = decoder_predict_model.predict(
            [decoder_input] + states, batch_size=batch_size)
        output = outputs_and_states[0]
        states = outputs_and_states[1:]

        # add predicted value
        y_predicted.append(output)

    return np.concatenate(y_predicted, axis=1)


def train_RNN_seq2seq(dir, dataset, region, dimension):

    if region == 'all':
        df_master = pd.read_csv(dir / f'{dataset}.csv', parse_dates=[0],
                                index_col=0, date_parser=parser)
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
            core_RNN_seq2seq(df_master)
    else:
        df_master = df_master.filter(['data', dimension])
        core_RNN_seq2seq(df_master)


def core_RNN_seq2seq(df_master):
    """
        IMPORT DATA
    """
    scaler = get_scaler(df_master.values)
    df_master_normalized = scaler.transform(df_master.values).reshape((len(df_master)))

    train_percentage = 0.68
    size = int(len(df_master_normalized) * train_percentage)
    train_dataset, test_dataset = df_master_normalized[0:size], df_master_normalized[size:len(df_master_normalized)]

    """
        DECODER
    """
    keras.backend.clear_session()

    num_layers = 35
    layers = [num_layers, num_layers]  # Number of hidden neurons in each layer of the encoder and decoder

    # The dimensionality of the input at each time step. In this case a 1D signal.
    num_input_features = 1
    # The dimensionality of the output at each time step. In this case a 1D signal.
    num_output_features = 1

    # There is no reason for the input sequence to be of same dimension as the output sequence. For instance,
    # using 3 input signals: consumer confidence, inflation and house prices to predict the future house prices.
    loss = "mse"  # Other loss functions are possible, see Keras documentation.

    # Regularisation isn't really needed for this application
    lambda_regulariser = 0.000001  # Will not be used if regulariser is None
    regulariser = None  # Possible regulariser: keras.regularizers.l2(lambda_regulariser)

    batch_size = 10
    steps_per_epoch = 20  # batch_size * steps_per_epoch = total number of training examples
    epochs = 50

    # Learning rate decay
    learning_rate = 0.0001
    decay = learning_rate / epochs

    # Other possible optimiser "sgd" (Stochastic Gradient Descent)
    optimiser = keras.optimizers.Adam(lr=learning_rate,
                                      decay=decay)

    input_sequence_length = 4  # Length of the sequence used by the encoder
    target_sequence_length = 4  # Length of the sequence predicted by the decoder
    num_steps_to_predict = 5  # Length to use when testing the models

    RNN_seq2seq = 'RNN_seq2seq'
    model_name = _make_hparam_string_light(RNN_seq2seq, input_sequence_length, target_sequence_length,
                                           learning_rate, batch_size, epochs, decay)
    MODEL_NAME = f'{model_name}'

    """
        DECODER
    """
    # Define an input sequence.
    encoder_inputs = keras.layers.Input(shape=(None, num_input_features))

    # Create a list of RNN Cells, these are then concatenated into a single layer
    # with the RNN layer.
    encoder_cells = []
    for hidden_neurons in layers:
        encoder_cells.append(keras.layers.GRUCell(hidden_neurons,
                                                  kernel_regularizer=regulariser,
                                                  recurrent_regularizer=regulariser,
                                                  bias_regularizer=regulariser))

    encoder = keras.layers.RNN(encoder_cells, return_state=True)

    encoder_outputs_and_states = encoder(encoder_inputs)

    # Discard encoder outputs and only keep the states.
    # The outputs are of no interest to us, the encoder's
    # job is to create a state describing the input sequence.
    encoder_states = encoder_outputs_and_states[1:]

    """
        ENCODER
    """
    # The decoder input will be set to zero (see random_sine function of the utils module).
    # Do not worry about the input size being 1, I will explain that in the next cell.
    decoder_inputs = keras.layers.Input(shape=(None, 1))

    decoder_cells = []
    for hidden_neurons in layers:
        decoder_cells.append(keras.layers.GRUCell(hidden_neurons,
                                                  kernel_regularizer=regulariser,
                                                  recurrent_regularizer=regulariser,
                                                  bias_regularizer=regulariser))

    decoder = keras.layers.RNN(decoder_cells, return_sequences=True, return_state=True)

    # Set the initial state of the decoder to be the ouput state of the encoder.
    # This is the fundamental part of the encoder-decoder.
    decoder_outputs_and_states = decoder(decoder_inputs, initial_state=encoder_states)

    # Only select the output of the decoder (not the states)
    decoder_outputs = decoder_outputs_and_states[0]

    # Apply a dense layer with linear activation to set output to correct dimension
    # and scale (tanh is default activation for GRU in Keras, our output sine function can be larger then 1)
    decoder_dense = keras.layers.Dense(num_output_features,
                                       activation='linear',
                                       kernel_regularizer=regulariser,
                                       bias_regularizer=regulariser)

    decoder_outputs = decoder_dense(decoder_outputs)

    """
        CREATE MODEL
    """
    # Create a models using the functional API provided by Keras.
    # The functional API is great, it gives an amazing amount of freedom in architecture of your NN.
    # A read worth your time: https://keras.io/getting-started/functional-api-guide/
    # models = keras.RNN_seq2seq.Model(inputs=[encoder_inputs, decoder_inputs], outputs=decoder_outputs)
    model = keras.models.Model(inputs=[encoder_inputs, decoder_inputs], outputs=decoder_outputs)
    model.compile(optimizer=optimiser, loss=loss)

    """
        FIT MODEL
    """
    train_data_generator = batch_data_generator(dataset=train_dataset,
                                                batch_size=batch_size,
                                                steps_per_epoch=steps_per_epoch,
                                                input_sequence_length=input_sequence_length,
                                                target_sequence_length=target_sequence_length,
                                                seed=1969,
                                                train=True)

    model_checkpoint = str(MODELS_FOLDER / RNN_seq2seq / (MODEL_NAME + '.{epoch:02d}-{loss:.2f}.hdf5'))
    callbacks = [
        TensorBoard(log_dir=str(LOGS_FOLDER / MODEL_NAME)),
        ModelCheckpoint(
            model_checkpoint,
            monitor='loss',
            verbose=1,
            save_best_only=True,
            mode='auto')
    ]

    model.fit_generator(train_data_generator, steps_per_epoch=steps_per_epoch,
                        epochs=epochs, callbacks=callbacks)

    """
        TEST MODEL
    """
    test_data_generator = batch_data_generator(dataset=test_dataset,
                                               batch_size=batch_size,
                                               steps_per_epoch=steps_per_epoch,
                                               input_sequence_length=input_sequence_length,
                                               target_sequence_length=target_sequence_length,
                                               seed=2000,
                                               train=False)

    (x_encoder_test, x_decoder_test), y_test = next(test_data_generator)  # x_decoder_test is composed of zeros.

    y_test_predicted = model.predict([x_encoder_test, x_decoder_test])

    """
        CREATE PREDICTION MODEL
    """
    encoder_predict_model = keras.models.Model(encoder_inputs,
                                               encoder_states)

    decoder_states_inputs = []

    # Read layers backwards to fit the format of initial_state For some reason, the states of the models are order
    # backwards (state of the first layer at the end of the list) If instead of a GRU you were using an LSTM Cell,
    # you would have to append two Input tensors since the LSTM has 2 states.
    for hidden_neurons in layers[::-1]:
        # One state for GRU
        decoder_states_inputs.append(keras.layers.Input(shape=(hidden_neurons,)))

    decoder_outputs_and_states = decoder(
        decoder_inputs, initial_state=decoder_states_inputs)

    decoder_outputs = decoder_outputs_and_states[0]
    decoder_states = decoder_outputs_and_states[1:]

    decoder_outputs = decoder_dense(decoder_outputs)

    decoder_predict_model = keras.models.Model(
        [decoder_inputs] + decoder_states_inputs,
        [decoder_outputs] + decoder_states)

    """
        PREDICT
    """
    test_data_generator = batch_data_generator(dataset=test_dataset,
                                               batch_size=batch_size,
                                               steps_per_epoch=steps_per_epoch,
                                               input_sequence_length=input_sequence_length,
                                               target_sequence_length=target_sequence_length,
                                               seed=2000,
                                               train=False)

    (x_test, _), y_test = next(test_data_generator)

    y_test_predicted = predict(x_test, encoder_predict_model, decoder_predict_model, num_steps_to_predict, batch_size)

    # Select random examples to plot
    indices = np.random.choice(range(x_test.shape[0]), replace=False, size=3)

    for index in indices:
        plot_prediction(scaler.inverse_transform(x_test[index, :, :]),
                        scaler.inverse_transform(y_test[index, :, :]),
                        scaler.inverse_transform(y_test_predicted[index, :, :]),
                        scaler.inverse_transform(df_master_normalized.reshape((len(df_master_normalized), 1))), size)

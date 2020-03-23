# COVID-19 pandemic predictor

This repository contains AI algos models for COVID-19 pandemic precition. 

The aim of the project is to provide an evolutable AI stack for COVID-19 pandemic precition and show the first results despite the limited public data available.

The models used are ARIMA, RNN seq2seq, Prophet forecast and MLPRegressor. I use Keras and Tensorflow (tf-nightly-build).

## Setup

This project requires Python3.6+.

1. Clone the repository

2. Install dependencies from `requirements.txt`.

3. In case you want to run a new training and prediction rounds, you'll need additional steps:

  - Run `download_datasets.sh`, that will download the updated dataset (from the DPC public github account) and put in `data/raw`:
  ```
  bash download_datasets.sh
  ```

  - The data files have the following structure:
  ```
  ├── data
  │   └── raw
  │       └── italy
  │         └── italydpc-covid19-ita-andamento-nazionale.csv
  │          └── dpc-covid19-ita-regioni.csv
  │       └── world
  │          └── covid_19_data.csv
  │   └── processed
  │       └── italy
  │       └── world
  │          └── covid_19_data_Hubei.csv
  │   └── training
  │       └── italy
  │       └── world
  ```

## Code organization

The code is organized following the [Cookiecutter Data Science](https://drivendata.github.io/cookiecutter-data-science/) project structure:

```
├── data
│   ├── processed/                <- Processed datasets, created with make_dataset.py
│   └── raw/                      <- The original data dump, created with download_datasets.sh
|
├── download_datasets.sh          <- Downloads the dataset
|
├── LICENSE                       <- License files
|
├── models/                       <- Folder with model files, created after running train.py
|
├── notebooks/                    <- Jupyter notebooks, with explorations and results
|
├── README.md                     <- Overall instructions about the project
|
├── requirements.txt              <- The requirements file for reproducing the environment
|
├── results/                      <- Folder with results JSONs, created after running predict.py
|
├── src
│   ├── data
│   │   └── make_dataset.py       <- Builds the IMDB-WIKI dataset for training
|   |
│   ├── models
│   │   ├── compare.py            <- Make some data comparison (Italy vs Hubei)
│   │   ├── explore.py            <- Make some data exploration (e.g. plot and dimension time series autocorrelation)
│   │   ├── prophet.py            <- Runs predictions with Trophet
│   │   ├── regression.py         <- Runs predictions with MLPRegressor
│   │   ├── stannard.py           <- Stannard nonlinear equations fitting and forecasting
│   │   ├── train.py              <- Trains the model Arima or RNNseq2seq or Trophet or MLPRegressor
│   │   ├── train_arima.py        <- Trains the model Arima
│   │   ├── train_RNNseq2seq.py   <- Trains the model RNNseq2seq
|   |
│   └── utils.py                  <- Auxiliary methods used in the whole project
│   └── settings.py                  <- Directory management in the whole project
|
└── tensorboard_logs/             <- Folder with TensorBoard logs, created after running train.py
```

4. Run explore.py to make some data exploration (e.g. plot and dimension time series autocorrelation):

  - Run `explore.py --country --region --compare`, to explore specific country/region dataset or to compare country regions:
  ```
  python explore.py --country --region --compare
  ```

5. Run compare.py to make some data comparison (e.g. plot Italy vs Hubei):

  - Run `explore.py --country --region --compare`, to compare country regions:
  ```
  python explore.py --country --region --compare
  ```

6. Run train.py to train the model (Arima or RNN seq2seq or Prophet or MLPRegressor):

  - Run `train.py --country --region --dimension --model --lag_observations --degree_differencing --moving_average`, to train specific ARIMA model on country/region dimension:
  ```
  python train.py --country --region --dimension --model --lag_observations --degree_differencing --moving_average

  ```

  - Run `train.py --country --region --dimension --model --initial_lrate --epochs --batch_size`, to train specific RNN seq2seq model on country/region dimension:
  ```
  python train.py --country --region --dimension --model --initial_lrate --epochs --batch_size

  ```

  - Run `train.py --country --region --dimension --model --changepoint_prior_scale --periods`, to train specific Prophet model on country/region dimension:
  ```
  python train.py --country --region --dimension --model --changepoint_prior_scale --periods

  ```

  - Run `stannard.py`, to compute Italy forecast based on Stannard model:
  ```
  python stannard.py

  ```


## Authors
This project was developed by [Manuel Cugliari](https://github.com/manuelcugliari), Co-Owner/Chief research officer @ Justbit

![Manuel Cugliari](https://mc-custom-images.s3-eu-west-1.amazonaws.com/2020-03-16+13.07.23.jpg)
# Bitcoin Price Forecasting with LSTM

A TensorFlow/Keras-based time-series forecasting project for predicting Bitcoin (`BTC-USD`) weekly closing prices using a Long Short-Term Memory (LSTM) neural network.

The project downloads historical weekly Bitcoin market data from Yahoo Finance, applies point-relative normalization, constructs sequential inputs, trains an LSTM model, and evaluates the resulting predictions using RMSE and MAPE.

## Overview

This project explores univariate Bitcoin price forecasting using historical weekly closing prices.

The main workflow is:

```text
Yahoo Finance
      ↓
Weekly BTC-USD Data
      ↓
Data Preparation
      ↓
Point Relative Normalization
      ↓
Train / Test Split
      ↓
Sequence Construction
      ↓
TensorFlow / Keras LSTM
      ↓
Multi-step Prediction
      ↓
De-normalization
      ↓
RMSE / MAPE Evaluation
```

The primary implementation uses **TensorFlow/Keras**. An experimental PyTorch implementation is also present in the data-preparation notebook, but that section is incomplete and is not part of the main working pipeline.

## Features

- Downloads historical `BTC-USD` data from Yahoo Finance.
- Works with weekly Bitcoin market data.
- Extracts and preprocesses Bitcoin closing prices.
- Uses point-relative normalization for the main forecasting pipeline.
- Performs a chronological 80/20 train/test split.
- Groups observations into sequences of four values.
- Uses an LSTM-based neural network implemented with TensorFlow/Keras.
- Produces four-value predictions from the trained model.
- Converts normalized predictions back to price values.
- Evaluates predictions using:
  - RMSE
  - MAPE
- Generates plots for:
  - Training loss
  - Normalized predictions
  - De-normalized predictions

## Project Structure

```text
.
├── 0.Get_Data.ipynb
├── 1.Data_Preparation.ipynb
├── 2.Model_Architecture.ipynb
├── 3.Train_Model.ipynb
├── 4. Evaluate Model.ipynb
│
├── data/
│   ├── BTC_modified.csv
│   ├── BTC_wk.csv
│   ├── Test_BTC.csv
│   └── Train_BTC.csv
│
├── scripts/
│   ├── bitcoin_model.py
│   ├── normalizations.py
│   └── utilities.py
│
└── .gitignore
```

### Notebooks

| File | Description |
|---|---|
| `0.Get_Data.ipynb` | Downloads weekly `BTC-USD` data from Yahoo Finance. |
| `1.Data_Preparation.ipynb` | Prepares the Bitcoin dataset and performs normalization. It also contains an experimental PyTorch section. |
| `2.Model_Architecture.ipynb` | Defines and saves the TensorFlow/Keras LSTM architecture. |
| `3.Train_Model.ipynb` | Loads the model, trains it for 100 epochs, generates predictions, and visualizes them. |
| `4. Evaluate Model.ipynb` | Evaluates predictions, performs de-normalization, and calculates RMSE and MAPE. |

### Python Scripts

#### `scripts/normalizations.py`

Contains normalization utilities including:

- `z_score`
- `point_relative_normalization`
- `maximum_and_minimum_normalization`

#### `scripts/utilities.py`

Contains utilities for:

- normalization
- grouping observations into LSTM inputs
- splitting grouped data into model inputs and targets
- RMSE calculation
- MAPE calculation

#### `scripts/bitcoin_model.py`

Provides a `BitcoinModel` class for loading, saving, grouping, de-normalizing, and predicting with a Keras model.

The `train()` and `evaluate()` methods in this class are not implemented as complete training/evaluation workflows.

## Technologies & Libraries

### Programming Language

- Python 3

### Deep Learning

- TensorFlow
- Keras
- LSTM

### Data Processing

- pandas
- NumPy

### Data Acquisition

- yfinance

### Visualization

- Matplotlib
- Plotly

### Machine Learning Utilities

- scikit-learn

### Experimental Implementation

- PyTorch

The PyTorch code is present in `1.Data_Preparation.ipynb` as an experimental path but is not the completed model used by the main project workflow.

## Dataset

### Source

The raw Bitcoin data is downloaded from Yahoo Finance using:

```python
yf.download("BTC-USD", period="max", interval="1wk")
```

### Raw Dataset

The repository contains `data/BTC_wk.csv`, which contains **543 weekly observations** with the following columns:

```text
Date
Close
High
Low
Open
Volume
```

The available data runs from September 2014 to February 2025 in the included dataset.

### Processed Dataset

The main preprocessing pipeline removes the first three observations and works with **540 observations**.

The processed dataset contains:

```text
date
iso_month
close
close_point_relative_normalization
```

The main model uses the Bitcoin `close` series and its point-relative normalized representation.

## Methodology

The primary TensorFlow/Keras pipeline consists of the following steps.

### 1. Data Acquisition

Historical weekly Bitcoin data is retrieved from Yahoo Finance for the `BTC-USD` ticker.

### 2. Data Preparation

The downloaded data is cleaned and transformed into a weekly closing-price series.

### 3. Normalization

Point-relative normalization is applied within each `iso_month`.

### 4. Temporal Split

The processed data is divided chronologically into training and testing portions.

### 5. Sequence Construction

The normalized series is grouped into sequences containing four observations.

For the training pipeline, the 108 groups generated from the training data are used to construct:

```text
X_train: (1, 107, 4)
Y:       (1, 4)
```

The model therefore receives 107 groups of four normalized observations and predicts the next group of four values.

### 6. Model Training

The TensorFlow/Keras model is trained for 100 epochs using Mean Squared Error as the loss function.

### 7. Prediction

The trained model generates four normalized values.

### 8. De-normalization

Predictions are converted back to price values using the latest reference price.

### 9. Evaluation

The evaluation notebook produces normalized and de-normalized predictions and calculates RMSE and MAPE.

## Model Architecture

The main model is a small sequential TensorFlow/Keras network.

```text
Input
  │
  ▼
LSTM
  ├── 4 units
  ├── ReLU activation
  ├── dropout = 0.2
  └── return_sequences = False
  │
  ▼
Dense
  ├── 4 units
  └── ReLU activation
  │
  ▼
4 predicted values
```

The input shape is:

```text
(107, 4)
```

The model contains:

```text
Total parameters:       164
Trainable parameters:   164
Non-trainable:            0
```

### Optimizer

```text
Adam
```

### Learning Rate

```text
0.001
```

### Loss Function

```text
Mean Squared Error
```

### Dropout

```text
0.2
```

## Training

The main training procedure is implemented in:

```text
3.Train_Model.ipynb
```

The notebook loads the model generated by `2.Model_Architecture.ipynb` and compiles it using Adam.

The training configuration used in the notebook is:

| Parameter | Value |
|---|---:|
| Epochs | 100 |
| Optimizer | Adam |
| Learning rate | 0.001 |
| Loss | Mean Squared Error |
| LSTM units | 4 |
| Dense units | 4 |
| Dropout | 0.2 |
| Input shape | `(107, 4)` |

The training notebook also plots the training loss across epochs.

## Evaluation

The evaluation workflow is implemented in:

```text
4. Evaluate Model.ipynb
```

The following metrics are calculated.

### Normalized RMSE

The notebook reports:

```text
0.11
```

### De-normalized RMSE

The notebook reports:

```text
$4,789.4 USD
```

### De-normalized MAPE

The notebook reports:

```text
6.9%
```

These values are the results recorded in the executed evaluation notebook.

They should not be interpreted as classification accuracy. In particular, the reported MAPE is not equivalent to an "accuracy percentage."

## Results

The executed evaluation notebook reports:

| Metric | Result |
|---|---:|
| Normalized RMSE | 0.11 |
| De-normalized RMSE | $4,789.4 USD |
| De-normalized MAPE | 6.9% |

The project also generates visualizations for:

- Training loss
- Normalized predictions versus observed values
- De-normalized predicted versus observed Bitcoin prices
- Per-window MSE during evaluation

The training notebook saves a prediction visualization as:

```text
BTC_USD-d.png
```

The evaluation notebook saves another visualization as:

```text
BTC_USD-w.png
```

These image files are generated by notebook execution and are not included in the current repository snapshot.

## Installation

No dependency specification file such as `requirements.txt`, `pyproject.toml`, or `environment.yml` is included in the repository.

The notebooks use Python 3 and import the following main packages:

```text
pandas
numpy
matplotlib
plotly
yfinance
scikit-learn
tensorflow
keras
torch
```

The PyTorch dependency is only relevant to the experimental section of `1.Data_Preparation.ipynb`.

Because the repository does not contain a pinned dependency file, package versions are not specified by the project itself.

## Usage

The notebooks are intended to be executed in sequence.

### 1. Download the dataset

Open:

```text
0.Get_Data.ipynb
```

This downloads weekly `BTC-USD` data from Yahoo Finance and writes:

```text
data/BTC_wk.csv
```

### 2. Prepare the data

Run:

```text
1.Data_Preparation.ipynb
```

This prepares the Bitcoin data and applies the normalization used by the main model.

The repository already contains the resulting CSV files:

```text
data/BTC_modified.csv
data/Train_BTC.csv
data/Test_BTC.csv
```

### 3. Build the model

Run:

```text
2.Model_Architecture.ipynb
```

The notebook creates the TensorFlow/Keras LSTM model and saves:

```text
real_bitcoin_lstm_v0_me.keras
```

### 4. Train the model

Run:

```text
3.Train_Model.ipynb
```

The notebook trains the model for 100 epochs and saves:

```text
real_bitcoin_lstm_v0_trained_me.keras
```

It also generates prediction visualizations.

### 5. Evaluate the model

Run:

```text
4. Evaluate Model.ipynb
```

This evaluates the predictions, performs de-normalization, creates comparison plots, and reports RMSE and MAPE.

> **Note:** The model `.keras` files are generated during notebook execution and are not included in the current repository snapshot. Therefore, `2.Model_Architecture.ipynb` should be run before `3.Train_Model.ipynb`.

## Example

The core data acquisition step is:

```python
import yfinance as yf

data = yf.download(
    "BTC-USD",
    period="max",
    interval="1wk"
)
```

The main normalization used by the project is:

```python
def point_relative_normalization(series):
    return (series / series.values[0]) - 1
```

The model architecture is created with:

```python
model = Sequential()

model.add(
    Input(shape=(107, 4))
)

model.add(
    LSTM(
        units=4,
        activation="relu",
        dropout=0.2,
        return_sequences=False,
        stateful=False
    )
)

model.add(
    Dense(
        units=4,
        activation="relu"
    )
)

model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss="mean_squared_error"
)
```

## Visualization

The notebooks generate several visual outputs.

### Training Loss

`3.Train_Model.ipynb` plots the training loss recorded during the 100 training epochs.

### Normalized Predictions

`4. Evaluate Model.ipynb` compares the normalized predicted values with the observed normalized series.

### De-normalized Predictions

The evaluation notebook converts the normalized predictions back to Bitcoin price values and compares them with observed closing prices.

The generated output filenames are:

```text
BTC_USD-d.png
BTC_USD-w.png
```

## Limitations

The current repository should be considered an experimental/educational forecasting project rather than a production forecasting system.

Several limitations are visible directly in the implementation:

- The main model uses only the Bitcoin closing-price series for forecasting.
- The training input is constructed as a single sample with shape `(1, 107, 4)`.
- The training notebook labels the target as `Y_validation`, but it is used directly as the target passed to `model.fit()`, rather than as a separately held-out validation dataset.
- No separate validation dataset is implemented in the main TensorFlow training workflow.
- No baseline forecasting model is included for comparison.
- The repository does not include a requirements file or pinned package versions.
- The `BitcoinModel.train()` and `BitcoinModel.evaluate()` methods in `scripts/bitcoin_model.py` are incomplete.
- The PyTorch experiment in `1.Data_Preparation.ipynb` is incomplete. In particular, its dataset implementation references an undefined `ids` variable, and its training loop references `val_loader` without defining it.
- The grouping and monthly normalization logic should be interpreted in the context of the existing implementation rather than as a generalized forecasting framework.

The reported RMSE and MAPE values therefore describe the outputs of the implemented notebook workflow and should not be treated as evidence of production-level forecasting performance.

## Future Improvements

The following are potential extensions rather than features implemented in the current version:

- Construct multiple sliding training windows instead of training from a single input sample.
- Add a dedicated validation split.
- Compare the LSTM against simple forecasting baselines.
- Evaluate additional time-series metrics such as MAE.
- Experiment with different sequence lengths.
- Investigate alternative normalization strategies.
- Add additional market features such as OHLCV-derived indicators.
- Complete and separate the experimental PyTorch implementation.
- Refactor the notebook workflow into reusable Python modules.
- Add a pinned dependency file for reproducible environments.
- Add automated evaluation and experiment tracking.

## Author

**Mohammad Sheydaei**

Email: `sheydaeemohammad.business@gmail.com`

# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.16.2
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# # Library import

# +
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import pickle

from sklearn.preprocessing import MinMaxScaler

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
# -

# # Read the data

df = pd.read_csv('../data/household_power_consumption.txt', delimiter=';', low_memory=False)
df.head()

# create a new column for date and time
df['DateTime'] = df['Date'].astype(str) + '-' + df['Time'].astype(str)
df['DateTime'] = pd.to_datetime(df['DateTime'], format='%d/%m/%Y-%H:%M:%S')
# set the DateTime as the index and drop the Date and Time columns
df.set_index('DateTime', inplace=True)
df.drop(['Date', 'Time'], axis=1, inplace=True)
df.head()

# # Data exploration

print(f'Shape of the data: {df.shape}')

df.info()

# check for '?' values
df.eq('?').any()

# replace the '?' with NaN
df = df.replace('?', np.nan)

# check for '?' values again
df.eq('?').any()

# check for NaN values
df.isnull().any()

# check for NaN values in each column
df.isnull().sum()

# check for the percentage of NaN values in each column
df.isnull().sum() / df.shape[0] * 100

# convert the data to float, all columns have float values, but their type is object, which does not allow to perform some operations
df = df.astype(float)

# set missing values to 1 for the Global_active_power column to visualize the missing data
missing_data = df[df['Global_active_power'].isnull()]
missing_data.loc[:, 'Global_active_power'] = 1
missing_data.head()

plt.figure(figsize=(20, 10))
plt.plot(df.index, df['Global_active_power'], color='blue', label='Global Active Power')
plt.plot(missing_data.index, missing_data['Global_active_power'], 'ro', label='Missing Data', markersize=2)
plt.title('Global Active Power')
plt.xlabel('Time')
plt.ylabel('Global Active Power')
plt.legend()
plt.savefig('../images/global_active_power_missing_data.png')
plt.show()

# perform forward fill to fill the NaN values
df = df.ffill()

# check for NaN values in each column
np.sum(df.isnull().values, axis=0)

df.isnull().any()

df.info()

# resample the data to hourly
df = df.resample('h').mean()
print(f'Shape of the final data: {df.shape}')
df.head()

# save df in pickle format
with open('../data/household_power_consumption.pkl', 'wb') as f:
    pickle.dump(df, f)

# # Train test split

# load df from pickle file
with open("../data/household_power_consumption.pkl", "rb") as f:
    df = pickle.load(f)

# +
test_split = round(len(df) * 0.20)

# split the data into training and testing data
future_prediction_split = 24
df_for_training = df[:-test_split - future_prediction_split]
df_for_testing = df[-test_split - future_prediction_split:-future_prediction_split]
df_24_hours_future = df[-future_prediction_split:]
print(f'Shape of the training data: {df_for_training.shape}')
print(f'Shape of the testing data: {df_for_testing.shape}')
print(f'Shape of the future prediction data: {df_24_hours_future.shape}')
# -

df_for_training.tail()

df_for_testing.tail()

df_24_hours_future.tail()

plt.figure(figsize=(20, 7))
plt.plot(df_for_training.index, df_for_training['Global_active_power'], color='blue', label='Training Data')
plt.plot(df_for_testing.index, df_for_testing['Global_active_power'], color='red', label='Testing Data')
plt.title('Global Active Power')
plt.xlabel('Time')
plt.ylabel('Global Active Power')
plt.savefig('../images/global_active_power.png')
plt.legend()
plt.show()

# # Scale the data

# +
# initialize the MinMaxScaler
scaler = MinMaxScaler(feature_range=(0, 1))

# scale the training and testing data
df_for_training_scaled = scaler.fit_transform(df_for_training)
df_for_testing_scaled = scaler.transform(df_for_testing)
print(f'Shape of the scaled training data: {df_for_training_scaled.shape}')
print(f'Shape of the scaled testing data: {df_for_testing_scaled.shape}')

# +
# save the scaler
import joblib

joblib.dump(scaler, '../models/scaler.pkl')


# -

# # Create the dataset

# create the X and y data by creating sequences of data
def createXY(dataset, n_past, df_indices):
    X, y = [], []
    indices = []
    for i in range(n_past, len(dataset)):
        X.append(dataset[i - n_past:i, 0:dataset.shape[1]])
        y.append(dataset[i, 0])
        indices.append(df_indices[i])

    return np.array(X), np.array(y), np.array(indices)


# create the X and y data for training and testing
X_train, y_train, train_indices = createXY(df_for_training_scaled, 24, df_for_training.index)
X_test, y_test, test_indices = createXY(df_for_testing_scaled, 24, df_for_testing.index)

print(f'Shape of the training data: {X_train.shape}, {y_train.shape}, {train_indices.shape}')
print(f'Shape of the testing data: {X_test.shape}, {y_test.shape}, {test_indices.shape}')

# save train, test and indices in pickle
with open('../data/global_active_power_train_test_indices.pkl', 'wb') as f:
    pickle.dump([X_train, y_train, train_indices, X_test, y_test, test_indices], f)

# # LSTM model architecture

# load train, test and indices from pickle
with open("../data/global_active_power_train_test_indices.pkl", "rb") as f:
    X_train, y_train, train_indices, X_test, y_test, test_indices = pickle.load(f)

"""
The LSTM model architecture is as follows:
- Input layer
- LSTM layer with 50 units and return sequences as True
- LSTM layer with 50 units
- Dropout layer with 0.2
- Dense layer with 1 unit
"""
model = Sequential()
model.add(Input(shape=(X_train.shape[1], X_train.shape[2])))
model.add(LSTM(50, return_sequences=True))
model.add(LSTM(50))
model.add(Dropout(0.2))
model.add(Dense(1))

# compile the model with adam optimizer and mean squared error loss
model.compile(optimizer='adam', loss='mean_squared_error')

model.summary()

# fit the model
history = model.fit(X_train, y_train, epochs=30, batch_size=64, verbose=1)

# save the model
model.save('../models/global_active_power_first_model.keras')

# +
# load the model
from tensorflow.keras.models import load_model

model = load_model('../models/global_active_power_first_model.keras')
# -

# predict the values
prediction = model.predict(X_test)
print(f'Shape of the prediction: {prediction.shape}')

# +
from household_power_consumption.utils import inverse_transform_data

# inverse transform the scaled data, repeat the prediction to match the shape of the original data
prediction = inverse_transform_data(prediction, scaler, df_for_testing.shape)
# -

# inverse transform the scaled data, repeat the original data to match the shape of the prediction
original = inverse_transform_data(y_test, scaler, df_for_testing.shape)

# +
# calculate the metrics
from household_power_consumption.utils import calculate_metrics

mse = calculate_metrics(original, prediction)

# +
# plot the original and predicted data
from household_power_consumption.utils import plot_original_and_predicted_data

plot_original_and_predicted_data(original, prediction, test_indices, mse, 'Global Active Power Prediction')
# -

# ## Predicting future values

df_24_hours_past = df.iloc[-2 * future_prediction_split:-future_prediction_split, :]
print(f'Shape of the 24 hours past data: {df_24_hours_past.shape}')
df_24_hours_past.tail()

df_24_hours_future_gap = df_24_hours_future['Global_active_power'].values.copy()
df_24_hours_future_gap

df_24_hours_future.loc[:, 'Global_active_power'] = 0
df_24_hours_future.tail()

old_scaled_array = scaler.transform(df_24_hours_past)
new_scaled_array = scaler.transform(df_24_hours_future)

new_scaled_df = pd.DataFrame(new_scaled_array)
new_scaled_df.iloc[:, 0] = np.nan
full_df = pd.concat([pd.DataFrame(old_scaled_array), new_scaled_df]).reset_index().drop(["index"], axis=1)
print(f'Shape of the full data: {full_df.shape}')

full_df.tail()

# +
full_df_scaled_array = full_df.values
all_data = []
time_step = 24

for i in range(time_step, len(full_df_scaled_array)):
    data_x = []
    data_x.append(full_df_scaled_array[i - time_step:i, 0:full_df_scaled_array.shape[1]])
    data_x = np.array(data_x)
    prediction = model.predict(data_x)
    all_data.append(prediction)
    full_df.iloc[i, 0] = prediction
# -

new_array = np.array(all_data).reshape(-1, 1)
y_pred_future_24_hours = inverse_transform_data(new_array, scaler, df_for_testing.shape)

# +
from household_power_consumption.utils import plot_24_hours_prediction

plot_24_hours_prediction(
    df_24_hours_future,
    df_24_hours_future_gap,
    y_pred_future_24_hours,
    "Global Active Power Prediction 24 hours",
)
# -

# ## Hyperparameter tuning

# load train, test and indices from pickle
with open('../data/global_active_power_train_test_indices.pkl', 'rb') as f:
    X_train, y_train, train_indices, X_test, y_test, test_indices = pickle.load(f)

import optuna

# +
from household_power_consumption.utils import create_model


def objective(trial):
    units1 = trial.suggest_categorical('units1', [20, 50, 80, 100])
    units2 = trial.suggest_categorical('units2', [20, 50, 80, 100])
    dropout = trial.suggest_categorical('dropout', [0.2, 0.4, 0.6])
    batch_size = trial.suggest_categorical('batch_size', [32, 64, 128])

    model = create_model({'units1': units1, 'units2': units2, 'dropout': dropout}, X_train.shape)

    history = model.fit(X_train, y_train, epochs=30, batch_size=batch_size, verbose=0, validation_split=0.2)

    val_loss = np.min(history.history['val_loss'])

    return val_loss


# +
from tqdm import tqdm

n_trials = 40
pbar = tqdm(total=n_trials, desc='Optimization Progress')


def tqdm_callback(study, trial):
    pbar.update(1)


# -

study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=n_trials, n_jobs=5, callbacks=[tqdm_callback])

# +
print("Best trial:")
trial = study.best_trial

print("  Value: ", trial.value)

print("  Params: ")
for key, value in trial.params.items():
    print(f"    {key}: {value}")
# -

model_with_best_params = create_model(trial.params)
model_with_best_params.summary()

history = model_with_best_params.fit(X_train, y_train, epochs=30, batch_size=trial.params["batch_size"], verbose=1)

# save the model
model_with_best_params.save('../models/gap_model_best_params.keras')

# +
# load model
from tensorflow.keras.models import load_model

model_with_best_params = load_model("../models/gap_model_best_params.keras")
# -

# predict the values
prediction = model_with_best_params.predict(X_test)
print(f"Shape of the prediction: {prediction.shape}")

# +
# load scaler
import joblib

scaler = joblib.load('../models/scaler.pkl')
# -

prediction = inverse_transform_data(prediction, scaler, df_for_testing.shape)
original = inverse_transform_data(y_test, scaler, df_for_testing.shape)

mse = calculate_metrics(original, prediction)

plot_original_and_predicted_data(original, prediction, test_indices, mse, "Global Active Power Prediction tuned LSTM")

# ## Predicting future values - tuned LSTM model

df_24_hours_past = df.iloc[-2 * future_prediction_split : -future_prediction_split, :]
print(f"Shape of the 24 hours past data: {df_24_hours_past.shape}")
df_24_hours_past.tail()

df_24_hours_future_gap = df_24_hours_future["Global_active_power"].values.copy()
df_24_hours_future_gap

df_24_hours_future.loc[:, "Global_active_power"] = 0
df_24_hours_future.tail()

old_scaled_array = scaler.transform(df_24_hours_past)
new_scaled_array = scaler.transform(df_24_hours_future)

new_scaled_df = pd.DataFrame(new_scaled_array)
new_scaled_df.iloc[:, 0] = np.nan
full_df = (
    pd.concat([pd.DataFrame(old_scaled_array), new_scaled_df])
    .reset_index()
    .drop(["index"], axis=1)
)
print(f"Shape of the full data: {full_df.shape}")

# +
full_df_scaled_array = full_df.values
all_data = []
time_step = 24

for i in range(time_step, len(full_df_scaled_array)):
    data_x = []
    data_x.append(
        full_df_scaled_array[i - time_step : i, 0 : full_df_scaled_array.shape[1]]
    )
    data_x = np.array(data_x)
    prediction = model_with_best_params.predict(data_x)
    all_data.append(prediction)
    full_df.iloc[i, 0] = prediction
# -

new_array = np.array(all_data).reshape(-1, 1)
y_pred_future_24_hours = inverse_transform_data(new_array, scaler, df_for_testing.shape)

plot_24_hours_prediction(
    df_24_hours_future,
    df_24_hours_future_gap,
    y_pred_future_24_hours,
    "Global Active Power Prediction tuned LSTM 24 hours",
)

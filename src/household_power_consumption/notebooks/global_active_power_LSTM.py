#%% md
# # Library import
#%%
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from sklearn.preprocessing import MinMaxScaler

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
#%% md
# # Read the data
#%%
df = pd.read_csv('../data/household_power_consumption.txt', delimiter=';', low_memory=False)
df.head()
#%%
# create a new column for date and time
df['DateTime'] = df['Date'].astype(str) + '-' + df['Time'].astype(str)
df['DateTime'] = pd.to_datetime(df['DateTime'], format='%d/%m/%Y-%H:%M:%S')
# set the DateTime as the index and drop the Date and Time columns
df.set_index('DateTime', inplace=True)
df.drop(['Date', 'Time'], axis=1, inplace=True)
df.head()
#%% md
# # Data exploration
#%%
print(f'Shape of the data: {df.shape}')
#%%
df.info()
#%%
# check for '?' values
df.eq('?').any()
#%%
# replace the '?' with NaN
df = df.replace('?', np.nan)
#%%
# check for '?' values again
df.eq('?').any()
#%%
# check for NaN values
df.isnull().any()
#%%
# perform forward fill to fill the NaN values
df = df.ffill()
#%%
# check for NaN values in each column
np.sum(df.isnull().values, axis=0)
#%%
df.isnull().any()
#%%
# convert the data to float, all columns have float values, but their type is object, which does not allow to perform some operations
df = df.astype(float)
#%%
df.info()
#%%
# resample the data to hourly
df = df.resample('h').mean()
print(f'Shape of the final data: {df.shape}')
df.head()
#%%
# df = df.rolling(window=12).mean()
#%% md
# # Train test split
#%%
test_split = round(len(df) * 0.20)

# split the data into training and testing data
future_prediction_split = 24
df_for_training = df[:-test_split - future_prediction_split]
df_for_testing = df[-test_split - future_prediction_split:-future_prediction_split]
df_24_hours_future = df[-future_prediction_split:]
print(f'Shape of the training data: {df_for_training.shape}')
print(f'Shape of the testing data: {df_for_testing.shape}')
print(f'Shape of the future prediction data: {df_24_hours_future.shape}')
#%%
df_for_training.tail()
#%%
df_for_testing.tail()
#%%
df_24_hours_future.tail()
#%%
plt.figure(figsize=(20, 7))
plt.plot(df_for_training.index, df_for_training['Global_active_power'], color='blue', label='Training Data')
plt.plot(df_for_testing.index, df_for_testing['Global_active_power'], color='red', label='Testing Data')
plt.title('Global Active Power')
plt.xlabel('Time')
plt.ylabel('Global Active Power')
plt.savefig('../images/global_active_power.png')
plt.legend()
plt.show()
#%% md
# # Scale the data
#%%
# initialize the MinMaxScaler
scaler = MinMaxScaler(feature_range=(0, 1))

# scale the training and testing data
df_for_training_scaled = scaler.fit_transform(df_for_training)
df_for_testing_scaled = scaler.transform(df_for_testing)
print(f'Shape of the scaled training data: {df_for_training_scaled.shape}')
print(f'Shape of the scaled testing data: {df_for_testing_scaled.shape}')
#%% md
# # Create the dataset
#%%
# create the X and y data by creating sequences of data
def createXY(dataset, n_past, df_indices):
    X, y = [], []
    indices = []
    for i in range(n_past, len(dataset)):
        X.append(dataset[i - n_past:i, 0:dataset.shape[1]])
        y.append(dataset[i, 0])
        indices.append(df_indices[i])

    return np.array(X), np.array(y), np.array(indices)
#%%
# create the X and y data for training and testing
X_train, y_train, train_indices = createXY(df_for_training_scaled, 24, df_for_training.index)
X_test, y_test, test_indices = createXY(df_for_testing_scaled, 24, df_for_testing.index)
#%%
print(f'Shape of the training data: {X_train.shape}, {y_train.shape}, {train_indices.shape}')
print(f'Shape of the testing data: {X_test.shape}, {y_test.shape}, {test_indices.shape}')
#%% md
# # LSTM model architecture
#%%
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
#%%
# compile the model with adam optimizer and mean squared error loss
model.compile(optimizer='adam', loss='mean_squared_error')
#%%
model.summary()
#%%
# fit the model
history = model.fit(X_train, y_train, epochs=50, batch_size=64, verbose=1)
#%%
# predict the values
prediction = model.predict(X_test)
print(f'Shape of the prediction: {prediction.shape}')
#%%
# calculate the metrics
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error, \
    root_mean_squared_error

mse = mean_squared_error(y_test, prediction)
print(f'Mean Squared Error: {mse:.4f}')
print(f'Mean Absolute Error: {mean_absolute_error(y_test, prediction):.4f}')
print(f'Mean Absolute Percentage Error: {mean_absolute_percentage_error(y_test, prediction):.4f}')
print(f'Root Mean Squared Error: {root_mean_squared_error(y_test, prediction):.4f}')
#%%
# inverse transform the scaled data, repeat the prediction to match the shape of the original data
prediction_copies = np.repeat(prediction, df_for_testing.shape[1], axis=-1)
prediction = scaler.inverse_transform(np.reshape(prediction_copies, (len(prediction), df_for_testing.shape[1])))[:, 0]
print(f'Shape of the prediction: {prediction.shape}')
#%%
# inverse transform the scaled data, repeat the original data to match the shape of the prediction
original_copies = np.repeat(y_test, df_for_testing.shape[1], axis=-1)
original = scaler.inverse_transform(np.reshape(original_copies, (len(y_test), df_for_testing.shape[1])))[:, 0]
print(f'Shape of the original: {original.shape}')
#%%
# plot the original and predicted data
plt.figure(figsize=(22, 12))
plt.plot(test_indices, original, color='red', label='Real Global Active Power')
plt.plot(test_indices, prediction, color='blue', label='Predicted Global Active Power')
plt.text(0.04, 0.9, f'MSE: {mse:.3f}', horizontalalignment='center', verticalalignment='center',
         transform=plt.gca().transAxes, fontsize=12)
plt.title('Global Active Power Prediction')
plt.xlabel('Time')
plt.ylabel('Global Active Power')
plt.legend()
plt.savefig('../images/global_active_power_prediction.png')
plt.show()
#%% md
# ## Predicting future values
#%%
df_24_hours_past = df.iloc[-2 * future_prediction_split:-future_prediction_split, :]
print(f'Shape of the 24 hours past data: {df_24_hours_past.shape}')
df_24_hours_past.tail()
#%%
df_24_hours_future_gap = df_24_hours_future['Global_active_power'].values.copy()
df_24_hours_future_gap
#%%
df_24_hours_future.loc[:, 'Global_active_power'] = 0
df_24_hours_future.tail()
#%%
old_scaled_array = scaler.transform(df_24_hours_past)
new_scaled_array = scaler.transform(df_24_hours_future)
#%%
new_scaled_df = pd.DataFrame(new_scaled_array)
new_scaled_df.iloc[:, 0] = np.nan
full_df = pd.concat([pd.DataFrame(old_scaled_array), new_scaled_df]).reset_index().drop(["index"], axis=1)
print(f'Shape of the full data: {full_df.shape}')
#%%
full_df.tail()
#%%
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
#%%
new_array = np.array(all_data)
new_array = new_array.reshape(-1, 1)
prediction_copies_array = np.repeat(new_array, df_for_testing.shape[1], axis=-1)
y_pred_future_24_hours = scaler.inverse_transform(
    np.reshape(prediction_copies_array, (len(new_array), df_for_testing.shape[1])))[:, 0]
print(f'Shape of the future prediction: {y_pred_future_24_hours.shape}')
#%%
plt.figure(figsize=(20, 10))
plt.plot(df_24_hours_future.index, y_pred_future_24_hours, color='blue', label='Predicted Global Active Power')
plt.plot(df_24_hours_future.index, df_24_hours_future_gap, color='red', label='Real Global Active Power')
plt.title('Global Active Power Prediction for the next 24 hours')
plt.xlabel('Time')
plt.ylabel('Global Active Power')
plt.legend()
plt.savefig('../images/global_active_power_24_h_prediction.png')
plt.show()
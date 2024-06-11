# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.16.2
#   kernelspec:
#     display_name: .venv
#     language: python
#     name: python3
# ---

# ## Hyperparameter tunning of LSTM

import optuna
from household_power_consumption.utils import (
    create_model,
    calculate_metrics,
    plot_original_and_predicted_data,
    inverse_transform_data,
    plot_24_hours_prediction,
)
import pickle
import numpy as np
import pandas as pd

# load df from pickle file
with open("../data/household_power_consumption.pkl", "rb") as f:
    df = pickle.load(f)

# load train, test and indices from pickle
with open('../data/global_active_power_train_test_indices.pkl', 'rb') as f:
    X_train, y_train, train_indices, X_test, y_test, test_indices = pickle.load(f)


def objective(trial):
    units1 = trial.suggest_categorical("units1", [20, 50, 80, 100])
    units2 = trial.suggest_categorical("units2", [20, 50, 80, 100])
    units3 = trial.suggest_categorical("units3", [20, 50, 80, 100])
    dropout = trial.suggest_categorical("dropout", [0.2, 0.4, 0.6])
    batch_size = trial.suggest_categorical("batch_size", [32, 64, 128])

    model = create_model(
        {"units1": units1, "units2": units2, "units3": units3, "dropout": dropout},
        X_train.shape,
    )

    history = model.fit(
        X_train,
        y_train,
        epochs=30,
        batch_size=batch_size,
        verbose=0,
        validation_split=0.2,
    )

    val_loss = np.min(history.history["val_loss"])

    return val_loss


# +
from tqdm import tqdm

n_trials = 60
pbar = tqdm(total=n_trials, desc="Optimization Progress")


def tqdm_callback(study, trial):
    pbar.update(1)


# -

study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=n_trials, n_jobs=10, callbacks=[tqdm_callback])

# +
print("Best trial:")
trial = study.best_trial

print("  Value: ", trial.value)

print("  Params: ")
for key, value in trial.params.items():
    print(f"    {key}: {value}")
# -

model_with_best_params = create_model(trial.params, X_train.shape)
model_with_best_params.summary()

history = model_with_best_params.fit(
    X_train, y_train, epochs=30, batch_size=trial.params["batch_size"], verbose=1
)

# save the model
model_with_best_params.save("../models/gap_model_best_params.keras")

# +
from tensorflow.keras.models import load_model

model_with_best_params = load_model("../models/gap_model_best_params.keras")
# -

# predict the values
prediction = model_with_best_params.predict(X_test)
print(f"Shape of the prediction: {prediction.shape}")

# +
# load scaler
import joblib

scaler = joblib.load("../models/scaler.pkl")
# -

prediction = inverse_transform_data(prediction, scaler, X_test.shape[1:])
original = inverse_transform_data(y_test, scaler, X_test.shape[1:])

mse = calculate_metrics(original, prediction)

plot_original_and_predicted_data(
    original,
    prediction,
    test_indices,
    mse,
    "Global Active Power Prediction tuned complex LSTM",
)

# ## Predicting future values - tuned LSTM model

future_prediction_split = 24
df_24_hours_past = df.iloc[-2 * future_prediction_split : -future_prediction_split, :]
df_24_hours_future = df[-future_prediction_split:]
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
y_pred_future_24_hours = inverse_transform_data(
    new_array, scaler, X_test.shape[1:]
)

plot_24_hours_prediction(
    df_24_hours_future,
    df_24_hours_future_gap,
    y_pred_future_24_hours,
    "Global Active Power Prediction tuned complex LSTM 24 hours",
)

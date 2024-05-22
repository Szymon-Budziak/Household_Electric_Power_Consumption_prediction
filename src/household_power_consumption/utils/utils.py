import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
    root_mean_squared_error,
)
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from tensorflow.keras.models import Sequential

__all__ = [
    "inverse_transform_data",
    "calculate_metrics",
    "plot_original_and_predicted_data",
    'plot_24_hours_prediction',
    'create_model'
]


def inverse_transform_data(
        data: np.ndarray, scaler: MinMaxScaler, shape: tuple
) -> np.ndarray:
    """
    Inverse transform the data using the MinMaxScaler.

    Args:
        data (np.ndarray): Data to be transformed.
        scaler (MinMaxScaler): Scaler object to be used for inverse transformation.
        shape (tuple): Shape of the data.

    Returns:
        np.ndarray: Inverse transformed data.
    """
    data_copies = np.repeat(data, shape[1], axis=-1)
    transformed_data = scaler.inverse_transform(
        np.reshape(data_copies, (len(data), shape[1]))
    )[:, 0]
    print(f"Shape of the transformed data: {transformed_data.shape}")

    return transformed_data


def calculate_metrics(original: np.ndarray, prediction: np.ndarray) -> float:
    """
    Calculate the metrics for the model.

    Args:
        original (np.ndarray): Orginal frame of the data.
        prediction (np.ndarray): Predicted frame of the data.

    Returns:
        float: Mean Squared Error.
    """
    mse = mean_squared_error(original, prediction)
    print(f"Mean Squared Error: {mse:.4f}")
    print(f"Mean Absolute Error: {mean_absolute_error(original, prediction):.4f}")
    print(
        f"Mean Absolute Percentage Error: {mean_absolute_percentage_error(original, prediction):.4f}"
    )
    print(
        f"Root Mean Squared Error: {root_mean_squared_error(original, prediction):.4f}"
    )

    return mse


def plot_original_and_predicted_data(
        original: np.ndarray,
        prediction: np.ndarray,
        test_indices: np.ndarray,
        mse: float,
        title: str,
) -> None:
    """
    Plot the original and predicted data.

    Args:
        original (np.ndarray): Original data.
        prediction (np.ndarray): Predicted data.
        test_indices (np.ndarray): Test indices.
        mse (float): Calculated Mean Squared Error.
        title (str): Title of the plot.
    """
    plt.figure(figsize=(22, 12))
    plt.plot(test_indices, original, color="red", label="Real Global Active Power")
    plt.plot(
        test_indices, prediction, color="blue", label="Predicted Global Active Power"
    )
    plt.text(
        0.04,
        0.9,
        f"MSE: {mse:.3f}",
        horizontalalignment="center",
        verticalalignment="center",
        transform=plt.gca().transAxes,
        fontsize=12,
    )

    plt.title(f"{title}")
    plt.xlabel("Time")
    plt.ylabel("Global Active Power")

    plt.legend()
    plt.savefig(f'../images/{title.lower().replace(" ", "_")}.png')
    plt.show()
    return None


def plot_24_hours_prediction(df_24_hours_future: pd.DataFrame, df_24_hours_future_gap: np.ndarray,
                             y_pred_future_24_hours: np.ndarray, title: str) -> None:
    """
    Plot the 24 hours prediction.

    Args:
        df_24_hours_future (pd.DataFrame): Dataframe of the 24 hours future data.
        df_24_hours_future_gap (np.ndarray): Dataframe of the 24 hours future global active power.
        y_pred_future_24_hours (np.ndarray): Predicted 24 hours global active power.
        title (str): Title of the plot.
    """
    plt.figure(figsize=(20, 10))
    plt.plot(df_24_hours_future.index, y_pred_future_24_hours, color='blue', label='Predicted Global Active Power')
    plt.plot(df_24_hours_future.index, df_24_hours_future_gap, color='red', label='Real Global Active Power')
    plt.title(f'{title}')

    plt.xlabel('Time')
    plt.ylabel('Global Active Power')
    plt.legend()
    plt.savefig(f'../images/{title.lower().replace(' ', '_')}.png')
    plt.show()


def create_model(params: dict, X_train_shape: tuple) -> Sequential:
    """
    Create the LSTM model.

    Args:
        params (dict): Dictionary of the parameters.
        X_train_shape (tuple): Shape of the training data.

    Returns:
        Sequential: LSTM model.
    """
    model = Sequential()
    model.add(Input(shape=(X_train_shape[1], X_train_shape[2])))
    model.add(LSTM(params['units1'], return_sequences=True))
    model.add(LSTM(params['units2']))
    model.add(Dropout(params['dropout']))
    model.add(Dense(1))

    model.compile(loss='mean_squared_error', optimizer='adam')

    return model

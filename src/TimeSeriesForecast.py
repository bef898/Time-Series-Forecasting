import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout
from sklearn.metrics import mean_absolute_error, mean_squared_error
from math import sqrt

class TimeSeriesForecast:
    def __init__(self, data, train_ratio=0.8):
        self.data = data
        self.train_ratio = train_ratio
        self.train_data = None
        self.test_data = None
        self.model = None
        self.predictions = None
        self.split_data()
        
    def split_data(self):
        """Split data into training and testing sets."""
        split_index = int(len(self.data) * self.train_ratio)
        self.train_data = self.data[:split_index]
        self.test_data = self.data[split_index:]
        print(f"Data split into {len(self.train_data)} training samples and {len(self.test_data)} testing samples.")

    def train_arima(self, order=(1, 1, 1)):
        """Train an ARIMA model."""
        self.model = ARIMA(self.train_data, order=order)
        self.model = self.model.fit()
        print("ARIMA model trained successfully.")
    
    def forecast_arima(self):
        """Forecast using the trained ARIMA model."""
        if not self.model:
            raise Exception("Model not trained. Call train_arima() first.")
        steps = len(self.test_data)
        self.predictions = self.model.forecast(steps=steps)
        return self.predictions
    
    def evaluate_forecast(self):
        """Evaluate the forecast using common metrics."""
        if self.predictions is None:
            raise Exception("No predictions found. Call forecast_arima() first.")
        mae = mean_absolute_error(self.test_data, self.predictions)
        rmse = sqrt(mean_squared_error(self.test_data, self.predictions))
        mape = np.mean(np.abs((self.test_data - self.predictions) / self.test_data)) * 100
        print(f"Mean Absolute Error (MAE): {mae:.4f}")
        print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
        print(f"Mean Absolute Percentage Error (MAPE): {mape:.2f}%")
        return mae, rmse, mape

    def plot_forecast(self):
        """Plot the actual vs predicted values."""
        plt.figure(figsize=(10, 6))
        plt.plot(self.data.index, self.data, label='Actual')
        plt.plot(self.test_data.index, self.predictions, label='Predicted', color='orange')
        plt.xlabel('Date')
        plt.ylabel('Stock Price')
        plt.title('Actual vs Predicted Stock Prices')
        plt.legend()
        plt.show()
    def train_sarima(self, order=(1, 1, 1), seasonal_order=(0, 1, 1, 12)):
        """Train a SARIMA model."""
        self.model = SARIMAX(self.train_data, order=order, seasonal_order=seasonal_order)
        self.model = self.model.fit()
        print("SARIMA model trained successfully.")
    
    def forecast_sarima(self):
        """Forecast using the trained SARIMA model."""
        if not self.model:
            raise Exception("Model not trained. Call train_sarima() first.")
        steps = len(self.test_data)
        self.predictions = self.model.forecast(steps=steps)
        return self.predictions
    def preprocess_for_lstm(self, window_size=60):
        """Preprocess data for LSTM model."""
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(self.data.values.reshape(-1, 1))
        
        # Creating a windowed dataset
        X, y = [], []
        for i in range(window_size, len(scaled_data)):
            X.append(scaled_data[i-window_size:i, 0])
            y.append(scaled_data[i, 0])
        
        X, y = np.array(X), np.array(y)
        X = np.reshape(X, (X.shape[0], X.shape[1], 1))  # Reshape for LSTM input
        train_size = int(self.train_ratio * len(X))
        self.X_train, self.X_test = X[:train_size], X[train_size:]
        self.y_train, self.y_test = y[:train_size], y[train_size:]
        self.scaler = scaler
        print("Data preprocessed for LSTM model.")

    def train_lstm(self, epochs=50, batch_size=32):
        """Train an LSTM model."""
        self.model = Sequential()
        self.model.add(LSTM(units=50, return_sequences=True, input_shape=(self.X_train.shape[1], 1)))
        self.model.add(Dropout(0.2))
        self.model.add(LSTM(units=50, return_sequences=False))
        self.model.add(Dropout(0.2))
        self.model.add(Dense(units=1))

        self.model.compile(optimizer='adam', loss='mean_squared_error')
        self.model.fit(self.X_train, self.y_train, epochs=epochs, batch_size=batch_size)
        print("LSTM model trained successfully.")
    
    def forecast_lstm(self):
        """Forecast using the trained LSTM model."""
        if not self.model:
            raise Exception("Model not trained. Call train_lstm() first.")
        predictions = self.model.predict(self.X_test)
        predictions = self.scaler.inverse_transform(predictions)
        self.predictions = predictions.ravel()
        return self.predictions
    def evaluate_forecast(self):
    
        if self.predictions is None:
            raise Exception("No predictions found. Call a forecasting method first.")

        # Check for length mismatch and align predictions with test data
        min_length = min(len(self.test_data), len(self.predictions))
        actual = self.test_data[:min_length]  # Trim the test data
        predictions = self.predictions[:min_length]  # Trim the predictions

        # Ensure both are pandas Series for compatibility
        if isinstance(actual, pd.DataFrame):
            actual = actual.squeeze()  # Convert DataFrame to Series if it has one column
        if isinstance(predictions, pd.DataFrame):
            predictions = predictions.squeeze()  # Convert DataFrame to Series if it has one column

        # Convert to numpy arrays if needed
        actual = np.array(actual).flatten()
        predictions = np.array(predictions).flatten()

        # Calculate evaluation metrics
        mae = mean_absolute_error(actual, predictions)
        rmse = sqrt(mean_squared_error(actual, predictions))
        mape = np.mean(np.abs((actual - predictions) / actual)) * 100

        print(f"Mean Absolute Error (MAE): {mae}")
        print(f"Root Mean Squared Error (RMSE): {rmse}")
        print(f"Mean Absolute Percentage Error (MAPE): {mape}%")
        return mae, rmse, mape
    def plot_forecast(self):
   
        if self.predictions is None:
            raise Exception("No predictions found. Call a forecasting method first.")

        # Align lengths of test data and predictions
        min_length = min(len(self.test_data), len(self.predictions))
        test_data_aligned = self.test_data[:min_length]
        predictions_aligned = self.predictions[:min_length]

        # Plotting
        plt.figure(figsize=(10, 6))
        plt.plot(self.data.index, self.data, label='Actual')
        plt.plot(test_data_aligned.index, predictions_aligned, label='Predicted', color='orange')
        plt.xlabel('Date')
        plt.ylabel('Stock Price')
        plt.title('Forecast vs Actual')
        plt.legend()
        plt.show()
        



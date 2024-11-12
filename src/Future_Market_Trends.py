import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

class FutureMarketForecast:
    def __init__(self, trained_model, historical_data, test_data,input_shape = 60):
        """
        Initialize the forecast class.
        Parameters:
        - trained_model: The trained model (ARIMA, SARIMA, etc.)
        - historical_data: The historical stock data used for training
        - test_data: Test data used for evaluating the model
        """
        self.model = trained_model
        self.data = historical_data
        self.test_data = test_data
        self.input_shape = input_shape

    def forecast_future(self, periods=180):
        """
        Generate future forecasts using the trained LSTM model.
        Parameters:
        - periods: Number of periods (days) to forecast
        Returns:
        - forecast_df: A DataFrame containing the forecasted values
        """
        if self.model is None:
            raise Exception("No trained model found. Train a model first.")

        # Assuming `self.test_data` is a DataFrame or Series containing recent data for input
        # Prepare input for the LSTM model
        last_known_data = self.test_data.values[-self.input_shape:]  # Use the last known sequence length
        future_predictions = []

        for _ in range(periods):
            # Reshape input data to match the LSTM input shape: (samples, timesteps, features)
            input_data = last_known_data.reshape((1, self.input_shape, 1))  # Adjust shape as necessary
            next_pred = self.model.predict(input_data)[0, 0]  # Predict next value

            # Store prediction and update input for the next step
            future_predictions.append(next_pred)
            last_known_data = np.append(last_known_data[1:], next_pred)  # Update window with new prediction

        # Generate forecast index based on existing data
        forecast_index = pd.date_range(start=self.test_data.index[-1] + pd.Timedelta(days=1), periods=periods, freq='B')
        forecast_df = pd.DataFrame({'Forecast': future_predictions}, index=forecast_index)

        self.forecast_df = forecast_df
        return forecast_df

    def visualize_forecast(self):
        """Visualize the forecasted stock prices along with historical data."""
        if not hasattr(self, 'forecast_df'):
            raise Exception("No forecast data found. Call forecast_future() first.")

        plt.figure(figsize=(12, 8))

        # Plot historical data
        plt.plot(self.data.index, self.data, label='Historical Data', color='blue')

        # Plot forecasted data
        plt.plot(self.forecast_df.index, self.forecast_df['Forecast'], label='Forecasted Price', color='orange')
        plt.xlabel('Date')
        plt.ylabel('Stock Price')
        plt.title('Tesla Stock Price Forecast')
        plt.legend()
        plt.show()

    def interpret_forecast(self):
        """
        Interpret the forecast results for trend analysis, volatility, and market opportunities/risks.
        """
        if not hasattr(self, 'forecast_df'):
            raise Exception("No forecast data found. Call forecast_future() first.")
        
        forecast_mean = self.forecast_df['Forecast']
        
        print("---- Forecast Interpretation ----")
        
        # Trend Analysis
        print("\n1. Trend Analysis:")
        if forecast_mean.is_monotonic_increasing:
            print("- The forecast shows a consistent upward trend in stock prices.")
        elif forecast_mean.is_monotonic_decreasing:
            print("- The forecast indicates a consistent downward trend in stock prices.")
        else:
            print("- The forecast suggests fluctuations in stock prices, with no clear consistent trend.")

        # Volatility and Risk (without CI)
        print("\n2. Volatility and Risk Analysis:")
        print("- No confidence intervals available for volatility analysis.")
        
        # Market Opportunities and Risks
        print("\n3. Market Opportunities and Risks:")
        if forecast_mean.iloc[-1] > forecast_mean.iloc[0]:
            print("- Potential market opportunity due to expected price increase.")
        else:
            print("- Be cautious of potential market decline.")




import unittest
import numpy as np
from src.task2_forecast import train_lstm_model, prepare_data  # Example functions

class TestTask2Forecast(unittest.TestCase):
    def setUp(self):
        """Set up test data."""
        # Create dummy time series data for testing
        self.data = np.random.rand(1000, 1)  # 1000 data points
        self.time_steps = 60

    def test_prepare_data(self):
        """Test data preparation function."""
        X, y = prepare_data(self.data, self.time_steps)
        self.assertEqual(X.shape[1], self.time_steps)  # Check number of timesteps
        self.assertEqual(len(X), len(y))  # Ensure matching input-output pairs

    def test_train_lstm_model(self):
        """Test LSTM model training."""
        X, y = prepare_data(self.data, self.time_steps)
        X = X.reshape((X.shape[0], X.shape[1], 1))  # Reshape for LSTM
        model = train_lstm_model(X, y)
        self.assertIsNotNone(model)  # Ensure model is returned
        self.assertEqual(len(model.layers), 3)  # Check model layers (example)

if __name__ == '__main__':
    unittest.main()

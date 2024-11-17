import unittest
import pandas as pd
from src.task1_preprocess import load_and_clean_data  # Example function

class TestTask1Preprocess(unittest.TestCase):
    def setUp(self):
        """Set up test data."""
        self.tickers = ['TSLA', 'BND', 'SPY']
        self.start_date = '2010-01-01'
        self.end_date = '2023-12-31'

    def test_load_and_clean_data(self):
        """Test loading and cleaning data."""
        data = load_and_clean_data(self.tickers, self.start_date, self.end_date)
        self.assertIsInstance(data, pd.DataFrame)
        self.assertEqual(len(data.columns), len(self.tickers))  # Check number of columns
        self.assertFalse(data.isnull().values.any())  # Ensure no missing values

if __name__ == '__main__':
    unittest.main()

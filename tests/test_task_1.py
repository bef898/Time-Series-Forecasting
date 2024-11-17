import unittest
import pandas as pd
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.data_processor import FinancialDataAnalysis

financial_data_analysis = FinancialDataAnalysis()

class TestTask1Preprocess(unittest.TestCase):
    def setUp(self):
        """Set up test data."""
        self.tickers = ['TSLA', 'BND', 'SPY']
        self.start_date = '2010-01-01'
        self.end_date = '2023-12-31'

    def test_load_and_clean_data(self):
        """Test loading and cleaning data."""
        data = financial_data_analysis.load_historical_data(self.tickers, self.start_date, self.end_date)
        self.assertIsInstance(data, pd.DataFrame)
        self.assertEqual(len(data.columns), len(self.tickers))  # Check number of columns
        self.assertFalse(data.isnull().values.any())  # Ensure no missing values

if __name__ == '__main__':
    unittest.main()

import pandas as pd

def get_historical_data():
    # Load your historical data from CSV, database, etc.
    data = pd.read_csv('data/brent_oil_prices.csv')
    return data.to_dict(orient='records')

def get_forecast_data():
    # Generate or load forecast data (dummy example)
    forecast_data = {'date': ['2024-01-01'], 'price': [90.0]}
    return forecast_data

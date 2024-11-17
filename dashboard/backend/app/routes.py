from flask import Blueprint, jsonify, request
from .data_handler import get_historical_data, get_forecast_data

main = Blueprint('main', __name__)

@main.route('/api/historical', methods=['GET'])
def get_historical():
    data = get_historical_data()
    return jsonify(data)

@main.route('/api/forecast', methods=['GET'])
def get_forecast():
    data = get_forecast_data()
    return jsonify(data)

# Additional routes to handle event highlights, correlations, etc., can be added here

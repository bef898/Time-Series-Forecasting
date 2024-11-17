from flask import Flask
from flask_cors import CORS

def create_app():
    app = Flask(__name__)
    CORS(app)  # Enable CORS for communication with React frontend
    
    # Load configurations
    app.config.from_object('config.Config')

    # Register routes
    from .routes import main as main_blueprint
    app.register_blueprint(main_blueprint)

    return app

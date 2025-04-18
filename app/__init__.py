from flask import Flask
from .routes import main_routes

def create_app():
    app = Flask(__name__)
    app.config['SECRET_KEY'] = 'your-secret-key'
    app.register_blueprint(main_routes)
    return app

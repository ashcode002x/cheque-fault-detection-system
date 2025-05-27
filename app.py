from flask import Flask
import os
from config import Config
from database import check_table, create_banking_tables
from routes import init_routes

def create_app():
    """Create and configure the Flask application."""
    app = Flask(__name__)
    app.config.from_object(Config)
    
    # Ensure upload directories exist
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    os.makedirs(app.config['SIGNATURE_FOLDER'], exist_ok=True)
    os.makedirs(app.config['ORIGINAL_SIGNATURE_FOLDER'], exist_ok=True)
    
    # Initialize routes
    init_routes(app)
    
    return app

if __name__ == '__main__':
    app = create_app()
    check_table()  # Ensure the accounts table exists
    create_banking_tables()  # Ensure banking tables exist
    app.run(debug=True)
import os

class Config:
    SECRET_KEY = 'your_secret_key_here'
    UPLOAD_FOLDER = 'uploads'
    SIGNATURE_FOLDER = 'signatures'
    ORIGINAL_SIGNATURE_FOLDER = 'original-signatures'
    ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
    
    # Database configuration
    DATABASE = {
        'host': 'localhost',
        'user': 'postgres',
        'password': 'Babita',
        'dbname': 'chequedb'
    }
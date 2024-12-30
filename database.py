from flask import Flask
from flask_sqlalchemy import SQLAlchemy

# Initialize the Flask app and SQLAlchemy
app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///detections.db'
db = SQLAlchemy(app)

# Define models
class MotionDetection(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    timestamp = db.Column(db.String(100), nullable=False)
    motion = db.Column(db.String(100), nullable=False)

    def __repr__(self):
        return f"<MotionDetection {self.id}, {self.timestamp}, {self.motion}>"

class EmotionDetection(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    timestamp = db.Column(db.String(100), nullable=False)
    emotion = db.Column(db.String(100), nullable=False)

    def __repr__(self):
        return f"<EmotionDetection {self.id}, {self.timestamp}, {self.emotion}>"

# Create the tables (SQLAlchemy handles this for you)
with app.app_context():
    db.create_all()

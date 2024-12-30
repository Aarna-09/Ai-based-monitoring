from datetime import datetime
from app import db  # Import the Flask app and db instance

class EmotionDetection(db.Model):
    __tablename__ = 'emotion_detection'
    
    id = db.Column(db.Integer, primary_key=True)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)
    emotion = db.Column(db.String(50))

    def __init__(self, timestamp, emotion):
        self.timestamp = timestamp
        self.emotion = emotion

    def __repr__(self):
        return f"<EmotionDetection {self.emotion} at {self.timestamp}>"

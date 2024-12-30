from flask import Flask, render_template, request
from flask_sqlalchemy import SQLAlchemy
import threading
import subprocess
from datetime import datetime

app = Flask(__name__)

# Database setup
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///detections.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)

# Database Models
class EmotionDetection(db.Model):
    __tablename__ = 'emotion_detection'
    id = db.Column(db.Integer, primary_key=True)
    timestamp = db.Column(db.DateTime, nullable=False)
    emotion = db.Column(db.String(100), nullable=False)

    def __repr__(self):
        return f'<EmotionDetection {self.id}, {self.timestamp}, {self.emotion}>'

class MotionDetection(db.Model):
    __tablename__ = 'motion_detection'
    id = db.Column(db.Integer, primary_key=True)
    timestamp = db.Column(db.DateTime, nullable=False)
    motion = db.Column(db.String(100), nullable=False)

    def __repr__(self):
        return f'<MotionDetection {self.id}, {self.timestamp}, {self.motion}>'

# Create tables (only once)
with app.app_context():
    db.create_all()

# Routes
@app.route('/')
def home():
    return render_template('index.html')  # Render the frontend page

@app.route('/start_emotion_detection', methods=['POST'])
def start_emotion_detection():
    def run_emotion_model():
        subprocess.run(['python', 'emotion.py'])  # Run the emotion model script

    threading.Thread(target=run_emotion_model, daemon=True).start()  # Run in a separate thread
    return "Emotion detection started!"

@app.route('/start_motion_detection', methods=['POST'])
def start_motion_detection():
    def run_motion_model():
        subprocess.run(['python', 'motion.py'])  # Run the motion model script

    threading.Thread(target=run_motion_model, daemon=True).start()  # Run in a separate thread
    return "Motion detection started!"

@app.route('/view_data')
def view_data():
    # Fetch emotion and motion data from the database
    emotion_data = EmotionDetection.query.all()
    motion_data = MotionDetection.query.all()
    motion_data = [{
        'id': motion.id,
        'timestamp': motion.timestamp,
        'motion': motion.motion
    } for motion in motion_data]

    # Convert the emotion data to a serializable format (list of dictionaries)
    emotion_data = [{
        'id': emotion.id,
        'timestamp': emotion.timestamp,
        'emotion': emotion.emotion
    } for emotion in emotion_data]
    return render_template('view_data.html', emotion_data=emotion_data, motion_data=motion_data)
# Save emotion detection data in database
def save_emotion_data(emotion):
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    new_record = EmotionDetection(timestamp=timestamp, emotion=emotion)
    db.session.add(new_record)
    db.session.commit()
    print(f"Emotion data saved: {emotion} at {timestamp}")

# Save motion detection data in database
def save_motion_data(motion):
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    new_record = MotionDetection(timestamp=timestamp, motion=motion)  # Use motion_status here
    db.session.add(new_record)
    db.session.commit()
    print(f"Motion data saved: {motion} at {timestamp}")

if __name__ == '__main__':
    app.run(debug=True)

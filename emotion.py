import cv2
import numpy as np
import datetime
from tensorflow.keras.models import load_model
from flask import Flask
from flask_sqlalchemy import SQLAlchemy

from playsound import playsound

# Initialize Flask app and database configuration
app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///detections.db'  # Update to your database URI
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)

# EmotionDetection Model - Defining the database table structure
class EmotionDetection(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    timestamp = db.Column(db.String(100))
    emotion = db.Column(db.String(50))

# Load the pre-trained emotion recognition model
model = load_model('emotion_recognition_model.h5')

# Emotion labels (ensure these match the output of your model)
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

def trigger_alarm():
    # Path to the alarm sound file (change this to your actual alarm sound file)
    alarm_sound_path = r'C:\Users\saanj\Desktop\detection\sound.mp3'  # Ensure this is a valid path
    playsound(alarm_sound_path)

# Save the emotion event to the database
def save_emotion_event(emotion):
    with app.app_context():  # Ensures the database interaction happens within the app context
        timestamp = str(datetime.datetime.now())  # Get the current timestamp
        new_event = EmotionDetection(timestamp=timestamp, emotion=emotion)
        
        db.session.add(new_event)
        db.session.commit()
        print(f"Emotion saved: {emotion} at {timestamp}")

# Function to predict emotion from the frame
def predict_emotion_from_frame(frame):
    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Load a pre-trained face detector (Haar Cascade)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    # Detect faces in the frame
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(48, 48))
    
    # Loop over detected faces and predict emotion
    for (x, y, w, h) in faces:
        face = gray[y:y+h, x:x+w]
        face_resized = cv2.resize(face, (48, 48))  # Resize to match model input size
        face_resized = face_resized.astype('float32') / 255  # Normalize the pixel values
        face_resized = np.expand_dims(face_resized, axis=-1)  # Add channel dimension (1 channel for grayscale)
        
        # Convert the single-channel image to a 3-channel image (RGB)
        face_resized_rgb = np.repeat(face_resized, 3, axis=-1)  # Repeat the grayscale channel to create an RGB image
        face_resized_rgb = np.expand_dims(face_resized_rgb, axis=0)  # Add batch dimension
        
        # Predict the emotion
        emotion_predictions = model.predict(face_resized_rgb)
        max_index = np.argmax(emotion_predictions[0])
        emotion = emotion_labels[max_index]
        
        # Check if the detected emotion is 'Fear' or 'Angry' and trigger alarm
        if emotion in ['Fear', 'Angry']:
            print(f"Emotion detected: {emotion}. Triggering alarm!")
            trigger_alarm()  # Trigger the alarm

        # Save the detected emotion to the database
        save_emotion_event(emotion)
        
        return emotion  # Return the detected emotion for further use

# Main function to run emotion detection
def run_emotion_detection():
    # Start capturing video (webcam or a video file)
    cap = cv2.VideoCapture(0)  # Use 0 for webcam, or specify a file path
    
    while True:
        # Read a frame from the video capture
        ret, frame = cap.read()
        if not ret:
            break
        
        # Call the emotion prediction function
        emotion = predict_emotion_from_frame(frame)
        
        # Display the frame with the detected emotion
        cv2.putText(frame, f'Emotion: {emotion}', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow('Emotion Detection', frame)
        
        # Exit on pressing 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Release the video capture and close the windows
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run_emotion_detection()
import cv2
import cvzone
import math
import threading
import os
from datetime import datetime
from playsound import playsound
import hashlib
import json

# Blockchain Implementation
class Blockchain:
    def __init__(self):
        self.chain = []
        self.create_block(previous_hash='0')  # Genesis block

    def create_block(self, motion=None, timestamp=None, previous_hash='0'):
        block = {
            'index': len(self.chain) + 1,
            'timestamp': timestamp if timestamp else datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'motion': motion,
            'previous_hash': previous_hash,
            'hash': ''
        }
        block['hash'] = self.hash_block(block)
        self.chain.append(block)
        return block

    @staticmethod
    def hash_block(block):
        block_string = json.dumps(block, sort_keys=True).encode()
        return hashlib.sha256(block_string).hexdigest()

    def get_last_block(self):
        return self.chain[-1]

    def add_event(self, motion):
        last_block = self.get_last_block()
        new_block = self.create_block(
            motion=motion,
            previous_hash=last_block['hash']
        )
        return new_block


# Initialize the blockchain
blockchain = Blockchain()

# Alarm function
def play_alarm():
    playsound(r'C:\Asan\detection\sound.mp3')  # Path to your alarm sound file

# Initialize the webcam
cap = cv2.VideoCapture(0)  # Use 0 for default camera

# Load the YOLO model
from ultralytics import YOLO
model = YOLO('yolov8s.pt')

# Load class names
classnames = []
classes_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'classes.txt')
with open(classes_path, 'r') as f:
    classnames = f.read().splitlines()

# Track alarm state and person's position
alarm_triggered = False
last_position = None
static_frame_count = 0
unconscious_threshold = 50  # Number of frames to detect unconsciousness

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame.")
        break

    # Resize frame for consistency
    frame = cv2.resize(frame, (980, 740))

    # Detect objects using YOLO
    results = model(frame)

    for info in results:
        parameters = info.boxes
        for box in parameters:
            # Get bounding box coordinates
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            confidence = box.conf[0]
            class_detect = box.cls[0]
            class_detect = int(class_detect)
            class_detect = classnames[class_detect]
            conf = math.ceil(confidence * 100)

            # Implement fall detection using bounding box dimensions
            height = y2 - y1
            width = x2 - x1
            aspect_ratio = width / height

            if conf > 80 and class_detect == 'person':
                # Draw bounding box and label
                cvzone.cornerRect(frame, [x1, y1, width, height], l=30, rt=6)
                cvzone.putTextRect(frame, f'{class_detect}', [x1 + 8, y1 - 12], thickness=2, scale=2)

                # Detect fall based on aspect ratio
                if aspect_ratio > 1.5:
                    blockchain.add_event("Fall Detected")
                    cvzone.putTextRect(frame, 'Fall Detected', [x1, y1 - 50], thickness=2, scale=2, colorR=(0, 0, 255))
                    
                    # Trigger alarm for fall
                    if not alarm_triggered:
                        alarm_triggered = True
                        threading.Thread(target=play_alarm, daemon=True).start()

                # Detect dizziness (erratic movement)
                current_position = ((x1 + x2) // 2, (y1 + y2) // 2)
                if last_position is not None:
                    movement = math.sqrt((current_position[0] - last_position[0]) ** 2 +
                                         (current_position[1] - last_position[1]) ** 2)
                    if movement > 50:  # Dizziness threshold (may need adjustment based on testing)
                        blockchain.add_event("Dizziness Detected")
                        cvzone.putTextRect(frame, 'Dizziness Detected', [x1, y1 - 80], thickness=2, scale=2, colorR=(255, 165, 0))
                else:
                    movement = 0  # Default value when last_position is None

                last_position = current_position

                # Detect unconsciousness (inactivity)
                if movement < 10:  # Inactivity threshold (may need adjustment based on testing)
                    static_frame_count += 1
                    if static_frame_count > unconscious_threshold:
                        blockchain.add_event("Unconscious Detected")
                        cvzone.putTextRect(frame, 'Unconscious Detected', [x1, y1 - 110], thickness=2, scale=2, colorR=(255, 0, 0))
                        
                        if not alarm_triggered:
                            alarm_triggered = True
                            threading.Thread(target=play_alarm, daemon=True).start()
                else:
                    static_frame_count = 0

    # Display the frame
    cv2.imshow('Live Fall and Behavior Detection', frame)

    # Exit on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()

# Print the blockchain for debugging
print("\nBlockchain:")
for block in blockchain.chain:
    print(json.dumps(block, indent=4))

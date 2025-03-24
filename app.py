import numpy as np
from gym import Env, spaces
import random
import cv2
from flask import Flask, render_template, Response, request, jsonify
from stable_baselines3 import PPO
import os
import yolov5
from threading import Thread
import json

# Flask app initialization
app = Flask(__name__, template_folder='templates', static_folder='static')

# Initialize YOLOv5 model
try:
   model_yolo = yolov5.load('./yolov5x.pt')  # or './yolov5x.pt'
   print("YOLOv5 model loaded successfully.")
except Exception as e:
    print(f"Error loading YOLOv5 model: {e}")
    model_yolo = None

# Global Variables
tracked_paths = {}
last_detected_frame = {}  # Track the last frame each object was detected
object_colors = {}
allowed_classes = []  # List of classes to track
max_allowed_objects = 10  # Threshold for alerts
detected_object_names = set()  # Use a set to store unique detected object names

# COCO class names mapping
COCO_CLASSES = {
    0: "Person",
    1: "Bicycle",
    2: "Car",
    3: "Motorcycle",
    4: "Airplane",
    5: "Bus",
    6: "Train",
    7: "Truck",
    8: "Boat",
    9: "Traffic Light",
    10: "Fire Hydrant",
    11: "Stop Sign",
    12: "Parking Meter",
    13: "Bench",
    14: "Bird",
    15: "Cat",
    16: "Dog",
    17: "Horse",
    18: "Sheep",
    19: "Cow",
    20: "Elephant",
    21: "Bear",
    22: "Zebra",
    23: "Giraffe",
    24: "Backpack",
    25: "Umbrella",
    26: "Handbag",
    27: "Tie",
    28: "Suitcase",
    29: "Frisbee",
    30: "Skis",
    31: "Snowboard",
    32: "Sports Ball",
    33: "Kite",
    34: "Baseball Bat",
    35: "Baseball Glove",
    36: "Skateboard",
    37: "Surfboard",
    38: "Tennis Racket",
    39: "Bottle",
    40: "Wine Glass",
    41: "Cup",
    42: "Fork",
    43: "Knife",
    44: "Spoon",
    45: "Bowl",
    46: "Banana",
    47: "Apple",
    48: "Sandwich",
    49: "Orange",
    50: "Broccoli",
    51: "Carrot",
    52: "Hot Dog",
    53: "Pizza",
    54: "Donut",
    55: "Cake",
    56: "Chair",
    57: "Couch",
    58: "Potted Plant",
    59: "Bed",
    60: "Dining Table",
    61: "Toilet",
    62: "TV",
    63: "Laptop",
    64: "Mouse",
    65: "Remote",
    66: "Keyboard",
    67: "Cell Phone",
    68: "Microwave",
    69: "Oven",
    70: "Toaster",
    71: "Sink",
    72: "Refrigerator",
    73: "Book",
    74: "Clock",
    75: "Vase",
    76: "Scissors",
    77: "Teddy Bear",
    78: "Hair Drier",
    79: "Toothbrush"
}

# Assign random colors to objects
def get_object_color(object_id):
    if object_id not in object_colors:
        object_colors[object_id] = tuple(np.random.randint(0, 255, 3).tolist())
    return object_colors[object_id]

# RL Environment for Multi-Object Tracking
class MultiObjectTrackingEnv(Env):
    def __init__(self, num_objects):
        super(MultiObjectTrackingEnv, self).__init__()
        self.num_objects = num_objects
        self.observation_space = spaces.Box(low=0, high=1, shape=(num_objects, 4), dtype=np.float32)
        self.action_space = spaces.Discrete(num_objects)
        self.objects = None
        self.target = None

    def reset(self):
        self.objects = np.random.rand(self.num_objects, 4)  # (x, y, vx, vy)
        self.objects[:, 2:] = np.random.uniform(-0.01, 0.01, (self.num_objects, 2))
        self.target = random.randint(0, self.num_objects - 1)
        return self.objects

    def step(self, action):
        acceleration = np.random.uniform(-0.002, 0.002, self.objects[:, 2:].shape)
        self.objects[:, 2:] += acceleration
        self.objects[:, :2] += self.objects[:, 2:]
        self.objects[:, :2] = np.clip(self.objects[:, :2], 0, 1)
        target_position = self.objects[self.target, :2]
        action_position = self.objects[action, :2]
        reward = 1.0 - np.linalg.norm(target_position - action_position)
        done = False
        return self.objects, reward, done, {}

# Train or Load PPO Models
def train_or_load_ppo_models(shapes, device="cpu"):
    models = {}
    for shape in shapes:
        num_objects = shape[0]
        model_path = f"multi_object_tracking_ppo_{num_objects}_objects.zip"
        if os.path.exists(model_path):
            print(f"Loading pre-trained model for observation shape: {shape}")
            models[shape] = PPO.load(model_path, device=device)
        else:
            print(f"Training PPO model for observation shape: {shape}")
            env = MultiObjectTrackingEnv(num_objects=num_objects)
            model = PPO(
                "MlpPolicy",
                env,
                verbose=1,
                n_steps=2048,
                batch_size=64,
                n_epochs=50,
                learning_rate=3e-4,
                device=device
            )
            model.learn(total_timesteps=500000)
            model.save(model_path)
            models[shape] = model
            print(f"Model for shape {shape} saved.")
    return models

# Train or load PPO models for the defined observation shapes
USE_GPU = True  # Set to True to use GPU
model_device = "cuda" if USE_GPU else "cpu"
trained_models = train_or_load_ppo_models([(5, 4), (10, 4), (15, 4)], device=model_device)

# Real-Time Object Detection and Tracking
def process_frame():
    global tracked_paths, last_detected_frame, detected_object_names
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        return (b'--frame\r\n'
                b'Content-Type: text/plain\r\n\r\n'
                b'Error: Could not access the webcam.\r\n')

    frame_counter = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame from webcam.")
            break

        frame_counter += 1

        # Perform YOLOv5 object detection
        results = model_yolo(frame) if model_yolo else None
        if results:
            detections = results.xyxy[0].cpu().numpy()
        else:
            detections = []

        frame_height, frame_width = frame.shape[:2]
        path_canvas = np.zeros((frame_height, frame_width, 3), dtype=np.uint8)  # Black canvas

        # Process detections
        current_detected_ids = set()

        for detection in detections:
            x_min, y_min, x_max, y_max, confidence, class_id = detection
            if confidence > 0.3 and (not allowed_classes or int(class_id) in allowed_classes):
                x, y = (x_min + x_max) / 2, (y_min + y_max) / 2
                width, height = x_max - x_min, y_max - y_min
                size = (width * height) / (frame_width * frame_height)
                if size < 0.01 or size > 1.5:
                    continue

                class_id = int(class_id)
                current_detected_ids.add(class_id)
                last_detected_frame[class_id] = frame_counter

                if class_id not in tracked_paths:
                    tracked_paths[class_id] = []

                tracked_paths[class_id].append((int(x), int(y)))
                if len(tracked_paths[class_id]) > 50:
                    tracked_paths[class_id] = tracked_paths[class_id][-50:]

                label = f"{COCO_CLASSES.get(class_id, 'Unknown')}: {confidence:.2f}"
                cv2.rectangle(frame, (int(x_min), int(y_min)), (int(x_max), int(y_max)), get_object_color(class_id), 2)
                cv2.putText(frame, label, (int(x_min), int(y_min) - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, get_object_color(class_id), 2)

                # Add detected object name to the set
                detected_object_names.add(COCO_CLASSES.get(class_id, 'Unknown'))
                # Add object image at the front of the path
                object_image = frame[int(y_min):int(y_max), int(x_min):int(x_max)]
                if object_image.size > 0:
                    resized_object = cv2.resize(object_image, (50, 50))
                    if len(tracked_paths[class_id]) > 1:
                        last_point = tracked_paths[class_id][-1]
                        y_offset, x_offset = last_point[1] - 25, last_point[0] - 25
                        y1, y2 = max(0, y_offset), min(frame_height, y_offset + 50)
                        x1, x2 = max(0, x_offset), min(frame_width, x_offset + 50)
                        overlay = path_canvas[y1:y2, x1:x2]
                        blended = cv2.addWeighted(overlay, 0.5, resized_object[:y2-y1, :x2-x1], 0.5, 0)
                        path_canvas[y1:y2, x1:x2] = blended

        # Remove paths for objects not detected recently
        disappearing_threshold = 30  # Number of frames to wait before removing path
        for class_id in list(tracked_paths.keys()):
            if frame_counter - last_detected_frame.get(class_id, 0) > disappearing_threshold:
                del tracked_paths[class_id]

        for class_id, path_points in tracked_paths.items():
            color = get_object_color(class_id)
            for i in range(1, len(path_points)):
                cv2.line(
                    path_canvas,
                    path_points[i - 1],
                    path_points[i],
                    color, 2
                )

        frame_combined = np.hstack((frame, path_canvas))
        _, buffer = cv2.imencode('.jpg', frame_combined)
        frame_combined = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_combined + b'\r\n')

    cap.release()

@app.route('/')
def index():
    return render_template("index.html")

@app.route('/detected_objects')
def detected_objects():
    return jsonify(list(detected_object_names))

@app.route('/video_feed')
def video_feed():
    return Response(process_frame(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/set_classes', methods=['POST'])
def set_classes():
    global allowed_classes
    allowed_classes = request.json.get('classes', [])
    return jsonify({'status': 'updated'})

@app.route('/save_paths')
def save_paths():
    with open("tracked_paths.json", "w") as f:
        json.dump(tracked_paths, f)
    return jsonify({'status': 'saved'})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)

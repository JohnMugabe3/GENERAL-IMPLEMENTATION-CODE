import os
import sys
import pybullet as p
import pybullet_data
import torch
import torch.nn as nn
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PyQt5 import QtWidgets, QtCore, QtGui
from concurrent.futures import ThreadPoolExecutor

# StereoNet Model Definition
class StereoNet(nn.Module):
    def __init__(self, in_channels):
        super(StereoNet, self).__init__()
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        self.refiner = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 1, kernel_size=3, padding=1),
            nn.Upsample(size=(480, 640), mode='bilinear', align_corners=True)
        )

    def forward(self, left_images, right_images):
        left_features = self.feature_extractor(left_images)
        right_features = self.feature_extractor(right_images)
        combined_features = torch.cat((left_features, right_features), dim=1)
        depth_map = self.refiner(combined_features)
        return depth_map

# LSTM Model Definition for Collision Prediction
class LSTMPredictor(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers):
        super(LSTMPredictor, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h_0 = torch.zeros(self.lstm.num_layers, x.size(0), self.lstm.hidden_size).to(x.device)
        c_0 = torch.zeros(self.lstm.num_layers, x.size(0), self.lstm.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h_0, c_0))
        out = self.fc(out[:, -1, :])
        return out

# YOLO Paths Specification
yolo_weights = r"C:\Users\johnm\Desktop\THESIS 19.08.2024\PyBULLET 2\StereoNet_PyTorch\drone-net\yolo-drone.weights"
yolo_cfg = r"C:\Users\johnm\Desktop\THESIS 19.08.2024\PyBULLET 2\StereoNet_PyTorch\drone-net\yolo-drone.cfg"
yolo_names = r"C:\Users\johnm\Desktop\THESIS 19.08.2024\PyBULLET 2\StereoNet_PyTorch\drone-net\drone.names"

# Load YOLO Model
net = cv2.dnn.readNet(yolo_weights, yolo_cfg)
with open(yolo_names, 'r') as f:
    classes = f.read().strip().split('\n')

layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

# Connect to PyBullet and Load Environment
p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
environment_urdf = r"C:\Users\johnm\Desktop\THESIS 19.08.2024\PyBULLET 2\StereoNet_PyTorch\SketchFab_Blender_NYC Environment\enviro.urdf"

if not os.path.exists(environment_urdf):
    print(f"URDF file not found at: {environment_urdf}")
    sys.exit()

environment_orientation = p.getQuaternionFromEuler([np.pi / 2, 0, 0])
environment_id = p.loadURDF(environment_urdf, basePosition=[0, 0, 9], baseOrientation=environment_orientation, useFixedBase=True, globalScaling=7)

# Define Path to Drone URDF
drone_urdf = r"C:\Users\johnm\Desktop\THESIS 19.08.2024\PyBULLET 1\rotors_simulator\rotors_description\urdf\ardrone.urdf"
if not os.path.exists(drone_urdf):
    print(f"URDF file not found at: {drone_urdf}")
    sys.exit()

# Create Drones and Set Starting Positions
drone1 = p.loadURDF(drone_urdf, [1, -1, 1])
drone2 = p.loadURDF(drone_urdf, [1, -1, 1])
drone3 = p.loadURDF(drone_urdf, [1, 1, 1])
initial_orientation_drone3 = p.getQuaternionFromEuler([0, 0, -np.deg2rad(0)])

# Camera Calibration Data Loading
calibration_file = r"C:\Users\johnm\Desktop\THESIS 19.08.2024\PyBULLET 1\ComputerVision\StereoVisionDepthEstimation\calibration_data.npz"
with np.load(calibration_file) as data:
    K1 = data['K1']
    D1 = data['D1']
    K2 = data['K2']
    D2 = data['D2']
    R = data['R']
    T = data['T']

# Initialize StereoNet
stereo_model = StereoNet(in_channels=3)
stereo_model = stereo_model.cpu()
stereo_model.eval()

# Load Pre-Trained StereoNet Weights
stereo_checkpoint_path = r"C:\Users\johnm\Desktop\THESIS 19.08.2024\PyBULLET 2\StereoNet_PyTorch\TRAINING STEREONET & LSTM BEST RESULTS\trained_stereonet.pth"
stereo_checkpoint = torch.load(stereo_checkpoint_path, map_location=torch.device('cpu'))

stereo_model_state_dict = stereo_model.state_dict()
adjusted_stereo_checkpoint = {}

for key in stereo_checkpoint:
    if key in stereo_model_state_dict:
        adjusted_stereo_checkpoint[key] = stereo_checkpoint[key]
    else:
        print(f"Skipping {key} as it is not found in the model's state_dict.")

stereo_model.load_state_dict(adjusted_stereo_checkpoint, strict=False)

# Load Pre-Trained LSTM Model
lstm_model_path = r"C:\Users\johnm\Desktop\THESIS 19.08.2024\PyBULLET 2\StereoNet_PyTorch\TRAINING STEREONET & LSTM BEST RESULTS\lstm_model.pth"

input_size = 3
hidden_size = 25
output_size = 3
num_layers = 1

def load_lstm_model(model_path, input_size, hidden_size, output_size, num_layers):
    lstm_model = LSTMPredictor(input_size, hidden_size, output_size, num_layers)
    checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
    lstm_model.load_state_dict(checkpoint)
    lstm_model.eval()
    return lstm_model

lstm_model = load_lstm_model(lstm_model_path, input_size, hidden_size, output_size, num_layers)

def detect_objects(image):
    height, width, _ = image.shape
    blob = cv2.dnn.blobFromImage(image, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)
    class_ids = []
    confidences = []
    boxes = []
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)
    indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
    detections = []
    if len(indices) > 0:
        for i in indices.flatten():
            box = boxes[i]
            detections.append((class_ids[i], confidences[i], box))
    return detections

def draw_boxes(img, detections):
    for (class_id, confidence, box) in detections:
        x, y, w, h = box
        label = f"{str(classes[class_id])} ({confidence:.2f})"
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(img, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    return img

def capture_stereo_images():
    width, height = 640, 480  
    fov = 60
    aspect = width / height
    near = 0.1
    far = 100
    proj_matrix = p.computeProjectionMatrixFOV(fov, aspect, near, far)
    
    drone3_position, drone3_orientation = p.getBasePositionAndOrientation(drone3)
    left_camera_trans = [0.2, 0.1, 0.1]  
    right_camera_trans = [0.2, -0.1, 0.1]  

    left_camera = p.multiplyTransforms(drone3_position, drone3_orientation, left_camera_trans, [0, 0, 0, 1])
    right_camera = p.multiplyTransforms(drone3_position, drone3_orientation, right_camera_trans, [0, 0, 0, 1])

    left_view_matrix = p.computeViewMatrix(
        cameraEyePosition=left_camera[0],
        cameraTargetPosition=[left_camera[0][0] + 1, left_camera[0][1], left_camera[0][2]],
        cameraUpVector=[0, 0, 1]
    )
    
    right_view_matrix = p.computeViewMatrix(
        cameraEyePosition=right_camera[0],
        cameraTargetPosition=[right_camera[0][0] + 1, right_camera[0][1], right_camera[0][2]],
        cameraUpVector=[0, 0, 1]
    )

    # Capture images
    left_img = p.getCameraImage(width, height, viewMatrix=left_view_matrix, projectionMatrix=proj_matrix)[2]
    right_img = p.getCameraImage(width, height, viewMatrix=right_view_matrix, projectionMatrix=proj_matrix)[2]
    depth_img = p.getCameraImage(width, height, viewMatrix=left_view_matrix, projectionMatrix=proj_matrix, renderer=p.ER_TINY_RENDERER)[3]

    # Convert to numpy arrays and then to uint8
    left_img = np.reshape(left_img, (height, width, 4))[:, :, :3].astype(np.uint8)
    right_img = np.reshape(right_img, (height, width, 4))[:, :, :3].astype(np.uint8)
    
    # Convert depth buffer to depth map
    depth_buffer = np.reshape(depth_img, (height, width))
    depth_map = far * near / (far - (far - near) * depth_buffer)
    
    return left_img, right_img

def preprocess_image(image):
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    image = torch.tensor(image).permute(2, 0, 1).unsqueeze(0).float() / 255.0
    return image

def estimate_depth(left_img, right_img):
    left_img = preprocess_image(left_img)
    right_img = preprocess_image(right_img)

    # FORWARD PASS THROUGH THE MODEL
    with torch.no_grad():
        depth_map = stereo_model(left_img, right_img)

    # CONVERTING DEPTH MAP TO NUMPY
    depth_map = depth_map.squeeze().cpu().numpy()
    return depth_map

def calculate_accuracy_metrics(est_depth_map, gt_depth_map):
    if est_depth_map.shape != gt_depth_map.shape:
        raise ValueError("Shape mismatch between estimated and ground truth depth maps.")

    mae = np.mean(np.abs(est_depth_map - gt_depth_map))
    rmse = np.sqrt(np.mean((est_depth_map - gt_depth_map) ** 2))

    return mae, rmse

def calculate_distance(detections, depth_map, width, height, drone_index):

    # Ensure the drone index is valid
    if drone_index >= len(detections):
        return float('inf')  

    # Get the bounding box for the specified drone index
    _, _, box = detections[drone_index]
    x, y, w, h = box

    # Ensure that the bounding box coordinates are within image bounds
    x1, y1 = max(0, x), max(0, y)
    x2, y2 = min(width, x + w), min(height, y + h)

    # Extract the depth values within the bounding box
    depth_values = depth_map[y1:y2, x1:x2]

    # Calculate the average depth value (ignoring zero values)
    valid_depth_values = depth_values[depth_values > 0]
    if len(valid_depth_values) == 0:
        return float('inf')  # Return an infinite distance if there are no valid depth values

    avg_depth = np.mean(valid_depth_values) * 10000  

def move_drones_and_capture_depth(update_data_signal, update_images_signal, update_alert_signal, update_positions_signal):
    width, height = 640, 480  
    path_data_drone1 = []
    path_data_drone2 = []
    positions_history_drone1 = []
    positions_history_drone2 = []

    # Initialize previous distances
    previous_distance_drone1 = float('inf')
    previous_distance_drone2 = float('inf')

    for step in range(1000):
        step_size = 0.005

        # MOVING DRONE 1 IN A LINEAR PATH TOWARDS THE ORIGIN
        drone1_position = [1 - step_size * step, -0.2, 14]
        p.resetBasePositionAndOrientation(drone1, drone1_position, [0, 0, 0, 1])

        # MOVING DRONE 2 IN A LINEAR PATH AS WELL TOWARDS THE ORIGIN
        drone2_position = [1 - step_size * step, -0.9, 14]
        p.resetBasePositionAndOrientation(drone2, drone2_position, [0, 0, 0, 1])

        # MOVING DRONE 3 IN A LINEAR PATH TOWARDS DRONE 1 AND DRONE 2
        drone3_position = [-1 + step_size * step, -0.86, 14]
        p.resetBasePositionAndOrientation(drone3, drone3_position, initial_orientation_drone3)
        p.stepSimulation()

        left_img, right_img = capture_stereo_images()

        # DETECTION OBJECTS USING THE LOADED YOLO OBJECT DETECTOR
        left_detections = detect_objects(left_img)
        right_detections = detect_objects(right_img)

        # CHECKING FOR EMPTY DETECTIONS
        if not left_detections and not right_detections:
            continue  

        # DRAWING BOUND BOXES ON THE DETECTED IMAGES
        left_img_with_boxes = draw_boxes(left_img.copy(), left_detections)
        right_img_with_boxes = draw_boxes(right_img.copy(), right_detections)

        depth_map = estimate_depth(left_img, right_img)

        # NORMALIZING THE DEPTH MAP FOR VISUALIZATION
        depth_map_normalized = cv2.normalize(depth_map, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8UC1)
        depth_map_colored = cv2.applyColorMap(depth_map_normalized, cv2.COLORMAP_JET)

        gt_depth_map = np.ones_like(depth_map) * 3
        mae, rmse = calculate_accuracy_metrics(depth_map, gt_depth_map)
        print(f"Step {step}, MAE: {mae:.4f}, RMSE: {rmse:.4f}")

        # CALCULATING DRONE 1 AND DRONE 2 DISTANCES AT THE SAME TIME (CONCURRENTLY)
        distances = {"Drone 1": 0, "Drone 2": 0}
        with ThreadPoolExecutor() as executor:
            future1 = executor.submit(calculate_distance, left_detections, depth_map, width, height, 0)
            future2 = executor.submit(calculate_distance, left_detections, depth_map, width, height, 1)
            new_distance_drone1 = future1.result()
            new_distance_drone2 = future2.result()

        # UPDATING DISTANCES
        if new_distance_drone1 < previous_distance_drone1:
            distances["Drone 1"] = new_distance_drone1
            previous_distance_drone1 = new_distance_drone1
            positions_history_drone1.append(new_distance_drone1)
            if len(positions_history_drone1) > 8:
                positions_history_drone1.pop(0)
        else:
            distances["Drone 1"] = previous_distance_drone1  

        if new_distance_drone2 < previous_distance_drone2:
            distances["Drone 2"] = new_distance_drone2
            previous_distance_drone2 = new_distance_drone2
            positions_history_drone2.append(new_distance_drone2)
            if len(positions_history_drone2) > 8:
                positions_history_drone2.pop(0)
        else:
            distances["Drone 2"] = previous_distance_drone2  

        # UPDATING PATH DATA FOR LSTM
        path_data_drone1.append(drone1_position[:3])  
        path_data_drone2.append(drone2_position[:3])  

        if len(path_data_drone1) >= 10:  # USING THE LAST 10 POSITIONS FOR PREDICTION
            input_data_drone1 = torch.tensor(np.array(path_data_drone1[-10:]), dtype=torch.float32).unsqueeze(0)
            input_data_drone2 = torch.tensor(np.array(path_data_drone2[-10:]), dtype=torch.float32).unsqueeze(0)

            lstm_model.eval()
            with torch.no_grad():
                predicted_position_drone1 = lstm_model(input_data_drone1).cpu().numpy()
                predicted_position_drone2 = lstm_model(input_data_drone2).cpu().numpy()

            print(f"Predicted Position Drone 1: {predicted_position_drone1}, shape: {predicted_position_drone1.shape}")
            print(f"Predicted Position Drone 2: {predicted_position_drone2}, shape: {predicted_position_drone2.shape}")

            predicted_distance_drone1 = np.linalg.norm(predicted_position_drone1[0] - np.array(drone3_position[:3])) * 10000  
            predicted_distance_drone2 = np.linalg.norm(predicted_position_drone2[0] - np.array(drone3_position[:3])) * 10000  

            # Calculate the time to collision
            velocity_drone1 = np.linalg.norm(predicted_position_drone1[0] - np.array(path_data_drone1[-1])) / step_size
            velocity_drone2 = np.linalg.norm(predicted_position_drone2[0] - np.array(path_data_drone2[-1])) / step_size

            time_to_collision_drone1 = predicted_distance_drone1 / velocity_drone1 if velocity_drone1 != 0 else float('inf')
            time_to_collision_drone2 = predicted_distance_drone2 / velocity_drone2 if velocity_drone2 != 0 else float('inf')

            alert_message = "None"
            if distances["Drone 1"] <= 3000 and time_to_collision_drone1 < float('inf'): 
                if predicted_position_drone1[0, 1] > drone3_position[1]:
                    alert_message = f"Drone Detected - Front Left! Collision in {time_to_collision_drone1:.2f} seconds"
                else:
                    alert_message = f"Drone Detected - Front Right! Collision in {time_to_collision_drone1:.2f} seconds"
            elif distances["Drone 2"] <= 3000 and time_to_collision_drone2 < float('inf'):  
                if predicted_position_drone2[0, 1] > drone3_position[1]:
                    alert_message = f"Drone Detected - Front Left! Collision in {time_to_collision_drone2:.2f} seconds"
                else:
                    alert_message = f"Drone Detected - Front Right! Collision in {time_to_collision_drone2:.2f} seconds"

            update_alert_signal.emit(alert_message)

        # UPDATING THE GUI WITH DISTANCES AND IMAGES
        update_data_signal.emit(distances)
        update_images_signal.emit(left_img_with_boxes, right_img_with_boxes, depth_map_colored)

        # UPDATING POSITIONS HISTORY FOR THE GUI
        positions_history_str_drone1 = "\n".join([f"Drone 1 Distance {i+1}: {dist:.2f} cm" for i, dist in enumerate(positions_history_drone1)])
        positions_history_str_drone2 = "\n".join([f"Drone 2 Distance {i+1}: {dist:.2f} cm" for i, dist in enumerate(positions_history_drone2)])
        update_positions_signal.emit(positions_history_str_drone1 + "\n" + positions_history_str_drone2)

class DroneDistanceApp(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()
        self.drones_distances = {"Drone 1": 0, "Drone 2": 0}

        self.worker_thread = QtCore.QThread()
        self.worker = DroneWorker()
        self.worker.moveToThread(self.worker_thread)

        self.worker.update_data.connect(self.update_distances)
        self.worker.update_images.connect(self.update_images)
        self.worker.update_alert.connect(self.update_alert)
        self.worker.update_positions.connect(self.update_positions)
        self.worker_thread.started.connect(self.worker.run)

        self.worker_thread.start()

    def initUI(self):
        self.setWindowTitle('Drone Distance Display')
        self.setGeometry(100, 100, 1200, 800)
        self.layout = QtWidgets.QGridLayout()

        self.drone1_label_1 = QtWidgets.QLabel("Drone 1 Distance 1: N/A")
        self.drone1_label_2 = QtWidgets.QLabel("Drone 1 Distance 2: N/A")
        self.drone1_label_3 = QtWidgets.QLabel("Drone 1 Distance 3: N/A")
        self.drone1_label_4 = QtWidgets.QLabel("Drone 1 Distance 4: N/A")
        self.drone1_label_5 = QtWidgets.QLabel("Drone 1 Distance 5: N/A")
        self.drone1_label_6 = QtWidgets.QLabel("Drone 1 Distance 6: N/A")
        self.drone1_label_7 = QtWidgets.QLabel("Drone 1 Distance 7: N/A")
        self.drone1_label_8 = QtWidgets.QLabel("Drone 1 Distance 8: N/A")
        
        self.drone2_label_1 = QtWidgets.QLabel("Drone 2 Distance 1: N/A")
        self.drone2_label_2 = QtWidgets.QLabel("Drone 2 Distance 2: N/A")
        self.drone2_label_3 = QtWidgets.QLabel("Drone 2 Distance 3: N/A")
        self.drone2_label_4 = QtWidgets.QLabel("Drone 2 Distance 4: N/A")
        self.drone2_label_5 = QtWidgets.QLabel("Drone 2 Distance 5: N/A")
        self.drone2_label_6 = QtWidgets.QLabel("Drone 2 Distance 6: N/A")
        self.drone2_label_7 = QtWidgets.QLabel("Drone 2 Distance 7: N/A")
        self.drone2_label_8 = QtWidgets.QLabel("Drone 2 Distance 8: N/A")

        self.left_image_label = QtWidgets.QLabel()
        self.right_image_label = QtWidgets.QLabel()
        self.depth_image_label = QtWidgets.QLabel()

        self.left_image_title = QtWidgets.QLabel("Left Camera YOLO Detection")
        self.right_image_title = QtWidgets.QLabel("Right Camera YOLO Detection")
        self.depth_image_title = QtWidgets.QLabel("Depth Map")

        self.alert_label = QtWidgets.QLabel("Alert: None")

        # Creating a vertical box layout for the drone distance labels
        self.drone1_layout = QtWidgets.QVBoxLayout()
        self.drone1_layout.addWidget(self.drone1_label_1)
        self.drone1_layout.addWidget(self.drone1_label_2)
        self.drone1_layout.addWidget(self.drone1_label_3)
        self.drone1_layout.addWidget(self.drone1_label_4)
        self.drone1_layout.addWidget(self.drone1_label_5)
        self.drone1_layout.addWidget(self.drone1_label_6)
        self.drone1_layout.addWidget(self.drone1_label_7)
        self.drone1_layout.addWidget(self.drone1_label_8)

        self.drone2_layout = QtWidgets.QVBoxLayout()
        self.drone2_layout.addWidget(self.drone2_label_1)
        self.drone2_layout.addWidget(self.drone2_label_2)
        self.drone2_layout.addWidget(self.drone2_label_3)
        self.drone2_layout.addWidget(self.drone2_label_4)
        self.drone2_layout.addWidget(self.drone2_label_5)
        self.drone2_layout.addWidget(self.drone2_label_6)
        self.drone2_layout.addWidget(self.drone2_label_7)
        self.drone2_layout.addWidget(self.drone2_label_8)

        # Adding all the components to the grid layout
        self.layout.addWidget(self.left_image_title, 0, 0)
        self.layout.addWidget(self.depth_image_title, 0, 1)
        self.layout.addWidget(self.left_image_label, 1, 0)
        self.layout.addWidget(self.depth_image_label, 1, 1)
        self.layout.addWidget(self.right_image_title, 2, 0)
        self.layout.addWidget(self.right_image_label, 3, 0)
        self.layout.addLayout(self.drone1_layout, 2, 1, 8, 1)  
        self.layout.addLayout(self.drone2_layout, 10, 1, 8, 1)
        self.layout.addWidget(self.alert_label, 18, 1)

        self.setLayout(self.layout)

    def update_distances(self, distances):
        self.drone1_label_1.setText(f"Drone 1 Distance: {distances['Drone 1']:.2f} cm")
        self.drone2_label_1.setText(f"Drone 2 Distance: {distances['Drone 2']:.2f} cm")

    def update_alert(self, alert_message):
        self.alert_label.setText(f"Alert: {alert_message}")

    def update_images(self, left_img, right_img, depth_map):
        left_img_qt = self.convert_cv_qt(left_img)
        right_img_qt = self.convert_cv_qt(right_img)
        depth_map_qt = self.convert_cv_qt(depth_map)

        self.left_image_label.setPixmap(left_img_qt)
        self.right_image_label.setPixmap(right_img_qt)
        self.depth_image_label.setPixmap(depth_map_qt)

    def update_positions(self, positions_history_str):
        distances = positions_history_str.split('\n')
        labels_drone1 = [
            self.drone1_label_1, self.drone1_label_2, self.drone1_label_3,
            self.drone1_label_4, self.drone1_label_5, self.drone1_label_6,
            self.drone1_label_7, self.drone1_label_8
        ]
        labels_drone2 = [
            self.drone2_label_1, self.drone2_label_2, self.drone2_label_3,
            self.drone2_label_4, self.drone2_label_5, self.drone2_label_6,
            self.drone2_label_7, self.drone2_label_8
        ]

        for i in range(len(labels_drone1)):
            if i < len(distances):
                labels_drone1[i].setText(distances[i])
            else:
                labels_drone1[i].setText("N/A")

        for i in range(len(labels_drone2)):
            if i + len(labels_drone1) < len(distances):
                labels_drone2[i].setText(distances[i + len(labels_drone1)])
            else:
                labels_drone2[i].setText("N/A")

    def convert_cv_qt(self, cv_img):
        rgb_image = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        convert_to_Qt_format = QtGui.QImage(rgb_image.data, w, h, bytes_per_line, QtGui.QImage.Format_RGB888)
        p = convert_to_Qt_format.scaled(600, 400, QtCore.Qt.KeepAspectRatio)
        return QtGui.QPixmap.fromImage(p)

class DroneWorker(QtCore.QObject):
    update_data = QtCore.pyqtSignal(dict)
    update_images = QtCore.pyqtSignal(np.ndarray, np.ndarray, np.ndarray)
    update_alert = QtCore.pyqtSignal(str)
    update_positions = QtCore.pyqtSignal(str)

    def run(self):
        move_drones_and_capture_depth(self.update_data, self.update_images, self.update_alert, self.update_positions)

if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    ex = DroneDistanceApp()
    ex.show()
    sys.exit(app.exec_())

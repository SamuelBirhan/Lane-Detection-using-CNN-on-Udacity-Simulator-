import socketio
import eventlet
from flask import Flask
import base64
from io import BytesIO
from PIL import Image
import cv2
import numpy as np
import tensorflow as tf
import argparse
from utils import *
import time

# Create a SocketIO server
sio = socketio.Server()
app = Flask(__name__)

# Define the maximum speed
max_speed = 30

# Flag to track if telemetry data is being received
telemetry_received = False

# Flag to track if the simulator is running
simulator_running = True

# Time to wait (in seconds) before considering the simulator stopped
timeout_duration = 5  # Adjust this as needed

# Last time telemetry was received
last_telemetry_time = None

# Preprocess the image
def preprocess_image(image):
    # Crop the image
    image = image[60:135, :, :]
    # Convert the image to YUV color space
    image = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
    # Apply Gaussian blur
    image = cv2.GaussianBlur(image, (3, 3), 0)
    # Resize the image to (200, 66)
    image = cv2.resize(image, (200, 66))
    # Normalize the image
    image = image / 255.0
    return image

# Event handler for telemetry data
@sio.on('telemetry')
def telemetry(sid, data):
    global telemetry_received, last_telemetry_time, model
    if data:
        telemetry_received = True
        last_telemetry_time = time.time()  # Update last telemetry time
        # Extract speed and image data
        speed = float(data['speed'])
        img_data = data['image']

        # Decode and preprocess the image
        image = Image.open(BytesIO(base64.b64decode(img_data)))
        image_array = np.asarray(image)
        preprocessed_image = preprocess_image(image_array)
        preprocessed_image = np.array([preprocessed_image])

        # Predict the steering angle using the model
        steering_angle = float(model.predict(preprocessed_image))

        # Calculate throttle based on the speed
        throttle = 1.0 - speed / max_speed

        # Log steering angle, throttle, and speed
        print(f'Steering Angle: {steering_angle}, Throttle: {throttle}, Speed: {speed}')

        # Send control commands
        send_control(steering_angle, throttle)
    else:
        # Handle missing data
        telemetry_received = False
        sio.emit('manual', data={}, skip_sid=True)

# Event handler for client connection
@sio.on('connect')
def connect(sid, environ):
    print(f'Connected to client: {sid}')
    # Initialize with zero steering angle and throttle
    send_control(0, 0)

# Function to send control commands to the simulator
def send_control(steering_angle, throttle):
    sio.emit(
        'steer',
        data={
            'steering_angle': str(steering_angle),
            'throttle': str(throttle)
        }
    )

# Function to check telemetry status and stop if not received
def check_telemetry_status():
    global simulator_running
    while simulator_running:
        if telemetry_received and last_telemetry_time is not None:
            current_time = time.time()
            if current_time - last_telemetry_time > timeout_duration:
                print("Warning: Telemetry data timeout. Stopping the script.")
                eventlet.kill(eventlet.greenthread.getcurrent())
                break
        eventlet.sleep(1)  # Check every 1 second

# Function to perform countdown before exiting
def countdown_before_exit(countdown_seconds):
    print(f"Exiting the program in {countdown_seconds} seconds...")
    for i in range(countdown_seconds, 0, -1):
        print(f"{i}...")
        time.sleep(1)
    print("Exiting the program.")

if __name__ == '__main__':
    # Default model filename
    default_model_filename = 'ModelConfigration2.keras'

    # Ask user to input the model filename, or use default
    model_filename = input(f"Enter the model filename (default is '{default_model_filename}', press Enter to use default): ")
    if model_filename.strip() == '':
        model_filename = default_model_filename

    # Load the specified model
    model = tf.keras.models.load_model(model_filename)

    # Wrap the Flask app with SocketIO middleware
    app = socketio.Middleware(sio, app)

    # Run the server on port 4567
    server = eventlet.spawn(eventlet.wsgi.server, eventlet.listen(('', 4567)), app)

    # Start checking telemetry status in a separate thread
    telemetry_checker = eventlet.spawn(check_telemetry_status)

    try:
        # Wait for the server to complete
        server.wait()
    except KeyboardInterrupt:
        print("User interrupted the program.")
    finally:
        # Set simulator_running to False to stop telemetry check loop
        simulator_running = False
        # Wait for telemetry checker to complete
        telemetry_checker.wait()
        # Perform any cleanup operations here
        countdown_before_exit(10)  # Wait for 10 seconds before exiting

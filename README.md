# Lane-Detection-using-CNN-on-Udacity-Simulator-
Lane Detection using CNN on Udacity Simulator, Self Driving car
Self Driving Car
This project involves training a neural network model to simulate self-driving car behavior using a simulation environment and TensorFlow/Keras.

Setup Instructions
1. Setting Up Virtual Environment
It is recommended to use a virtual environment to manage dependencies for this project. Follow these steps to create and activate a virtual environment:

Using venv (built-in to Python)

	# Create a new virtual environment
	python -m venv self-driving-env

	# Activate the virtual environment on Windows
	self-driving-env\Scripts\activate

	# Activate the virtual environment on macOS/Linux
	source self-driving-env/bin/activate
	
2. Installing Dependencies
Install the required Python packages using pip:

	pip install -r requirements.txt
	
	The requirements.txt file should contain the following dependencies:

		plaintext
		Copy code
		pandas==1.3.5
		numpy==1.21.6
		matplotlib==3.5.1
		scikit-learn==1.0.2
		imgaug==0.4.0
		opencv-python-headless==4.5.5.64
		tensorflow==2.8.0
		keras==2.8.0
		eventlet==0.33.0
		Flask==2.0.2
		socketio==0.2.1
		Pillow==8.4.0
		
3. Running the Self-Driving Car Simulation
Structure of the Code
The project is organized into three parts:

	Utils: Contains utility functions for data import, preprocessing, augmentation, and model creation.
	Training: Script for training the neural network model using TensorFlow/Keras.
	Testing: Script for running the self-driving car simulation using a trained model.
		
	3.1 Utils (utils.py)
		This module includes functions for:

		importDataInfo(path): Importing and preprocessing driving log data.
		balanceData(data, display): Balancing the data distribution of steering angles.
		loadData(path, data): Loading images and corresponding steering angles.
		augmentImages(imgPath, steering): Augmenting images for better training data variety.
		preProcessing(img): Preprocessing images by cropping, resizing, and normalizing.
		
	3.2 Training (train.py)
		This script trains a convolutional neural network (CNN) model using TensorFlow/Keras:

		Loads and preprocesses driving data.
		Balances data distribution.
		Creates and trains a CNN model for predicting steering angles.
		Saves the trained model and training history.
		Evaluates the model performance using loss and mean absolute error (MAE) metrics.
		Generates plots of training/validation loss and MAE.
		To train the model, run the following command:

			python train.py
			
			When prompted, provide the following inputs:

				Model Name: Enter a name for the model to be saved (e.g., Model1).
				Number of Epochs: Enter the number of training epochs (default is 100).
			SampleData containing the images and csv file is provided to check and train the model
	3.3 Testing (test.py)
		This script sets up a SocketIO server to simulate the self-driving car:
		the udacity self-driving-car-simsimulator can be downloaded here 
		https://github.com/udacity/self-driving-car-sim?tab=readme-ov-fileed
		and data can be collected by setting the simulator to training mode and to test the trained model
		use autonomous mode.
		Loads a pre-trained CNN model.
		Integrates with a simulator through SocketIO for telemetry data exchange.
		Preprocesses incoming images from the simulator.
		Predicts steering angles and computes throttle based on speed.
		Controls the car in the simulator by sending steering and throttle commands.
		Monitors telemetry data and handles disconnections or timeouts gracefully.
		To run the simulation, execute the following command:

			python test.py
			
			When prompted, provide the following input:
			
				Model Name: Enter a name of the saved model (e.g., ModelConfigration2.keras).
		
4. Additional Notes
Adjust parameters, such as epochs for training, in train.py as needed.
Ensure the simulator is running and configured to connect to the server.

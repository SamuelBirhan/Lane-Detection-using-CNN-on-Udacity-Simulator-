import pickle
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping
from utils import *  # Import utility functions

print('Setting UP')
print("Available devices:", tf.config.list_physical_devices())
if tf.config.list_physical_devices('GPU'):
    for gpu in tf.config.experimental.list_physical_devices('GPU'):
        tf.config.experimental.set_memory_growth(gpu, True)
    print("GPU is available and memory growth is set")
else:
    print("No GPU available")

# Step 1: Import the data
path = 'SampleData'
data = importDataInfo(path)

# imgRe = preProcessing(mpimg.imread('test1.jpg'))
# plt.imshow(imgRe)
# plt.show()

# Step 2: Balance data
balanceData(data, display=True)

# Step 3: Load data
imagesPath, steering = loadData(path, data)
print(imagesPath[0], steering[0])

# Step 4: Split the data into training and validation sets
xtrain, xval, ytrain, yval = train_test_split(imagesPath, steering, test_size=0.2, random_state=5, shuffle=True)
print('Total training images:', len(xtrain))
print('Total validation images:', len(xval))

# Step 8: Create the model
model = creatModel()
model.summary()

# Ask user for model name to be saved
model_name = input("Enter the model name to save (e.g., Model1): ")

# Ask user for number of training epochs, default is 100
num_epochs = input(f"Enter the number of training epochs (default is 100): ")
if num_epochs == '':
    num_epochs = 100
else:
    num_epochs = int(num_epochs)

# Step 9: Define EarlyStopping callback
early_stopping = EarlyStopping(
    monitor='val_loss',  # Monitor the validation loss
    mode='min',  # We're minimizing the loss
    patience=5,  # Stop if no improvement for 5 epochs
    verbose=1  # Print information when stopping
)

# Step 10: Train the model with EarlyStopping callback
history = model.fit(batchGen(xtrain, ytrain, batchSize=100, trainFlag=1),
                    steps_per_epoch= 300,
                    epochs=num_epochs,
                    validation_data=batchGen(xval, yval, batchSize=100, trainFlag=0),
                    validation_steps = 200,
                    callbacks=[early_stopping]
                    )

# Step 11: Save the model
model.save(f'{model_name}.keras')
print(f'Model saved as {model_name}.keras')

# Step 12: Save training history
history_dict = history.history

# Save the training history including loss and metrics
history_filename = f'training_history_{model_name}.pkl'
with open(history_filename, 'wb') as f:
    pickle.dump(history_dict, f)
print(f'Training history saved as {history_filename}')

# Step 13: Calculate additional metrics
training_loss = history_dict['loss']
validation_loss = history_dict['val_loss']

# Calculate Mean Absolute Error (MAE) for training and validation
training_mae = history_dict['mae']
validation_mae = history_dict['val_mae']

# Plotting
# Calculate mean and standard deviation of validation loss
mean_val_loss = np.mean(validation_loss)
std_val_loss = np.std(validation_loss)

# Plot training and validation loss
plt.plot(training_loss, label='Training Loss')
plt.plot(validation_loss, label='Validation Loss')
plt.axhline(mean_val_loss, color='red', linestyle='--', label='Mean Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.savefig(f'loss_curve_{model_name}.png')  # Save the figure
plt.show()

# Plot training and validation MAE
plt.plot(training_mae, label='Training MAE')
plt.plot(validation_mae, label='Validation MAE')
plt.title('Training and Validation MAE')
plt.xlabel('Epoch')
plt.ylabel('Mean Absolute Error')
plt.legend()
plt.grid(True)
plt.savefig(f'mae_curve_{model_name}.png')  # Save the figure
plt.show()

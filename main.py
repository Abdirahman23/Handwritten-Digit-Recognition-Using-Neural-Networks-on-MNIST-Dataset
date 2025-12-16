# Import necessary libraries
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt

from sklearn.metrics import classification_report
import numpy as np



# Load MNIST dataset
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

# Preprocessing: Flatten the images and normalize pixel values
x_train = x_train.reshape(-1, 784).astype("float32") / 255.0
x_test = x_test.reshape(-1, 784).astype("float32") / 255.0


#MODEL A


# ONLY 1 hidden layer with 40 neurons
model_a = keras.Sequential([
    layers.Input(shape=(784,)),              # Input layer: 784 neurons
    layers.Dense(40, activation='relu'),     # Hidden layer: 40 neurons, ReLU activation
    layers.Dense(10, activation='softmax')   # Output layer: 10 neurons, Softmax activation
])

# Compile the model
model_a.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Train the model
history_a = model_a.fit(
    x_train, y_train,
    epochs=10,
    batch_size=32,
    validation_data=(x_test, y_test)
)

# Evaluate the model
test_loss_a, test_accuracy_a = model_a.evaluate(x_test, y_test)
print("Test Accuracy for Part (a):", test_accuracy_a)

# Plot accuracy vs epochs
plt.plot(history_a.history['accuracy'], label='Training Accuracy')
plt.plot(history_a.history['val_accuracy'], label='Test Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Accuracy vs Epoch (Part a)')
plt.legend()
plt.show()

# Save the model
model_a.save('/content/model_a.h5') 
print("Model for Part (a) saved.")


# Get model summary for parameters
model_a.summary()


#MODEL B

#Lets make 2 hidden layers, each with 20 neurons
model_b = keras.Sequential([
    layers.Input(shape=(784,)),              # Input layer: 784 neurons
    layers.Dense(20, activation='relu'),     # Hidden layer 1: 20 neurons, ReLU activation
    layers.Dense(20, activation='relu'),     # Hidden layer 2: 20 neurons, ReLU activation
    layers.Dense(10, activation='softmax')   # Output layer: 10 neurons, Softmax activation
])

# Compile 
model_b.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Train 
history_b = model_b.fit(
    x_train, y_train,
    epochs=10,
    batch_size=32,
    validation_data=(x_test, y_test)
)

# Evaluate the model
test_loss_b, test_accuracy_b = model_b.evaluate(x_test, y_test)
print("Test Accuracy for Part (b):", test_accuracy_b)

# Plot accuracy vs epochs
plt.plot(history_b.history['accuracy'], label='Training Accuracy')
plt.plot(history_b.history['val_accuracy'], label='Test Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Accuracy vs Epoch (Part b)')
plt.legend()
plt.show()

# Get model summary for parameters
model_b.summary()

# Save the model
model_b.save('/content/model_b.h5')  
print("Model for Part (b) saved.")


#MODEL C

# Lets make 1 hidden layer with 50 neurons
model_c = keras.Sequential([
    layers.Input(shape=(784,)),              # Input layer: 784 neurons
    layers.Dense(50, activation='relu'),     # Hidden layer: 50 neurons, ReLU activation
    layers.Dense(10, activation='softmax')   # Output layer: 10 neurons, Softmax activation
])

# Compile the model
model_c.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Train the model
history_c = model_c.fit(
    x_train, y_train,
    epochs=10,
    batch_size=32,
    validation_data=(x_test, y_test)
)

# Evaluate the model
test_loss_c, test_accuracy_c = model_c.evaluate(x_test, y_test)
print("Test Accuracy for Part (c):", test_accuracy_c)

# Plot accuracy vs epochs
plt.plot(history_c.history['accuracy'], label='Training Accuracy')
plt.plot(history_c.history['val_accuracy'], label='Test Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Accuracy vs Epoch (Part c)')
plt.legend()
plt.show()

# Get model summary for parameters
model_c.summary()


# Save the model
model_c.save('/content/model_c.h5')  # Save the model for Part (c)
print("Model for Part (c) saved.")


#MODEL D

#LETS Change activation function to 'sigmoid'
model_d = keras.Sequential([
    layers.Input(shape=(784,)),              # Input layer: 784 neurons
    layers.Dense(50, activation='sigmoid'),  # Hidden layer: 50 neurons, Sigmoid activation
    layers.Dense(10, activation='softmax')   # Output layer: 10 neurons, Softmax activation
])

# Compile 
model_d.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Train 
history_d = model_d.fit(
    x_train, y_train,
    epochs=10,
    batch_size=32,
    validation_data=(x_test, y_test)
)

# TEST the model
test_loss_d, test_accuracy_d = model_d.evaluate(x_test, y_test)
print("Test Accuracy for Part (d):", test_accuracy_d)

# Plot accuracy vs epochs
plt.plot(history_d.history['accuracy'], label='Training Accuracy')
plt.plot(history_d.history['val_accuracy'], label='Test Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Accuracy vs Epoch (Part d)')
plt.legend()
plt.show()

# Get model summary for parameters
model_d.summary()

# Save the model
model_d.save('/content/model_d.h5') 
print("Model for Part (d) saved.")

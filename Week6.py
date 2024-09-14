# Import necessary libraries
import tensorflow as tf
from tensorflow.keras import layers, models

# Load the MNIST dataset
mnist = tf.keras.datasets.mnist
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

# Preprocess the data: Normalize the images to values between 0 and 1
train_images = train_images / 255.0
test_images = test_images / 255.0

# Build the neural network model
model = models.Sequential()

# Input Layer: Flatten the 28x28 images into 784-dimensional vectors
model.add(layers.Flatten(input_shape=(28, 28)))

# Hidden Layer: Dense layer with 128 neurons and ReLU activation function
model.add(layers.Dense(128, activation='relu'))

# Output Layer: Dense layer with 10 neurons (one for each digit) and softmax activation function
model.add(layers.Dense(10, activation='softmax'))

# Compile the model: Use Adam optimizer, sparse categorical crossentropy loss, and accuracy metric
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Train the model
model.fit(train_images, train_labels, epochs=5)

# Evaluate the model on the test data
test_loss, test_acc = model.evaluate(test_images, test_labels)

# Print the test accuracy
print(f"Test accuracy: {test_acc}")


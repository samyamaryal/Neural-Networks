import tensorflow as tf
from tensorflow import keras
from matplotlib import pyplot as plt
import numpy as np

# LOAD THE DATASET
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()

# Looking at a random data & the associated label
n = 300
plt.imshow(x_train[n])
print(y_train[n])

# The output is labeled from 0 to 9
# This is a list of the outputs
clothes = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal',	'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),
    keras.layers.Dense(units=100, activation='swish'),
    keras.layers.Dense(units=200, activation='swish'),
    keras.layers.Dense(units=200, activation='swish'),
    keras.layers.Dense(units=100, activation='swish'),
    keras.layers.Dense(units=10, activation='softmax')
])

print(model.summary)

# Scale the data between 0 & 1.
train_data = x_train / 255
train_label = y_train / 255

# Train the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics='accuracy')
history = model.fit(x_train, y_train, epochs=20, verbose=1)

# Plot the training loss and accuracy
plt.plot(history.history['loss'])
plt.plot(history.history['accuracy'])

test_loss, test_acc = model.evaluate(x_test, y_test)
# Find the test accuracy
print(test_acc)

# Making predictions on the test data
pred = model.predict(x_test)

# Print the prediction & actual output on the nth data.
print(clothes[np.argmax(pred[n])])
plt.imshow(x_test[n])


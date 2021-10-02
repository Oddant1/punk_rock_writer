import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

mnist = tf.keras.datasets.mnist # 28 x 28 images of hand written digits 0 - 9

# Get the data and split into manageable chunks
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Normalize the data, axis=1 means take the inverse of the value (1/value)
# This step is not required but it improves performance
x_train = tf.keras.utils.normalize(x_train, axis=1)
x_test = tf.keras.utils.normalize(x_test, axis=1)

# Create the tensorflow model
model = tf.keras.models.Sequential()
# Add a layer that flattens our 28 by 28 array
model.add(tf.keras.layers.Flatten())
# Layer of 128 neurons firing based on a rectified linear function
model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu, input_shape= x_train.shape[1:]))
model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))
# Output layer with number of neurons equal to possible classificaitions using
# Softmax for activation
model.add(tf.keras.layers.Dense(10, activation=tf.nn.softmax))

# Build the model
model.compile(optimizer='adam', # Optimzer determines how network will attempt to minimize loss
              loss='sparse_categorical_crossentropy', # How network defines loss
              metrics=['accuracy']) # What the network will track for us

# Give the model test data so it can attempt to learn classifications
model.fit(x_train, y_train, epochs=3)

# Determine if the model overfitted
val_loss, val_acc = model.evaluate(x_test, y_test)
print(val_loss, val_acc )

# Make predictions with the model
predictions = model.predict([x_test])
print(np.argmax(predictions[0]))

# Show the image corresponding to the printed predictions
plt.imshow(x_test[0])
plt.show()
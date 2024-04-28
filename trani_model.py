import tensorflow as tf
import numpy as np

train = np.genfromtxt('train.csv', delimiter=',')
result = np.genfromtxt('train.csv', delimiter=',')

# Data to learn on it
X_train = train[1:, 1:]
y_train = result[1:, 1:]

model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(128, activation='relu', input_shape=(11,)),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

# Compile the model
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=10, batch_size=32)



import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder


X_train =  pd.read_csv('train.csv')
y_train = X_train.pop('Survived')

label_encoder = LabelEncoder()

X_train['Name'] = X_train['Name'].apply(lambda x:  label_encoder.fit_transform(x))
X_train['Sex'] = X_train['Sex'].apply(lambda x:  label_encoder.fit_transform(x))
X_train['Ticket'] = X_train['Ticket'].apply(lambda x:  label_encoder.fit_transform(x))
X_train['Embarked'] = X_train['Embarked'].apply(lambda x:  label_encoder.fit_transform(x))


print(y_train)

model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(128, activation='relu', input_shape=(13,)),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

# Compile the model
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=10, batch_size=32)



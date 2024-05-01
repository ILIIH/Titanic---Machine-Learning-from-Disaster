import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder


X_train =  pd.read_csv('train.csv')
y_train = X_train.pop('Survived')
X_train.pop('Ticket')
X_train.pop('Cabin')

mean_age = X_train['Age'].mean()

label_encoder = LabelEncoder()
X_train['Name'] = label_encoder.fit_transform(X_train['Name'].astype(str))
X_train['Sex'] = label_encoder.fit_transform(X_train['Sex'])
X_train['Embarked'] = label_encoder.fit_transform(X_train['Embarked'])

X_train['Age'] = X_train['Age'].apply(lambda x: mean_age if pd.isnull(x) else x)

X_train['Embarked'] = label_encoder.fit_transform(X_train['Embarked'].astype(str))

model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(64, activation='relu', input_shape=(9,)),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

# Compile the model
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=10, batch_size=32)



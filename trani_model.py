import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder


def prepareData(data):
    data.pop('Ticket')
    if 'Survived' in data:
        data.pop('Survived')
    data.pop('Cabin')

    mean_age = data['Age'].mean()

    label_encoder = LabelEncoder()
    data['Name'] = label_encoder.fit_transform(data['Name'].astype(str))
    data['Sex'] = label_encoder.fit_transform(data['Sex'])
    data['Embarked'] = label_encoder.fit_transform(data['Embarked'])

    data['Age'] = data['Age'].apply(lambda x: mean_age if pd.isnull(x) else x)

    data['Embarked'] = label_encoder.fit_transform(data['Embarked'].astype(str))
    return data 


def binary_prediction(probability):
    if probability >= 0.5:
        return 1
    else:
        return 0
    
X_train =  pd.read_csv('train.csv')
y_train = X_train.pop('Survived')
X_train = prepareData(X_train)

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

# Make predictions
test_data =  pd.read_csv('test.csv')
test_data = prepareData(test_data)
predictions = model.predict(test_data)

for i in range(len(predictions)):
    predictions[i] = binary_prediction(predictions[i]) 

# Print predictions
print(predictions)


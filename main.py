import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

def prepare_data(data):
    data.pop('Ticket')
    data.pop('Cabin')

    mean_age = data['Age'].mean()
    data['Age'].fillna(mean_age, inplace=True)

    label_encoder = LabelEncoder()
    categorical_cols = ['Name', 'Sex', 'Embarked']
    for col in categorical_cols:
        data[col] = label_encoder.fit_transform(data[col].astype(str))

    return data 

def binary_prediction(probability):
    return 1 if probability >= 0.5 else 0

def train_model(X_train, y_train):
    model = tf.keras.models.Sequential([
        tf.keras.layers.Dense(64, activation='relu', input_shape=(9,)),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])

    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

    model.fit(X_train, y_train, epochs=10, batch_size=32)
    return model

def make_predictions(model, test_data):
    test_data = prepare_data(test_data)
    predictions = model.predict(test_data)
    predictions = [binary_prediction(p) for p in predictions.flatten()]
    df = pd.DataFrame()

    df['PassengerId'] = test_data.pop('PassengerId')
    df['Survived'] = predictions

    return df

def main():
    X_train = pd.read_csv('train.csv')
    y_train = X_train.pop('Survived')
    X_train = prepare_data(X_train)

    model = train_model(X_train, y_train)

    test_data = pd.read_csv('test.csv')
    predictions = make_predictions(model, test_data)

    predictions.to_csv('predictions.csv', index=False)  
    print(predictions)

if __name__ == "__main__":
    main()

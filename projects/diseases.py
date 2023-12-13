import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, mean_squared_error

data = pd.read_csv('tmp/cardio_train.csv')

# print(data.columns)
features = ["age", "height", "weight", "ap_hi", "ap_lo", "cholesterol", "gluc", "smoke", "alco", "active"]
x = data[features]
y = data["cardio"]
# y_pred = np.array([[9405, 173, 82, 132, 68, 1, 2, 0, 0, 1]])

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

model = tf.keras.Sequential([
        tf.keras.layers.Dense(64, activation='relu', input_shape=(x_train.shape[1],)),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
        ])

model.summary()
model.compile(optimizer='adam', loss=tf.keras.losses.BinaryCrossentropy(), metrics=['accuracy'])

history = model.fit(x_train, y_train, epochs=50, batch_size=32, validation_split=0.2)

y_pred_probs = model.predict(x_test)
y_pred = (y_pred_probs > 0.5).astype(int)
print(y_pred_probs)
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("\nMean squared error:\n", mean_squared_error(y_test, y_pred))

predict_res = model.evaluate(x_test, y_test, verbose=2)
print("\nEvaluate:\n   Test Loss: ", predict_res[0])
print("\n    Test Accuracy: ", predict_res[1])


def get_user_inputs():
    age = float(input("Enter age in days: "))
    height = float(input("Enter height (in cm): "))
    weight = float(input("Enter weight (in kg): "))
    ap_hi = float(input("Enter systolic blood pressure (ap_hi): "))
    ap_lo = float(input("Enter diastolic blood pressure (ap_lo): "))
    cholesterol = int(input("Enter cholesterol level (1, 2, or 3): "))
    gluc = int(input("Enter glucose level (1, 2, or 3): "))
    smoke = int(input("Do you smoke? (0 for No, 1 for Yes): "))
    alco = int(input("Do you consume alcohol? (0 for No, 1 for Yes): "))
    active = int(input("Are you physically active? (0 for No, 1 for Yes): "))

    user_inputs = np.array([age, height, weight, ap_hi, ap_lo, cholesterol, gluc, smoke, alco, active], dtype=float)
    user_inputs = scaler.transform(user_inputs.reshape(1, -1))
    return user_inputs

# user_inputs = get_user_inputs()
# user_prediction_prob = model.predict(user_inputs)
# print("\n Probability of Cardiovascular Disease:", user_prediction_prob[0][0])
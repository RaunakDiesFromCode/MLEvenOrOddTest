import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam


# Function to generate data
def generate_data(n_samples=1000):
    X = np.random.randint(0, 1000, n_samples)
    y = X % 2
    return X, y


# Generate data
X, y = generate_data()

# Define the model
model = Sequential(
    [
        Dense(64, input_dim=1, activation="relu"),
        Dense(32, activation="relu"),
        Dense(1, activation="sigmoid"),
    ]
)

model.compile(optimizer=Adam(), loss="binary_crossentropy", metrics=["accuracy"])

# Train the model
model.fit(X, y, epochs=10, batch_size=32)


# Function to predict even or odd
def predict_even_odd(number):
    prediction = model.predict(np.array([number]))
    if prediction < 0.5:
        return "Even"
    else:
        return "Odd"


# Test the model
while True:
  test_number = input("Enter a number to classify as Even or Odd (or type 'exit' to quit): ")
  if test_number == 'exit':
    break
  result = predict_even_odd(int(test_number))
  print(f"The number {test_number} is {result}.")

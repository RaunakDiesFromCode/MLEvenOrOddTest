# Import required libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report


totRange = 101

# Step 1: Generate the dataset
def create_dataset():
    data = {
        "Number": range(1, totRange),  # Numbers from 1 to 1000
        "Label": ["Even" if num % 2 == 0 else "Odd" for num in range(1, totRange)],
    }
    df = pd.DataFrame(data)

    # Add a derived feature for modulo-2
    df["Modulo2"] = df["Number"] % 2
    return df


# Step 2: Prepare the data
def prepare_data(df):
    X = df[["Number", "Modulo2"]]  # Features: Number and Modulo2
    y = df["Label"]  # Target: "Even" or "Odd"

    # Encode labels
    encoder = LabelEncoder()
    y_encoded = encoder.fit_transform(y)

    # Shuffle and split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2, random_state=42, shuffle=True
    )
    return X_train, X_test, y_train, y_test, encoder

# Step 3: Train the model
def train_model(X_train, y_train):
    model = DecisionTreeClassifier()
    model.fit(X_train, y_train)
    return model


# Step 4: Evaluate the model
def evaluate_model(model, X_test, y_test, encoder):
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, target_names=encoder.classes_)
    print("Model Evaluation:")
    print(f"Accuracy: {accuracy * 100:.2f}%")
    print("Classification Report:\n", report)


# Step 5: Interactive Prediction in Terminal
def interactive_prediction(model, encoder):
    print("\nEnter a number to classify as Even or Odd (type 'exit' to quit):")
    while True:
        user_input = input("Enter number: ")
        if user_input.lower() == "exit":
            break
        try:
            number = int(user_input)
            # Calculate Modulo2 for the input
            modulo2 = number % 2
            input_data = pd.DataFrame({"Number": [number], "Modulo2": [modulo2]})
            prediction = model.predict(input_data)[0]
            label = encoder.inverse_transform([prediction])[0]
            print(f"The number {number} is classified as: {label}")
        except ValueError:
            print("Please enter a valid integer or 'exit' to quit.")
        except Exception as e:
            print(f"An error occurred: {e}")


# Main function to run the steps
def main():
    # 1. Create and prepare dataset
    df = create_dataset()
    X_train, X_test, y_train, y_test, encoder = prepare_data(df)

    # 2. Train the model
    model = train_model(X_train, y_train)

    # 3. Evaluate the model
    evaluate_model(model, X_test, y_test, encoder)

    # 4. Run interactive prediction
    interactive_prediction(model, encoder)


# Run the main function if the script is executed directly
if __name__ == "__main__":
    main()


import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

# Load the data
filepath = "/Users/cullenbaker/school/comps/bluetooth-tracking-ultimate/merged_player_positions.csv"
data = pd.read_csv(filepath)

# Define the target and input columns
target_columns = ['gt_x', 'gt_y']
input_columns = [col for col in data.columns if col.startswith('x') or col.startswith('y')]

# Split the data into inputs and targets
X = data[input_columns]
y = data[target_columns]

# # Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# # Train the random forest model
# model = RandomForestRegressor(n_estimators=100, random_state=42)
# model.fit(X_train, y_train)

# # Make predictions
# y_pred = model.predict(X_test)

# # Evaluate the model
# mse = mean_squared_error(y_test, y_pred)
# print(f"Mean Squared Error: {mse}")

# # Save the model

# model_filepath = "/Users/cullenbaker/school/comps/bluetooth-tracking-ultimate/random_forest_model.pkl"
# joblib.dump(model, model_filepath)
# print(f"Model saved to {model_filepath}")


def predict_with_model(input_data):
    # Load the model
    model_filepath = "/Users/cullenbaker/school/comps/bluetooth-tracking-ultimate/random_forest_model.pkl"
    model = joblib.load(model_filepath)
    
    # Ensure input_data is a DataFrame
    if not isinstance(input_data, pd.DataFrame):
        input_data = pd.DataFrame(input_data)
    
    # Make predictions
    predictions = model.predict(input_data)
    
    return predictions

# Example usage
example_input = X_test.head(1)  # Replace with your actual input data
print(example_input)
predictions = predict_with_model(example_input)
print(f"Predictions: {predictions[0][0]}")

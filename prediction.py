import joblib
import pandas as pd

# Function to check if a string can be converted to a float
def is_float(value):
    try:
        float(value)
        return True
    except ValueError:
        return False

# Get user input for each feature
concave_points_mean = input("Input concave points mean: ")
area_se = input("Input area se: ")
smoothness_se = input("Input smoothness se: ")
symmetry_se = input("Input symmetry se: ")
radius_worst = input("Input radius worst: ")
texture_worst = input("Input texture worst: ")
area_worst = input("Input area worst: ")
smoothness_worst = input("Input smoothness worst: ")
concavity_worst = input("Input concavity worst: ")
concave_points_worst = input("Input concave points worst: ")

# List of inputs
inputs = [
    concave_points_mean, area_se, smoothness_se, symmetry_se,
    radius_worst, texture_worst, area_worst, smoothness_worst,
    concavity_worst, concave_points_worst
]

# Check if any input is null or not a valid numerical value
if None in inputs:
    print("One or more input values are null. Please provide values for all features.")
elif not all(map(is_float, inputs)):
    print("One or more input values are not valid numerical values.")
else:
    # Convert input to float
    inputs = list(map(float, inputs))

    # Create a DataFrame with the input values
    input_df = pd.DataFrame({
        "concave points_mean": [inputs[0]],
        "area_se": [inputs[1]],
        "smoothness_se": [inputs[2]],
        "symmetry_se": [inputs[3]],
        "radius_worst": [inputs[4]],
        "texture_worst": [inputs[5]],
        "area_worst": [inputs[6]],
        "smoothness_worst": [inputs[7]],
        "concavity_worst": [inputs[8]],
        "concave points_worst": [inputs[9]],
    })

    # Load the model and scaler
    model = joblib.load("cancer_prediction_model.joblib")
    scaler = joblib.load("scaler.joblib")

    # Scale the input data
    input_scaled = scaler.transform(input_df)

    # Make prediction
    prediction = model.predict(input_scaled)

    # The output of predict is an array, so we get the first (and only) element
    prediction_label = "Benign" if prediction[0] == 0 else "Malignant"

    print("The prediction is", prediction_label)
1 
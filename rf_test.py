import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer

def random_forest_impute(data, n_estimators=100, max_depth=None):
    data_nan = data.copy()

    # Create a mask to identify the location of NaN values
    nan_mask = np.isnan(data_nan)

    for col in range(data_nan.shape[1]):
        # Split the data into X and y
        X = data_nan.iloc[:, [i for i in range(data_nan.shape[1]) if i != col]].values
        y = data_nan.iloc[:, col].values

        # Impute missing values with median
        imputer = SimpleImputer(strategy="median")
        X_imputed = imputer.fit_transform(X)

        # Train the random forest regressor only on rows without NaNs in the target column
        not_nan_rows = ~np.isnan(y)
        rf = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth, random_state=0)
        rf.fit(X_imputed[not_nan_rows], y[not_nan_rows])

        # Predict missing values
        missing_rows = np.isnan(y)
        if np.any(missing_rows):
            y_missing = rf.predict(X_imputed[missing_rows])
            data_nan.loc[missing_rows, data_nan.columns[col]] = y_missing

    return data_nan

data = pd.read_csv('D:\\TS\\Group Project\\laptop_new\\expanded_data.csv')

# Select 5 valid data points and set them to NaN
num_points = 5
selected_points = set()
original_values = []

# Find and store valid data points
while len(selected_points) < num_points:
    row = np.random.randint(0, data.shape[0])
    col = np.random.randint(0, data.shape[1])

    value = data.iat[row, col]
    if not np.isnan(value) and value != 0:
        point = (row, col)
        if point not in selected_points:
            selected_points.add(point)
            original_values.append(value)

selected_points = list(selected_points)

# Set the selected points to NaN in the dataset
for row, col in selected_points:
    data.iat[row, col] = np.NaN

# Perform Random Forest imputation
imputed_data = random_forest_impute(data)

# Find the imputed values for the selected points
imputed_values = [imputed_data.iat[row, col] for row, col in selected_points]

# Calculate the absolute differences and the accuracy
diffs = np.abs(np.array(imputed_values) - np.array(original_values))
relative_diffs = diffs / np.array(original_values)
accuracy = 1 - np.linalg.norm(np.mean(relative_diffs))

print(f"Accuracy: {accuracy:.4f}")

# Calculate the loss metrics
mae = np.mean(diffs)
mse = np.mean(diffs**2)

print(f"Mean Absolute Error (MAE): {mae:.4f}")
print(f"Mean Squared Error (MSE): {mse:.4f}")

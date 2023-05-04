import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.decomposition import TruncatedSVD

def truncated_svd_impute(data, n_components=5):
    data_nan = data.copy()
    
    # Impute missing values with mean
    mean_imputer = SimpleImputer(strategy="mean")
    data_imputed = mean_imputer.fit_transform(data_nan)
    
    # Apply Truncated SVD
    svd = TruncatedSVD(n_components=n_components)
    reduced_data = svd.fit_transform(data_imputed)
    
    # Reconstruct the data
    reconstructed_data = reduced_data.dot(svd.components_)
    
    # Update only the imputed values
    nan_mask = np.isnan(data_nan)
    for row, col in zip(*np.where(nan_mask)):
        data_nan.iat[row, col] = reconstructed_data[row, col]

    return data_nan

data = pd.read_csv('D:\\TS\\Group Project\\laptop_new\\expanded_data.csv')

# Select 5 valid data points and set them to NaN
num_points = 5
selected_points = []
original_values = []

# Find and store valid data points
while len(selected_points) < num_points:
    row = np.random.randint(0, data.shape[0])
    col = np.random.randint(0, data.shape[1])

    value = data.iat[row, col]
    if not np.isnan(value) and value != 0:
        selected_points.append((row, col))
        original_values.append(value)

# Set the selected points to NaN in the dataset
for row, col in selected_points:
    data.iat[row, col] = np.NaN

# Perform Truncated SVD imputation
imputed_data = truncated_svd_impute(data)

# Find the imputed values for the selected points
imputed_values = [imputed_data.iat[row, col] for row, col in selected_points]

# Calculate the absolute differences and the accuracy
diffs = np.abs(np.array(imputed_values) - np.array(original_values))
relative_diffs = diffs / np.array(original_values)
accuracy = 1 - np.mean(relative_diffs)

print(f"Accuracy: {accuracy:.4f}")

# Calculate the loss metrics
mae = np.mean(diffs)
mse = np.mean(diffs**2)

print(f"Mean Absolute Error (MAE): {mae:.4f}")
print(f"Mean Squared Error (MSE): {mse:.4f}")

import numpy as np
import pandas as pd
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.linear_model import BayesianRidge

data = pd.read_csv('D:\\TS\\Group Project\\df_env\\df_env.csv')

# Select 5 valid data points and set them to NaN
selected_points = [(0, 2), (5, 6), (15, 8), (35, 12), (50, 4)]
original_values = []

for row, col in selected_points:
    original_values.append(data.iat[row, col])
    data.iat[row, col] = np.NaN

# Perform Bayesian Ridge Regression imputation
imputer = IterativeImputer(estimator=BayesianRidge(), max_iter=10, random_state=0)
imputed_data = imputer.fit_transform(data)
imputed_data_df = pd.DataFrame(imputed_data, columns=data.columns)

# Find the imputed values for the selected points
imputed_values = [imputed_data_df.iat[row, col] for row, col in selected_points]

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

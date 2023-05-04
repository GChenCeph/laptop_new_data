import pandas as pd
import numpy as np

# Read the processed_data.csv file
file_path = "D:\\TS\\Group Project\\current\\processed_data.csv"
data = pd.read_csv(file_path)

# Get the unique timesteps and nodeids
unique_times = data['time'].unique()
unique_nodeids = data['nodeid'].unique()

# Create a new DataFrame with all possible combinations of time and nodeid
time_nodeid_combinations = pd.DataFrame(
    np.array(np.meshgrid(unique_times, unique_nodeids)).T.reshape(-1, 2),
    columns=['time', 'nodeid']
)

# Merge the new DataFrame with the processed data, filling in missing values with NaN
expanded_data = pd.merge(
    time_nodeid_combinations,
    data,
    on=['time', 'nodeid'],
    how='left'
)

# Reorder the nodeids within each timestep
expanded_data.sort_values(['time', 'nodeid'], inplace=True)
expanded_data.reset_index(drop=True, inplace=True)

expanded_data.to_csv("expanded_data.csv", index=False)
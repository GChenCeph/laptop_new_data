import pandas as pd

# Read the data
file_path = "D:\\TS\\Group Project\\labapp3-data-new.txt"
column_names = ['time', 'nodeid', 'temperature', 'humidity', 'light', 'voltage']
data = pd.read_csv(file_path, delimiter=" ", names=column_names)

# Get the first 5000 timesteps
unique_times = data['time'].unique()
first_5000_times = unique_times[:5000]
data = data[data['time'].isin(first_5000_times)]

# Calculate spasticity for each nodeid
spasticity = data.groupby('nodeid')['time'].nunique() / len(first_5000_times)
#print("spasticity: ", spasticity)

# Select top 40 nodeids with the highest spasticity
top_40_nodeids = spasticity.nlargest(40).index
#print("top40: ", top_40_nodeids)

# Filter the data to include only the top 40 nodeids
filtered_data = data[data['nodeid'].isin(top_40_nodeids)]

filtered_data.to_csv("processed_data.csv", index=False, header=["time", "nodeid", "temperature", "humidity", "light", "voltage"], sep=",")
import numpy as np
import os

folder_path = r"/datasets/data/1"  

file_paths = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith('.npy')]

processed_data = []

for file_path in file_paths:
    data = np.load(file_path)
    data = data[:, :, 1]
    data_flattened = data.flatten()
    processed_data.append(data_flattened)
processed_data = np.array(processed_data)
print(processed_data.shape)
np.save('processed_data_1.npy', processed_data)

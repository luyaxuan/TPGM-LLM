import numpy as np
import os

# Load raw data
x = np.load('/root/autodl-tmp/new_npy/x.npy')
y = np.load('/root/autodl-tmp/new_npy/y.npy')
xt = np.load('/root/autodl-tmp/new_npy/xt.npy')

# Set time window length
time_steps = 15

# Number of available samples
num_samples = y.shape[0] - 2 * time_steps + 1

new_x = []
new_xt = []
new_y = []

# Build sliding window
for i in range(num_samples):
    window_x = x[i:i + time_steps]
    window_xt = xt[i:i + time_steps]
    window_y = y[i + time_steps:i + 2 * time_steps]

    new_x.append(window_x)
    new_xt.append(window_xt)
    new_y.append(window_y)

# Convert to numpy arrays
new_x = np.array(new_x)
new_xt = np.array(new_xt)
new_y = np.array(new_y)

# Print overall dimension information
print(f"Total number of samples: {num_samples}")
print(f"Original shape of single sample x: {new_x[0].shape}")
print(f"Original shape of single sample xt: {new_xt[0].shape}")
print(f"Original shape of single sample y: {new_y[0].shape}")


# Select the index of the sample to extract
test_idx = 0

# Extract the corresponding sample and add batch dimension (axis=0, add 1 at the front)
test_x = np.expand_dims(new_x[test_idx], axis=0)  # Shape becomes (1, 15, 1296, 4)
test_xt = np.expand_dims(new_xt[test_idx], axis=0)  # Shape becomes (1, 15, 1)
test_y = np.expand_dims(new_y[test_idx], axis=0)  # Shape becomes (1, 15, 1296, 2)

# Set save path (ensure the directory exists)
save_path = '/root/TPGM-LLM-main/new_npy/'  # Consistent with the subsequent loading path
if not os.path.exists(save_path):
    os.makedirs(save_path, exist_ok=True)

# Save data with batch dimension
np.save(f"{save_path}sensor_input.npy", test_x)  # Sensor data with batch dimension
np.save(f"{save_path}text.npy", test_xt)        # Text data with batch dimension
# Optional: Save labels with batch dimension
# np.save(f"{save_path}label.npy", test_y)

# Print the shape after saving (confirm batch dimension is added)
print(f"\nSuccessfully saved the {test_idx}-th sample with batch dimension to {save_path}")
print(f"sensor_input.npy shape: {test_x.shape} (meets (batch_size, input_len, num_nodes, input_dim))")
print(f"text.npy shape: {test_xt.shape} (meets (batch_size, input_len, input_text_dim), adjust dimension order if needed)")
print(f"test_y shape: {test_y.shape}")
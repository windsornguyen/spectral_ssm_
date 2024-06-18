import numpy as np

# Load the .npy file
data = np.load('data/Walker2D-v1/train_inputs.npy')

# Select the first 29 features
data_new = data[:, :, :18]

print(data_new.shape)

# Save the new data to a .npy file
np.save('data/Walker2D-v1/train_inputs_obs.npy', data_new)

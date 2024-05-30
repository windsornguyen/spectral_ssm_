import numpy as np

input_data = np.load('input_data.npy')  # (1000, 37)
output_data = np.load('output_data.npy')  # (1000, 29)

print("Input Data Stats:")
print("Means:", np.mean(input_data, axis=0))
print("Standard Deviations:", np.std(input_data, axis=0))
print("Min:", np.min(input_data, axis=0))
print("Max:", np.max(input_data, axis=0))

print("\nOutput Data Stats:")
print("Means:", np.mean(output_data, axis=0))
print("Standard Deviations:", np.std(output_data, axis=0))
print("Min:", np.min(output_data, axis=0))
print("Max:", np.max(output_data, axis=0))

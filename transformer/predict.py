import torch
import numpy as np
import matplotlib.pyplot as plt
from model import Transformer, TransformerConfig

def main():
    # Load the trained model
    model_path = 'best_model.safetensors'
    model_args = {
        'n_layer': 6,
        'n_head': 1,
        'n_embd': 37,
        'scale': 4,
        'd_out': 29,
        'max_len': 1_000,
        'bias': True,
        'dropout': 0.25
    }
    config = TransformerConfig(**model_args)
    model = Transformer(config)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    # Load the test data
    test_inputs = 'data/Ant-v1/test_inputs.npy'
    test_targets = 'data/Ant-v1/test_targets.npy'
    test_inputs = torch.tensor(np.load(test_inputs), dtype=torch.float32)
    test_targets = torch.tensor(np.load(test_targets), dtype=torch.float32)
    # Print dims of inputs and targets
    print(test_inputs.shape, test_targets.shape)
  
    # Select a specific slice of trajectories
    seq_idx = 0
    input_trajectories = test_inputs[seq_idx:seq_idx+5]  # Select 5 input trajectories starting from seq_idx
    target_trajectories = test_targets[seq_idx:seq_idx+5]  # Select 5 target trajectories starting from seq_idx
    # Print dims of inputs and targets
    print(input_trajectories.shape, target_trajectories.shape)
    # Predict the next states using the model
    init_idx = 0
    t = 100  # Number of time steps to predict
    predicted_states, loss = model.predict(input_trajectories, targets=target_trajectories, init=init_idx, t=t)

    # Extract the individual losses from the loss tuple
    _, metrics = loss
    coordinate_loss = metrics['coordinate_loss']
    orientation_loss = metrics['orientation_loss']
    angle_loss = metrics['angle_loss']
    coordinate_velocity_loss = metrics['coordinate_velocity_loss']
    angular_velocity_loss = metrics['angular_velocity_loss']

    # Plot the predicted states and ground truth states
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(range(init_idx, init_idx + t), target_trajectories[0, init_idx:init_idx + t, 0], label='Ground Truth')
    ax.plot(range(init_idx, init_idx + t), [state[0] for state in predicted_states], label='Predicted')
    ax.set_xlabel('Time Step')
    ax.set_ylabel('State')
    ax.set_title('Predicted vs Ground Truth States')
    ax.legend()

    # Plot the individual losses
    fig2, axs = plt.subplots(2, 3, figsize=(15, 8))
    axs[0, 0].plot(range(init_idx, init_idx + t), coordinate_loss)
    axs[0, 0].set_title('Coordinate Loss')
    axs[0, 1].plot(range(init_idx, init_idx + t), orientation_loss)
    axs[0, 1].set_title('Orientation Loss')
    axs[0, 2].plot(range(init_idx, init_idx + t), angle_loss)
    axs[0, 2].set_title('Angle Loss')
    axs[1, 0].plot(range(init_idx, init_idx + t), coordinate_velocity_loss)
    axs[1, 0].set_title('Coordinate Velocity Loss')
    axs[1, 1].plot(range(init_idx, init_idx + t), angular_velocity_loss)
    axs[1, 1].set_title('Angular Velocity Loss')
    axs[1, 2].axis('off')  # Leave the last subplot empty

    for ax in axs.flat:
        ax.set(xlabel='Time Step', ylabel='Loss')

    fig2.tight_layout()

    plt.show()

if __name__ == '__main__':
    main()

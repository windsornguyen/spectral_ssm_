import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
from model import Transformer, TransformerConfig

def smooth_curve(points, sigma=2):
    return gaussian_filter1d(points, sigma=sigma)

def plot_losses(losses, title, x_values=None, ylabel='Loss'):
    if x_values is None:
        x_values = list(range(len(losses)))
    plt.plot(x_values, smooth_curve(losses), label=title)
    plt.xlabel('Time Step')
    plt.ylabel(ylabel)
    plt.legend()

def main():
    # Load the trained model
    controller = 'Ant-v1'
    model_path = f'best_{controller}.safetensors'
    model_args = {
        'n_layer': 6,
        'n_embd': 37,
        'n_head': 1,
        'scale': 16,
        'd_out': 29,
        'max_len': 1_000,
        'bias': False,
        'dropout': 0.0
    }
    config = TransformerConfig(**model_args)
    model = Transformer(config)
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()

    # Load the test data
    test_inputs = f'data/{controller}/test_inputs.npy'
    test_targets = f'data/{controller}/test_targets.npy'
    test_inputs = torch.tensor(np.load(test_inputs), dtype=torch.float32)
    test_targets = torch.tensor(np.load(test_targets), dtype=torch.float32)

    # Select a specific slice of trajectories
    seq_idx = 0
    num_trajectories = 5
    input_trajectories = test_inputs[seq_idx:seq_idx+num_trajectories]
    target_trajectories = test_targets[seq_idx:seq_idx+num_trajectories]

    # Predict the next states using the model
    predicted_states, loss = model.predict(input_trajectories, targets=target_trajectories)

    # Extract the individual losses from the loss tuple
    total_loss, metrics = loss
    coordinate_loss = metrics['coordinate_loss'].detach().cpu().numpy()
    orientation_loss = metrics['orientation_loss'].detach().cpu().numpy()
    angle_loss = metrics['angle_loss'].detach().cpu().numpy()
    coordinate_velocity_loss = metrics['coordinate_velocity_loss'].detach().cpu().numpy()
    angular_velocity_loss = metrics['angular_velocity_loss'].detach().cpu().numpy()

    # Set up plotting style
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, axs = plt.subplots(2, 2, figsize=(12, 8))

    # Plot the predicted states and ground truth states
    for traj_idx in range(num_trajectories):
        axs[0, 0].plot(range(input_trajectories.shape[1] - 1), target_trajectories[traj_idx, 1:, 0].detach().cpu().numpy(), label=f'Ground Truth {traj_idx+1}')
        axs[0, 0].plot(range(input_trajectories.shape[1] - 1), predicted_states[traj_idx, :, 0].detach().cpu().numpy(), label=f'Predicted {traj_idx+1}')
    axs[0, 0].set_xlabel('Time Step')
    axs[0, 0].set_ylabel('State')
    axs[0, 0].set_title('Predicted vs Ground Truth States')
    axs[0, 0].legend()

    # Plot the total loss
    axs[0, 1].plot(range(input_trajectories.shape[1] - 1), total_loss.detach().cpu().numpy())
    axs[0, 1].set_xlabel('Time Step')
    axs[0, 1].set_ylabel('Loss')
    axs[0, 1].set_title('Total Loss')

    # Plot the individual losses
    axs[1, 0].plot(range(input_trajectories.shape[1] - 1), coordinate_loss, label='Coordinate Loss')
    axs[1, 0].plot(range(input_trajectories.shape[1] - 1), orientation_loss, label='Orientation Loss')
    axs[1, 0].plot(range(input_trajectories.shape[1] - 1), angle_loss, label='Angle Loss')
    axs[1, 0].plot(range(input_trajectories.shape[1] - 1), coordinate_velocity_loss, label='Coordinate Velocity Loss')
    axs[1, 0].plot(range(input_trajectories.shape[1] - 1), angular_velocity_loss, label='Angular Velocity Loss')
    axs[1, 0].set_xlabel('Time Step')
    axs[1, 0].set_ylabel('Loss')
    axs[1, 0].set_title('Individual Losses')
    axs[1, 0].legend()

    plt.tight_layout()
    plt.savefig('results/prediction_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()

if __name__ == '__main__':
    main()

import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
from spectral_ssm.model import STU

def smooth_curve(points, sigma=2):
    return gaussian_filter1d(points, sigma=sigma)

def plot_losses(losses, title, x_values=None, ylabel='Loss', color=None):
    if x_values is None:
        x_values = list(range(len(losses)))
    plt.plot(x_values, smooth_curve(losses), label=title, color=color)
    plt.xlabel('Time Step')
    plt.ylabel(ylabel)
    plt.legend()

def main():
    # Load the trained model
    controller = 'Ant-v1'
    model_path = f'physics_checkpoints_ant/Ant-v1-checkpoint-step180-20240607125206.pt'
    model_args = {
        'd_out': 37,
        'input_len': 1000,
        'num_eigh': 24,
        'auto_reg_k_u': 3,
        'auto_reg_k_y': 2,
        'learnable_m_y': True,
    }
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
    model = STU(**model_args).to(device)
    # model.load_state_dict(torch.load(model_path))
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()

    # Load the test data
    test_inputs = f'../data/{controller}/test_inputs.npy'
    test_targets = f'../data/{controller}/test_targets.npy'
    test_inputs = torch.tensor(np.load(test_inputs), dtype=torch.float32).to(device)
    test_targets = torch.tensor(np.load(test_targets), dtype=torch.float32).to(device)

    # Select a specific slice of trajectories
    seq_idx = 0
    num_trajectories = 5
    # TODO: Should really randomize the trajectories and give them as slices?
    input_trajectories = test_inputs[seq_idx:seq_idx+num_trajectories]
    target_trajectories = test_targets[seq_idx:seq_idx+num_trajectories]

    model.eval()
    predicted_states, loss = model.predict(
        inputs=input_trajectories,
        targets=target_trajectories,
        init=0,
        steps=10,
        ar_steps=1
    )
    model.train()

    # Extract the trajectory losses from the loss tuple
    avg_loss, metrics, trajectory_losses = loss

    print(f"Average Loss: {avg_loss.item():.4f}")
    print(f"Shape of predicted states: {predicted_states.shape}")

    # Set up plotting style
    plt.style.use('seaborn-v0_8-whitegrid')
    num_rows = num_trajectories
    num_cols = 2
    fig, axs = plt.subplots(num_rows, num_cols, figsize=(8 * num_cols, 4 * num_rows))

    # Generate random colors
    colors = [f"#{random.randint(0, 0xFFFFFF):06x}" for _ in range(num_trajectories)]

    # Plot the predicted states, ground truth states, and trajectory losses
    for traj_idx in range(num_trajectories):
        time_steps = predicted_states.shape[1]
        print(f"Plotting trajectory {traj_idx+1} over {time_steps} time steps")

        # Plot the predicted states and ground truth states
        axs[traj_idx, 0].clear()
        axs[traj_idx, 0].plot(range(time_steps), target_trajectories[traj_idx, :time_steps, 5].detach().cpu().numpy(), label=f'Ground Truth {traj_idx+1}', color=colors[traj_idx], linewidth=2, linestyle='--')
        axs[traj_idx, 0].plot(range(time_steps), predicted_states[traj_idx, :, 5].detach().cpu().numpy(), label=f'Predicted {traj_idx+1}', color=colors[traj_idx], linewidth=2)
        axs[traj_idx, 0].set_title(f'Trajectory {traj_idx+1}: Predicted vs Ground Truth')
        axs[traj_idx, 0].set_xlabel('Time Step')
        axs[traj_idx, 0].set_ylabel('State')
        axs[traj_idx, 0].legend()
        axs[traj_idx, 0].grid(True)

        # Plot the trajectory losses
        axs[traj_idx, 1].clear()
        axs[traj_idx, 1].plot(range(time_steps), smooth_curve(trajectory_losses[traj_idx].detach().cpu().numpy()), color=colors[traj_idx], linewidth=2)
        axs[traj_idx, 1].set_title(f'Trajectory {traj_idx+1}: Loss')
        axs[traj_idx, 1].set_xlabel('Time Step')
        axs[traj_idx, 1].set_ylabel('Loss')
        axs[traj_idx, 1].grid(True)

    plt.tight_layout()
    plt.savefig(f'results/{controller}_predictions.png', dpi=300, bbox_inches='tight')
    plt.show()


if __name__ == '__main__':
    main()

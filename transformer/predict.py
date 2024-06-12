# =============================================================================#
# Authors: Isabel Liu, Windsor Nguyen
# File: predict.py
# =============================================================================#

import os
import sys

script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(script_dir)
sys.path.append(parent_dir)

import torch
import numpy as np
import matplotlib.pyplot as plt
import random
from scipy.ndimage import gaussian_filter1d
from model import Transformer, TransformerConfig
from losses.loss_ant import AntLoss
from losses.loss_cheetah import HalfCheetahLoss
from losses.loss_walker import Walker2DLoss
import safetensors

controller = 'HalfCheetah-v1'


def smooth_curve(points, sigma=2):
    return gaussian_filter1d(points, sigma=sigma)


def predict_and_plot(
    model,
    input_trajectories,
    target_trajectories,
    num_trajectories,
    init_steps,
    pred_steps,
    ar_steps,
):
    model.eval()
    predicted_states, loss = model.predict(
        inputs=input_trajectories,
        targets=target_trajectories,
        init=init_steps,
        steps=pred_steps,
        ar_steps=ar_steps,
    )
    print('First few predicted states for each trajectory:')
    for traj_idx in range(num_trajectories):
        print(f'Trajectory {traj_idx+1}: {predicted_states[traj_idx, :5, 0]}')

    avg_loss, metrics, trajectory_losses = loss
    print(f'Average Loss: {avg_loss.item():.4f}')
    print(f'Shape of predicted states: {predicted_states.shape}')

    predicted_states = predicted_states.detach().cpu().numpy()
    target_trajectories = target_trajectories.detach().cpu().numpy()
    trajectory_losses = trajectory_losses.detach().cpu().numpy()

    plt.style.use('seaborn-v0_8-whitegrid')
    num_rows = num_trajectories
    num_cols = 2
    fig, axs = plt.subplots(
        num_rows, num_cols, figsize=(8 * num_cols, 4 * num_rows)
    )

    colors = [
        f'#{random.randint(0, 0xFFFFFF):06x}' for _ in range(num_trajectories)
    ]

    for traj_idx in range(num_trajectories):
        time_steps = predicted_states.shape[1]
        print(f'Plotting trajectory {traj_idx+1} over {time_steps} time steps')

        axs[traj_idx, 0].plot(
            range(time_steps),
            target_trajectories[traj_idx, :time_steps, 0],
            color='red',
            linestyle='--',
            label=f'Ground Truth {traj_idx+1}',
            linewidth=2,
        )
        axs[traj_idx, 0].plot(
            range(time_steps),
            predicted_states[traj_idx, :, 0],
            color=colors[traj_idx],
            linestyle='-',
            label=f'Predicted {traj_idx+1}',
            linewidth=2,
        )
        axs[traj_idx, 0].set_title(
            f'Trajectory {traj_idx+1}: Predicted vs Ground Truth'
        )
        axs[traj_idx, 0].set_xlabel('Time Step')
        axs[traj_idx, 0].set_ylabel('State')
        axs[traj_idx, 0].legend()
        axs[traj_idx, 0].grid(True)

        axs[traj_idx, 1].plot(
            range(time_steps),
            smooth_curve(trajectory_losses[traj_idx]),
            color='green',
            linestyle='-',
            label=f'Loss {traj_idx+1}',
            linewidth=2,
        )
        axs[traj_idx, 1].set_title(f'Trajectory {traj_idx+1}: Loss')
        axs[traj_idx, 1].set_xlabel('Time Step')
        axs[traj_idx, 1].set_ylabel('Loss')
        axs[traj_idx, 1].legend()
        axs[traj_idx, 1].grid(True)

    plt.tight_layout()
    plt.savefig(
        f'results/{controller}_predictions.png', dpi=300, bbox_inches='tight'
    )
    plt.show()


def main():
    model_path = f'best_{controller}.safetensors'
    loss_fn = {
        'HalfCheetah-v1': HalfCheetahLoss,
        'Walker2D-v1': Walker2DLoss,
        'Ant-v1': AntLoss,
    }[controller]()
    model_args = {
        'n_layer': 6,
        'n_embd': 24,
        'n_head': 8,
        'scale': 16,
        'd_out': 18,
        'max_len': 1000,
        'bias': False,
        'dropout': 0.25,
        'loss_fn': loss_fn,
    }
    device = torch.device(
        'cuda'
        if torch.cuda.is_available()
        else 'mps'
        if torch.backends.mps.is_available()
        else 'cpu'
    )
    config = TransformerConfig(**model_args)
    model = Transformer(config).to(device)
    state_dict = safetensors.load_file(model_path)
    model.load_state_dict(state_dict)

    test_inputs = torch.tensor(
        np.load(f'../data/{controller}/test_inputs.npy'), dtype=torch.float32
    ).to(device)
    test_targets = torch.tensor(
        np.load(f'../data/{controller}/test_targets.npy'), dtype=torch.float32
    ).to(device)

    num_trajectories = 5
    idxs = torch.randint(0, test_inputs.shape[0], (num_trajectories,))
    input_trajectories = test_inputs[idxs]
    target_trajectories = test_targets[idxs]

    init_steps = 0
    pred_steps = 10
    ar_steps = 1
    predict_and_plot(
        model,
        input_trajectories,
        target_trajectories,
        num_trajectories,
        init_steps,
        pred_steps,
        ar_steps,
    )


if __name__ == '__main__':
    main()

# def smooth_curve(points, sigma=2):
#     return gaussian_filter1d(points, sigma=sigma)


# def predict_and_plot(model, input_trajectories, target_trajectories, num_trajectories, init_steps, pred_steps, ar_steps):
#     model.eval()
#     predicted_states, loss = model.predict(
#         inputs=input_trajectories,
#         targets=target_trajectories,
#         init=init_steps,
#         steps=pred_steps,
#         ar_steps=ar_steps
#     )

#     avg_loss, metrics, trajectory_losses = loss
#     print(f"Average Loss: {avg_loss.item():.4f}")
#     print(f"Shape of predicted states: {predicted_states.shape}")

#     predicted_states = predicted_states.detach().cpu().numpy()
#     target_trajectories = target_trajectories.detach().cpu().numpy()
#     trajectory_losses = trajectory_losses.detach().cpu().numpy()

#     plt.style.use('seaborn-v0_8-whitegrid')
#     colors = [f"#{random.randint(0, 0xFFFFFF):06x}" for _ in range(num_trajectories)]

#     for traj_idx in range(num_trajectories):
#         plt.figure(figsize=(10, 4))
#         plt.subplot(121)
#         plt.plot(target_trajectories[traj_idx, :, 5], 'r--', label='Ground Truth')
#         plt.plot(predicted_states[traj_idx, :, 5], 'b-', label='Predicted')
#         plt.title(f'Trajectory {traj_idx + 1}')
#         plt.xlabel('Time Step')
#         plt.ylabel('State')
#         plt.legend()

#         plt.subplot(122)
#         plt.plot(smooth_curve(trajectory_losses[traj_idx]), 'g-', label='Loss')
#         plt.title(f'Trajectory {traj_idx + 1} Loss')
#         plt.xlabel('Time Step')
#         plt.ylabel('Loss')
#         plt.legend()
#         plt.tight_layout()
#         plt.show()


# def main():
#     controller = 'HalfCheetah-v1'
#     model_path = f'best_{controller}.safetensors'
#     loss_fn = {'HalfCheetah-v1': HalfCheetahLoss, 'Walker2D-v1': Walker2DLoss, 'Ant-v1': AntLoss}[controller]()
#     model_args = {
#         'n_layer': 6,
#         'n_embd': 24,
#         'n_head': 8,
#         'scale': 16,
#         'd_out': 18,
#         'max_len': 1000,
#         'bias': False,
#         'dropout': 0.25,
#         'loss_fn': loss_fn
#     }
#     device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
#     config = TransformerConfig(**model_args)
#     model = Transformer(config).to(device)
#     model.load_state_dict(torch.load(model_path, map_location=device))

#     test_inputs = torch.tensor(np.load(f'../data/{controller}/test_inputs.npy'), dtype=torch.float32).to(device)
#     test_targets = torch.tensor(np.load(f'../data/{controller}/test_targets.npy'), dtype=torch.float32).to(device)

#     num_trajectories = 5
#     idxs = torch.randint(0, test_inputs.shape[0], (num_trajectories,))
#     input_trajectories = test_inputs[idxs]
#     target_trajectories = test_targets[idxs]

#     init_steps = 0
#     pred_steps = 10
#     ar_steps = 1

#     predict_and_plot(model, input_trajectories, target_trajectories, num_trajectories, init_steps, pred_steps, ar_steps)


# if __name__ == '__main__':
#     main()

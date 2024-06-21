import torch
import numpy as np
import matplotlib.pyplot as plt
import random
from scipy.ndimage import gaussian_filter1d
from model import Transformer, TransformerConfig
from losses.loss_ant import AntLoss
from losses.loss_cheetah import HalfCheetahLoss
from losses.loss_walker import Walker2DLoss
from torch.nn import MSELoss

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
    # Set seed for reproducibility
    # torch.manual_seed(42)
    # np.random.seed(42)
    # random.seed(42)

    # Load the trained model
    controller = 'Ant-v1'
    model_path = f'best_{controller}.safetensors'
    # loss_fn = HalfCheetahLoss() if controller == 'HalfCheetah-v1' else Walker2DLoss() if controller == 'Walker2D-v1' else AntLoss()
    loss_fn = MSELoss()
    model_args = {
        'n_layers': 12,
        'n_embd': 512,  # Embedding dimension
        'n_head': 16,  # Constraint: n_head % n_embd == 0
        'sl': 300, # Sequence length
        'scale': 4,
        'bias': False,  # True: bias in Linears and LayerNorms, like GPT-2. False: a bit better and faster
        'dropout': 0.10,
        'use_dilated_attn': False,
        'loss_fn': None,
        'lr': 6e-4,
        'controls': {'task': 'mujoco-v3', 'controller': 'Ant-v1'}
    }
    device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
    config = TransformerConfig(**model_args)
    model = Transformer(config).to(device)
    # model.load_state_dict(torch.load(model_path))
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))

    # Load the test data
    test_inputs = f'../data/{controller}/test_inputs.npy'
    test_targets = f'../data/{controller}/test_targets.npy'
    test_inputs = torch.tensor(np.load(test_inputs), dtype=torch.float32).to(device)
    test_targets = torch.tensor(np.load(test_targets), dtype=torch.float32).to(device)

    # Select a specific number videos to predict
    seq_idx = 0
    num_videos = 5
    # TODO: Should really randomize the videos and give them as slices?
    input_videos = test_inputs[seq_idx : seq_idx + num_videos]
    target_videos = test_targets[seq_idx : seq_idx + num_videos]

    model.eval()
    predicted_states, loss = model.predict(
        inputs=input_videos,
        targets=target_videos,
        init=0,
        steps=50,
        ar_steps=300
    )
    model.train()

    # Extract the video losses from the loss tuple
    avg_loss, video_losses = loss

    print(f'Average Loss: {avg_loss.item():.4f}')
    print(f'Shape of predicted states: {predicted_states.shape}')

    # Set up plotting style
    plt.style.use('seaborn-v0_8-whitegrid')
    num_rows = num_videos
    num_cols = 2
    fig, axs = plt.subplots(num_rows, num_cols, figsize=(8 * num_cols, 4 * num_rows))

    # Generate random colors
    colors = [f'#{random.randint(0, 0xFFFFFF):06x}' for _ in range(num_videos)]

    # Plot the predicted states, ground truth states, and video losses
    for video_idx in range(num_videos):
        time_steps = predicted_states.shape[1]
        print(f'Plotting video {video_idx + 1} over {time_steps} time steps')

        # # Plot the predicted states and ground truth states
        # axs[video_idx, 0].clear()
        # axs[video_idx, 0].plot(range(time_steps), target_videos[video_idx, :time_steps, 5].detach().cpu().numpy(), label=f'Ground Truth {video_idx+1}', color=colors[video_idx], linewidth=2, linestyle='--')
        # axs[video_idx, 0].plot(range(time_steps), predicted_states[video_idx, :, 5].detach().cpu().numpy(), label=f'Predicted {video_idx+1}', color=colors[video_idx], linewidth=2)
        # axs[video_idx, 0].set_title(f'video {video_idx+1}: Predicted vs Ground Truth')
        # axs[video_idx, 0].set_xlabel('Time Step')
        # axs[video_idx, 0].set_ylabel('State')
        # axs[video_idx, 0].legend()
        # axs[video_idx, 0].grid(True)

        # Plot the video losses
        axs[video_idx, 1].clear()
        axs[video_idx, 1].plot(range(time_steps), smooth_curve(video_losses[video_idx].detach().cpu().numpy()), color=colors[video_idx], linewidth=2)
        axs[video_idx, 1].set_title(f'video {video_idx+1}: Loss')
        axs[video_idx, 1].set_xlabel('Time Step')
        axs[video_idx, 1].set_ylabel('Loss')
        axs[video_idx, 1].grid(True)

    plt.tight_layout()
    plt.savefig(f'results/{controller}_predictions.png', dpi=300, bbox_inches='tight')
    plt.show()


if __name__ == '__main__':
    main()
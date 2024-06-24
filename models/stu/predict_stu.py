import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
import random
from scipy.ndimage import gaussian_filter1d
from safetensors.torch import load_file


from models.stu.model import SSSM, SSSMConfigs
from torch.nn import MSELoss


def smooth_curve(points, sigma=2):
    return gaussian_filter1d(points, sigma=sigma)


def plot_losses(losses, title, x_values=None, ylabel="Loss", color=None):
    if x_values is None:
        x_values = list(range(len(losses)))
    plt.plot(x_values, smooth_curve(losses), label=title, color=color)
    plt.xlabel("Time Step")
    plt.ylabel(ylabel)
    plt.legend()


def main():
    # Process command line flags
    parser = argparse.ArgumentParser(
        description="Inference script for sequence prediction"
    )
    parser.add_argument(
        "--controller",
        type=str,
        default="Ant-v1",
        choices=["Ant-v1", "HalfCheetah-v1", "Walker2D-v1"],
        help="Controller to use for the MuJoCo environment.",
    )
    parser.add_argument(
        "--task",
        type=str,
        default="mujoco-v3",
        choices=["mujoco-v1", "mujoco-v2", "mujoco-v3"],
        help="Task to run inference on.",
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load the trained model
    # model_path = f"_sssm-{args.controller}-model_step-239-2024-06-24-14-23-16.pt"
    model_path = "pls_work.pt"

    configs = SSSMConfigs(
        n_layers=2,
        n_embd=512,
        d_out=512,
        sl=300,
        scale=4,
        bias=False,
        dropout=0.10,
        num_eigh=24,
        auto_reg_k_u=3,
        auto_reg_k_y=2,
        learnable_m_y=True,
        loss_fn=MSELoss(),
        controls={"task": args.task, "controller": args.controller},
    )

    # Initialize and load the model
    model = SSSM(configs).to(device)
    model = torch.compile(model)
    state_dict = load_file(model_path, device="cuda:0")
    model.load_state_dict(state_dict)
    model.eval()

    # Load the test data
    if args.task in ["mujoco-v1", "mujoco-v2"]:
        base_path = f"data/{args.task}/{args.controller}/"
        test_inputs = np.load(f"{base_path}/test_inputs.npy")
        test_targets = np.load(f"{base_path}/test_targets.npy")
        test_inputs = torch.from_numpy(test_inputs).float().to(device)
        test_targets = torch.from_numpy(test_targets).float().to(device)
    elif args.task == "mujoco-v3":
        test_data = torch.load(
            f"data/{args.task}/{args.controller}/{args.controller}_ResNet-18_test.pt",
            map_location=device,
        )
        test_inputs = test_data
        test_targets = test_data  # For mujoco-v3, inputs and targets are the same
    else:
        raise ValueError("Invalid task")

    # Run inference
    num_videos = 5
    predicted_states = []
    video_losses = []

    with torch.no_grad():
        for i in range(num_videos):
            input_video = test_inputs[i : i + 1]
            target_video = test_targets[i : i + 1]

            pred_states, (avg_loss, video_loss) = model.predict_frames(
                inputs=input_video, targets=target_video, init=100, steps=50, ar_steps=300
            )

            predicted_states.append(pred_states)
            video_losses.append(video_loss)

    predicted_states = torch.cat(predicted_states, dim=0)
    video_losses = torch.cat(video_losses, dim=0)

    print(f"Shape of predicted states: {predicted_states.shape}")
    print(f"Shape of video losses: {video_losses.shape}")

    # Print out predictions for each video and check if they're all the same
    for i in range(num_videos):
        print(f"\nPredictions for Video {i+1}:")
        print(predicted_states[i, :5, 0])  # Print first 5 time steps of first feature

    # Check if all predictions are the same
    all_same = True
    for i in range(1, num_videos):
        if not torch.allclose(predicted_states[0], predicted_states[i], atol=1e-6):
            all_same = False
            break

    print(f"\nAll predictions are the same: {all_same}")

    if all_same:
        print(
            "All predictions are identical. This might indicate an issue with the model or data processing."
        )
    else:
        print(
            "Predictions differ between videos, which is expected for different inputs."
        )

    # Add the new code here to save predictions and ground truths
    print("pred state shape", predicted_states.shape)
    np.savetxt("predictions.txt", predicted_states[:5, :, 500].cpu().numpy())
    np.savetxt(
        "ground_truths.txt",
        test_targets[:5, : predicted_states.shape[1], 500].cpu().numpy(),
    )
    print(
        "Predictions and ground truths saved to 'predictions.txt' and 'ground_truths.txt' respectively."
    )

    # Plotting
    plt.style.use("seaborn-v0_8-whitegrid")
    num_rows = num_videos
    num_cols = 2
    fig, axs = plt.subplots(num_rows, num_cols, figsize=(8 * num_cols, 4 * num_rows))

    colors = [f"#{random.randint(0, 0xFFFFFF):06x}" for _ in range(num_videos)]

    for video_idx in range(num_videos):
        time_steps = predicted_states.shape[1]
        print(f"Plotting video {video_idx + 1} over {time_steps} time steps")

        # Plot the predicted states and ground truth states
        for feature_idx in range(3):  # Plot first three features
            axs[video_idx, 0].plot(
                range(time_steps),
                test_targets[video_idx, :time_steps, feature_idx].cpu().numpy(),
                label=f"Ground Truth {video_idx+1}, Feature {feature_idx+1}",
                color=colors[video_idx],
                linewidth=2,
                linestyle="--",
            )
            axs[video_idx, 0].plot(
                range(time_steps),
                predicted_states[video_idx, :, feature_idx].cpu().numpy(),
                label=f"Predicted {video_idx+1}, Feature {feature_idx+1}",
                color=colors[video_idx],
                linewidth=2,
            )

        axs[video_idx, 0].set_title(f"Video {video_idx+1}: Predicted vs Ground Truth")
        axs[video_idx, 0].set_xlabel("Time Step")
        axs[video_idx, 0].set_ylabel("State")
        axs[video_idx, 0].legend()
        axs[video_idx, 0].grid(True)

        # Plot the video losses (scaled up by 100)
        scaled_losses = smooth_curve(video_losses[video_idx].cpu().numpy()) * 100
        axs[video_idx, 1].plot(
            range(time_steps),
            scaled_losses,
            color=colors[video_idx],
            linewidth=2,
        )
        axs[video_idx, 1].set_title(f"Video {video_idx+1}: Loss (scaled x100)")
        axs[video_idx, 1].set_xlabel("Time Step")
        axs[video_idx, 1].set_ylabel("Loss (scaled)")
        axs[video_idx, 1].grid(True)

    plt.tight_layout()
    plt.savefig(
        f"results/{args.controller}_{args.task}_predictions_scaled_losses.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.show()


if __name__ == "__main__":
    main()

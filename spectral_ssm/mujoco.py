# ==============================================================================#
# Authors: Windsor Nguyen
# File: mujoco.py
# ==============================================================================#

"""For MuJoCo experimentation."""

from collections import UserDict

import gym
registry = UserDict(gym.envs.registration.registry)
registry.env_specs = gym.envs.registration.registry
gym.envs.registration.registry = registry

import pybullet_envs
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.evaluation import evaluate_policy

import stu_utils


class STU(nn.Module):
    def __init__(
        self,
        d_out: int = 256,
        input_len: int = 28,  # Adjusted to match RL observation length
        num_eigh: int = 24,
        auto_reg_k_u: int = 3,
        auto_reg_k_y: int = 2,
        learnable_m_y: bool = True,
    ) -> None:
        super(STU, self).__init__()
        self.d_out = d_out
        self.eigh = stu_utils.get_top_hankel_eigh(input_len, num_eigh)
        self.l, self.k = input_len, num_eigh
        self.auto_reg_k_u = auto_reg_k_u
        self.auto_reg_k_y = auto_reg_k_y
        self.learnable_m_y = learnable_m_y
        self.m_x_var = 1.0 / (float(self.d_out) ** 0.5)

        if learnable_m_y:
            self.m_y = nn.Parameter(
                torch.zeros([self.d_out, self.auto_reg_k_y, self.d_out])
            )
        else:
            self.register_buffer(
                'm_y', torch.zeros([self.d_out, self.auto_reg_k_y, self.d_out])
            )

        self.m_u = nn.Parameter(
            stu_utils.get_random_real_matrix((d_out, d_out, auto_reg_k_u), self.m_x_var)
        )

        self.m_phi = nn.Parameter(torch.zeros(d_out * num_eigh, d_out))

    def apply_stu(self, inputs: torch.Tensor) -> torch.Tensor:
        eig_vals, eig_vecs = self.eigh
        eig_vals = eig_vals.to(inputs.device)
        eig_vecs = eig_vecs.to(inputs.device)
        self.m_phi = self.m_phi.to(inputs.device)
        self.m_u = self.m_u.to(inputs.device)
        self.m_y = self.m_y.to(inputs.device)
        x_tilde = stu_utils.compute_x_tilde(inputs, (eig_vals, eig_vecs))
        delta_phi = x_tilde @ self.m_phi
        delta_ar_u = stu_utils.compute_ar_x_preds(self.m_u, inputs)
        return stu_utils.compute_y_t(self.m_y, delta_phi + delta_ar_u)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return torch.vmap(self.apply_stu)(inputs)


class Architecture(nn.Module):
    def __init__(
        self,
        d_model=256,
        d_target=28,  # Changed to match the observation dimension
        num_layers=6,
        dropout=0.1,
        input_len=28,
        num_eigh=24,
        auto_reg_k_u=3,
        auto_reg_k_y=2,
        learnable_m_y=True,
    ):
        super(Architecture, self).__init__()
        self.embedding = nn.Linear(input_len, d_model)
        self.layer_norms = nn.ModuleList(
            [nn.LayerNorm(d_model) for _ in range(num_layers)]
        )
        self.layers = nn.ModuleList(
            [
                nn.Sequential(
                    STU(
                        d_model,
                        input_len,
                        num_eigh,
                        auto_reg_k_u,
                        auto_reg_k_y,
                        learnable_m_y,
                    ),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(d_model, 2 * d_model),
                    nn.GLU(dim=-1),
                    nn.Dropout(dropout),
                )
                for _ in range(num_layers)
            ]
        )
        self.projection = nn.Linear(d_model, d_target)

    def forward(self, inputs):
        x = self.embedding(inputs)

        for i, layer in enumerate(self.layers):
            z = x
            x = self.layer_norms[i](x)
            x = layer(x) + z

        x = x[:, -1]  # Take the last output as the prediction
        return self.projection(x)


# Create the environment
def make_env():
    env = gym.make("AntBulletEnv-v0")
    return env

env = DummyVecEnv([make_env])

# Define the dimensions
OBSERVATION_DIM = env.observation_space.shape[0]
ACTION_DIM = env.action_space.shape[0]

# Instantiate the model
model = Architecture(d_model=256, d_target=OBSERVATION_DIM)

# Define training loop
optimizer = optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.MSELoss()

num_episodes = 1000
seq_length = 32  # Length of the observation sequence

for episode in range(num_episodes):
    env.env_method("seed", episode)  # Set the seed for each environment
    obs = env.reset()
    obs = obs[0]  # Get the initial observation
    done = False
    total_loss = 0

    obs_sequence = [obs]  # Initialize the observation sequence

    while not done:
        if len(obs_sequence) >= seq_length:
            obs_tensor = torch.tensor(obs_sequence[-seq_length:], dtype=torch.float32).unsqueeze(0)
            pred = model(obs_tensor)

            target = torch.tensor(obs, dtype=torch.float32)

            loss = criterion(pred.squeeze(), target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        action = env.action_space.sample()
        action = np.array(action).reshape(1, -1)  # Reshape the action to (1, 8)
        new_obs, reward, terminated, info = env.step(action)
        new_obs = new_obs[0]  # Get the new observation

        obs_sequence.append(new_obs)
        obs = new_obs

    print(f"Episode {episode + 1}/{num_episodes}, Loss: {total_loss:.4f}")

print("Training completed.")

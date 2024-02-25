# Copyright 2024 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Example training loop."""

import tqdm
from spectral_ssm import model, cifar10, optimizer
from spectral_ssm.experiment import Experiment


def main():
    # Hyperparameters
    train_batch_size = 17
    eval_batch_size = 16
    num_steps = 180_000
    eval_period = 1000
    warmup_steps = 18_000
    learning_rate = 5e-4
    weight_decay = 1e-1
    m_y_learning_rate = 5e-5
    m_y_weight_decay = 0

    # Define the model
    spectral_ssm = model.Architecture(
        d_model=32,
        d_target=10,
        num_layers=6,
        dropout=0.1,
        input_len=1024,
        num_eigh=24,
        auto_reg_k_u=3,
        auto_reg_k_y=2,
        learnable_m_y=True,
    )

    # Assuming `get_optimizer` returns both optimizer and scheduler
    opt, scheduler = optimizer.get_optimizer(
        spectral_ssm,
        num_steps=num_steps,
        warmup_steps=warmup_steps,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        m_y_learning_rate=m_y_learning_rate,
        m_y_weight_decay=m_y_weight_decay,
    )

    exp = Experiment(model=spectral_ssm, optimizer=opt)

    training_loader = cifar10.get_dataset("train", batch_size=train_batch_size)
    eval_loader = cifar10.get_dataset("test", batch_size=eval_batch_size)

    pbar = tqdm.tqdm(range(num_steps))
    for global_step, (inputs, targets) in enumerate(training_loader):
        metrics = exp.step(inputs, targets)
        pbar.set_description(
            f'Step {global_step} - train/acc: {metrics["accuracy"]:.2f} train/loss: {metrics["loss"]:.2f}'
        )
        scheduler.step()  # Update learning rate

        if global_step > 0 and global_step % eval_period == 0:
            epoch_metrics = exp.evaluate(
                eval_loader
            )  # Adjusted to use PyTorch DataLoader
            print(
                f"Eval {global_step}: acc: {epoch_metrics['accuracy']:.2f}, loss: {epoch_metrics['loss']:.2f}"
            )

        if global_step >= num_steps:
            break


if __name__ == "__main__":
    main()

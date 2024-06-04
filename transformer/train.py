import torch
import torch.nn as nn
import os
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

from model import Transformer, TransformerConfig
from data import get_dataloader

def main():
    # hyperparameters
    batch_size = 5  # how many independent sequences will we process in parallel?
    max_len = 1_000  # what is the maximum context length for predictions?
    max_iters = 3_500 // 6
    eval_interval = 25
    learning_rate = 1e-3
    device = 'mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu'
    print('Running on device:', device)
    n_embd = 37
    d_out = 29
    n_head = 1
    scale = 4 # 4 is default
    n_layer = 6
    dropout = 0.25
    bias = True
    patience = 10  # Number of validation steps to wait for improvement
    # ------------

    # Set seed for reproducibility
    torch.manual_seed(1337)

    # data loading
    train_inputs = 'data/Ant-v1/yagiz_train_inputs.npy'
    train_targets = 'data/Ant-v1/yagiz_train_targets.npy'
    val_inputs = 'data/Ant-v1/yagiz_val_inputs.npy'
    val_targets = 'data/Ant-v1/yagiz_val_targets.npy'

    # Get dataloaders
    train_loader = get_dataloader(train_inputs, train_targets, 'train', batch_size, device)
    val_loader = get_dataloader(val_inputs, val_targets, 'val', batch_size, device)

    @torch.no_grad()
    def evaluate(loader):
        print('Evaluating on validation set...')
        model.eval()
        losses = []
        for X, y in tqdm(loader, desc='Evaluating', unit='iter'):
            X, y = X.to(device), y.to(device)
            preds, loss = model(X, y)
            loss, _ = loss
            losses.append(loss.item())
        model.train()
        return np.mean(losses)

    model_args = {
        'n_layer': n_layer,
        'n_head': n_head,
        'n_embd': n_embd,
        'scale': scale,
        'd_out': d_out,
        'max_len': max_len,
        'bias': bias, 
        'dropout': dropout
    }

    config = TransformerConfig(**model_args)
    model = Transformer(config)
    m = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    # Print the number of parameters in the model
    print(sum(p.numel() for p in m.parameters()) / 1e6, 'M parameters')

    best_val_loss = float('inf')
    patience_counter = 0

    # Initialize lists to store losses
    train_losses = []
    val_losses = []
    coord_losses = []
    orient_losses = []
    angle_losses = []
    coord_vel_losses = []
    ang_vel_losses = []

    pbar = tqdm(range(max_iters), desc='Training', unit='iter')
    val_loss = None
    for step, (xb, yb) in zip(pbar, train_loader):
        # Store total training loss and losses specific to each observation feature
        train_losses.append(loss.item())
        coord_losses.append(metrics['coordinate_loss'])
        orient_losses.append(metrics['orientation_loss'])
        angle_losses.append(metrics['angle_loss'])
        coord_vel_losses.append(metrics['coordinate_velocity_loss'])
        ang_vel_losses.append(metrics['angular_velocity_loss'])
        
        # every once in a while evaluate the loss on train and val sets
        if (step != 0 and step % eval_interval == 0) or step == max_iters - 1:
            val_loss = evaluate(val_loader)
            val_losses.append(val_loss)
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                torch.save(model.state_dict(), 'best_model.safetensors')
            else:
                patience_counter += 1
            
            if patience_counter >= patience:
                print(f'Early stopping triggered. Best validation loss: {best_val_loss:.4f}')
                break
            
        # Evaluate the loss
        xb, yb = xb.to(device), yb.to(device)
        preds, loss = model(xb, yb)
        loss, metrics = loss

        # Check if any inputs, outputs or weights contain NaNs
        if torch.isnan(preds).any() or torch.isnan(loss):
            print("NaN detected!")
            print("Inputs: ", xb)
            print("Outputs: ", preds)
            print("Loss: ", loss)
            for name, param in model.named_parameters():
                if param is not None and torch.isnan(param).any():
                    print(f"NaN in {name}")

        optimizer.zero_grad(set_to_none=True)
        loss.backward()

        # Print gradients to check for NaN and compute grad norm
        grad_norm = 0.0
        for name, param in model.named_parameters():
            if param.grad is not None:
                grad_norm += param.grad.data.norm(2).item() ** 2
                if torch.isnan(param.grad).any():
                    print(f"NaN gradient in {name}")
        grad_norm = grad_norm ** 0.5
        pbar.set_postfix({
            'Training Loss': loss.item(),
            'Validation Loss': val_loss,
            'Gradient Norm': grad_norm,
            'Coordinate Loss': metrics['coordinate_loss'],
            'Orientation Loss': metrics['orientation_loss'],
            'Angle Loss': metrics['angle_loss'],
            'Coordinate Velocity Loss': metrics['coordinate_velocity_loss'],
            'Angular Velocity Loss': metrics['angular_velocity_loss']
        })

        optimizer.step()

    # After training, plot the losses
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Training Loss')
    plt.plot([i * eval_interval for i in range(len(val_losses))], val_losses, label='Validation Loss')
    plt.title('Losses over Time')
    plt.xlabel('Time Step')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

    plt.figure(figsize=(10, 5))
    plt.plot(coord_losses, label='Coordinate Loss')
    plt.plot(orient_losses, label='Orientation Loss')
    plt.plot(angle_losses, label='Angle Loss')
    plt.plot(coord_vel_losses, label='Coordinate Velocity Loss')
    plt.plot(ang_vel_losses, label='Angular Velocity Loss')
    plt.title('Other Losses over Time')
    plt.xlabel('Time Step')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

if __name__ == '__main__':
    main()

import torch
import torch.nn as nn
import os
import numpy as np
from tqdm import tqdm

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
    d_out = 18
    n_head = 1
    scale = 16 # 4 is default
    n_layer = 6
    dropout = 0.25
    bias = True
    # ------------

    # Set seed for reproducibility
    torch.manual_seed(1337)

    # data loading
    train_inputs = 'data/HalfCheetah/train_inputs.npy'
    train_targets = 'data/HalfCheetah/train_targets.npy'
    val_inputs = 'data/HalfCheetah/val_inputs.npy'
    val_targets = 'data/HalfCheetah/val_targets.npy'

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


    pbar = tqdm(range(max_iters), desc='Training', unit='iter')
    val_loss = None
    for step, (xb, yb) in zip(pbar, train_loader):
        # every once in a while evaluate the loss on train and val sets
        if (step != 0 and step % eval_interval == 0) or step == max_iters - 1:
            val_loss = evaluate(val_loader)
            # pbar.set_postfix({'train_loss': train_loss, 'val_loss': val_loss})

        # evaluate the loss
        xb, yb = xb.to(device), yb.to(device)
        preds, loss = model(xb, yb)
        loss, _ = loss

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
        pbar.set_postfix({'Training Loss': loss.item(), 'Validation Loss': val_loss, 'Gradient Norm': grad_norm})

        optimizer.step()


if __name__ == '__main__':
    main()

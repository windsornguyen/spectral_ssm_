import torch
import torch.nn as nn
import os
import numpy as np
from tqdm import tqdm

from model.model import SSM, SSMConfig

# hyperparameters
batch_size = 128  # how many independent sequences will we process in parallel?
ctxt_len = 1_024  # what is the maximum context length for predictions?
max_iters = 1
eval_interval = 25
learning_rate = 1e-3
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 25
d_embd = 64
n_head = 4
n_layer = 1
dropout = 0.0
bias = True
# ------------

torch.manual_seed(1337)
dataset = 'tiny_shakespeare'

# data loading
data_dir = os.path.join('data', dataset)

def get_batch(split):
    # We recreate np.memmap every batch to avoid a memory leak, as per
    # https://stackoverflow.com/questions/45132940/numpy-memmap-memory-usage-want-to-iterate-once/61472122#61472122
    if split == 'train':
        data = np.memmap(os.path.join(data_dir, 'train.bin'), dtype=np.uint16, mode='r')
    else:
        data = np.memmap(os.path.join(data_dir, 'val.bin'), dtype=np.uint16, mode='r')
    ix = torch.randint(len(data) - ctxt_len, (batch_size,))
    x = torch.stack([torch.from_numpy((data[i:i+ctxt_len]).astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy((data[i+1:i+1+ctxt_len]).astype(np.int64)) for i in ix])
    if device == 'cuda':
        # pin arrays x,y, which allows us to move them to GPU asynchronously (non_blocking=True)
        x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
    else:
        x, y = x.to(device), y.to(device)
    return x, y

@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

model_args = dict(n_layer=n_layer, n_head=n_head, d_embd=d_embd, ctxt_len=ctxt_len,
                  bias=bias, vocab_size=None, dropout=dropout)
meta_vocab_size = None  # Add this line to define meta_vocab_size
model_args['vocab_size'] = 65  # Set vocab_size to 65 for tiny_shakespeare dataset
config = SSMConfig(**model_args)
model = SSM(config)
m = model.to(device)
# print the number of parameters in the model
print(sum(p.numel() for p in m.parameters())/1e6, 'M parameters')

# create a PyTorch optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

pbar = tqdm(range(max_iters), desc='Training', unit='iter')

for iter in pbar:
    # every once in a while evaluate the loss on train and val sets
    if iter % eval_interval == 0 or iter == max_iters - 1:
        losses = estimate_loss()
        pbar.set_postfix({'train_loss': losses['train'], 'val_loss': losses['val']})

    # sample a batch of data
    xb, yb = get_batch('train')

    # evaluate the loss
    logits, loss = model(xb, yb)

    # Check if any inputs, outputs or weights contain NaNs
    if torch.isnan(logits).any() or torch.isnan(loss):
        print("NaN detected!")
        print("Inputs: ", xb)
        print("Outputs: ", logits)
        print("Loss: ", loss)
        for name, param in model.named_parameters():
            if torch.isnan(param).any():
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
    pbar.set_postfix({'train_loss': loss.item(), 'grad_norm': grad_norm})

    optimizer.step()

# generate from the model
print('Generating text...')
model.eval()
context = torch.zeros((1, 1), dtype=torch.long, device=device)
generated_text = model.generate(context, max_new_tokens=2_000)[0].tolist()
print(generated_text)

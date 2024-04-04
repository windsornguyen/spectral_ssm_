# =============================================================================#
# Authors: Windsor Nguyen
# File: benchmark.py
# =============================================================================#

"""Benchmarking on synthetic long-context datasets."""

import torch
from torch.utils.data import DataLoader

from spectral_ssm import experiment
from synthetic import (
    generate_copy,
    generate_adding,
    generate_induction_heads,
    generate_associative_recall,
)

device = torch.device(
    'cuda'
    if torch.cuda.is_available()
    else 'mps'
    if torch.backends.mps.is_available()
    else 'cpu'
)

model = torch.load('checkpoint.pt', map_location=device)
exp = experiment.Experiment(model=model, optimizer=None, device=device)
batch_size = 48
datasets = [
    (
        'Copy',
        generate_copy(num_examples=1000, num_categories=10, copy_len=10, blank_len=5),
    ),
    (
        'Adding', generate_adding(num_examples=1000, sequence_len=10)
    ),
    (
        'Induction Heads',
        generate_induction_heads(num_examples=1000, sequence_len=30, vocab_size=20),
    ),
    (
        'Associative Recall',
        generate_associative_recall(num_examples=1000, sequence_len=30, vocab_size=10),
    ),
]

# Benchmark on each dataset
for dataset_name, dataset in datasets:
    data_loader = DataLoader(dataset, batch_size=batch_size)
    metrics = exp.evaluate(data_loader)

    print(f'Dataset: {dataset_name}')
    print(f'Average Loss: {metrics["loss"]:.4f}')
    print(f'Average Accuracy: {metrics["accuracy"]:.2f}%')
    print('---')

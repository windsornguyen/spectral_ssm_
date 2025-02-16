# Spectral State Space Models

This repository contains a PyTorch implementation for training and evaluating the
spectral state space models and accompanies the paper [Spectral State Space
Models](https://arxiv.org/abs/2312.06837).

The original JAX implementation was
written by Daniel Suo and can be found in this
[repository](https://github.com/google-deepmind/spectral_ssm).

The paper studies sequence modeling for prediction tasks with long range
dependencies. We propose a new formulation for state space models (SSMs) based
on learning linear dynamical systems with the spectral filtering algorithm
(Hazan et al. (2017)). This gives rise to a novel sequence prediction
architecture we call a spectral state space model.

Spectral state space models have two primary advantages. First, they have
provable robustness properties as their performance depends on neither the
spectrum of the underlying dynamics nor the dimensionality of the problem.
Second, these models are constructed with fixed convolutional filters that do
not require learning while still outperforming SSMs in both theory and practice.
The resulting models are evaluated on synthetic dynamical systems and long-range
prediction tasks of various modalities. These evaluations support the
theoretical benefits of spectral filtering for tasks requiring very long range
memory.

## Installation

Clone and navigate to the `spectral_ssm` directory containing `setup.py`.

Optionally, create a virtual environment:

```zsh
python3 -m venv ssm_env
source ssm_env/bin/activate
```

If you want to train on Apple's Metal Performance Shaders (MPS) backend,
you need to install PyTorch Nightly before installing the rest of the requirements:

```zsh
pip install --pre torch torchvision --index-url https://download.pytorch.org/whl/nightly/cpu
```

To install the required packages, run:

```zsh
pip install -r requirements.txt
```

or

```zsh
pip install -e .
```

## Usage

The `example.py` file contains the full training pipeline. `model.py` contains
code for the model itself, including the Spectral Temporal Unit (STU) block.

```zsh
torchrun --nproc_per_node=1 example.py
```

## Citing this work

```latex
@misc{agarwal2024spectral,
      title={Spectral State Space Models},
      author={Naman Agarwal and Daniel Suo and Xinyi Chen and Elad Hazan},
      year={2024},
      eprint={2312.06837},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```

## License and disclaimer

Copyright 2024 DeepMind Technologies Limited

All software is licensed under the Apache License, Version 2.0 (Apache 2.0);
you may not use this file except in compliance with the Apache 2.0 license.
You may obtain a copy of the Apache 2.0 license [here](https://www.apache.org/licenses/LICENSE-2.0).

All other materials are licensed under the Creative Commons Attribution 4.0
International License (CC-BY). You may obtain a copy of the CC-BY license [here](https://creativecommons.org/licenses/by/4.0/legalcode).

Unless required by applicable law or agreed to in writing, all software and
materials distributed here under the Apache 2.0 or CC-BY licenses are
distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND,
either express or implied. See the licenses for the specific language governing
permissions and limitations under those licenses.

This is not an official Google product.

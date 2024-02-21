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

"""Data pipeline for CIFAR10."""

import torch

def preprocess(example: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
  """Preprocess each example in the dataset.

  Args:
    example: A dict with data for a single example.

  Returns:
    A preprocessed example.
  """
  # Floats in [0, 1] instead of ints in [0, 255]
  x = example['image'].float() / 255.0
  x = x.view(-1, x.shape[-1])

  # Reshape label to have single dim
  y = example['label']
  y = torch.tensor(y, dtype=torch.long).view(1)
  data = { 'src': x, 'tgt': y }

  # Normalize by dataset-specific statistics
  means = torch.tensor([0.49139968, 0.48215841, 0.44653091]).view(1, -1)
  stds = torch.tensor([0.24703223, 0.24348513, 0.26158784]).view(1, -1)
  data['src'] = (data['src'] - means) / stds

  return data

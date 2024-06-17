import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import torchvision.models as models
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from tqdm import tqdm

model = models.resnet50()
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)

dataset = datasets.FakeData(
    size=1000,
    transform=transforms.ToTensor())
loader = DataLoader(
    dataset,
    num_workers=1,
    pin_memory=True
)

model.to('cuda')

for data, target in tqdm(loader, desc='Training'):
    data = data.to('cuda', non_blocking=True)
    target = target.to('cuda', non_blocking=True).long()
    optimizer.zero_grad()
    output = model(data)
    loss = criterion(output, target)
    loss.backward()
    optimizer.step()
    
print('Done')

from img2vec_pytorch import Img2Vec
from PIL import Image
import torch

# Initialize Img2Vec with GPU
img2vec = Img2Vec(model='efficientnet_b5', cuda=True)

# Read in images
imgs = []
imgs.append(Image.open('/scratch/gpfs/mn4560/ssm/data/asd1.jpg'))
imgs.append(Image.open('/scratch/gpfs/mn4560/ssm/data/asd2.jpg'))

# Get vectors for each image, returned as torch FloatTensors
vecs = [img2vec.get_vec(img, tensor=True) for img in imgs]

# Combine vectors into a single tensor
vecs_tensor = torch.stack(vecs)

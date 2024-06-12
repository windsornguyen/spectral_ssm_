# import cv2
# import numpy as np
# import torch
# import torch.nn.functional as F
# import torchvision.transforms as transforms
# from PIL import Image
# from sklearn.decomposition import PCA
# from sklearn.neighbors import NearestNeighbors
# import matplotlib.pyplot as plt
# from sklearn.manifold import TSNE


# def best_buddies_nearest_neighbor(features_m, features_q):
#     """
#     Calculate the Best Buddies Pairs (BBPs) to detect matches between two images
#     (Amir et al., 2022, "Deep ViT Features as Dense Visual Descriptors").

#     Let M = {m_i} and Q = {q_j} be sets of binned descriptors from images I_M
#     and I_Q, respectively. The set of BBPs is defined as:

#     BB(M, Q) = {(m, q) | m \in M, q \in Q, NN(m, Q) = q AND NN(q, M) = m}

#     where NN(m, Q) is the nearest neighbor of m in Q under cosine similarity.

#     Args:
#     features_m (torch.Tensor): Feature descriptors from image M, shape (N, D).
#     features_q (torch.Tensor): Feature descriptors from image Q, shape (M, D).

#     Returns:
#     List of tuples: Each tuple contains indices (i, j), where m_i from M and q_j from Q form a BBP.
#     """
#     # Normalize the feature matrices to have unit norm
#     features_m_norm = F.normalize(features_m, p=2, dim=1)
#     features_q_norm = F.normalize(features_q, p=2, dim=1)

#     # Compute the cosine similarity matrix
#     sim_matrix = torch.mm(features_m_norm, features_q_norm.t())

#     # Find nearest neighbors
#     nn_m_to_q = sim_matrix.argmax(dim=1)
#     nn_q_to_m = sim_matrix.argmax(dim=0)

#     # Find mutual nearest neighbors
#     mutual_mask = nn_q_to_m[nn_m_to_q] == torch.arange(len(features_m), device=features_m.device)
#     mutual_indices = torch.stack((torch.where(mutual_mask)[0], nn_m_to_q[mutual_mask])).t()

#     return mutual_indices


# def load_video(video_path):
#     cap = cv2.VideoCapture(video_path)
#     if not cap.isOpened():
#         print("Error opening video file.")
#         return None
#     return cap

# def get_frame_features(frame, model, transform, device):
#     frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#     frame_pil = Image.fromarray(frame_rgb)
#     transformed_frame = transform(frame_pil).unsqueeze(0).to(device)
#     with torch.no_grad():
#         features = model(transformed_frame)
#     return features.cpu().numpy().squeeze()

# def visualize_features_with_pca(features, num_components=3, save_path='pca_plot.png'):
#     pca = PCA(n_components=num_components)
#     transformed_features = pca.fit_transform(features)
#     plt.figure(figsize=(8, 6))
#     plt.scatter(transformed_features[:, 0], transformed_features[:, 1], c='blue')
#     plt.xlabel('PCA Component 1')
#     plt.ylabel('PCA Component 2')
#     plt.title('PCA of Extracted Features')
#     plt.savefig(save_path)  # Save the figure to a file
#     plt.close()  # Close the plot to free up memory


# # Setup
# device = 'cuda' if torch.cuda.is_available() else 'cpu'
# model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14').to(device).eval()
# transform = transforms.Compose([
#     transforms.Resize(256),
#     transforms.CenterCrop(224),
#     transforms.ToTensor(),
#     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
# ])
# video_path = 'Images/video168.mp4'
# cap = load_video(video_path)

# # Process video
# features = []
# while True:
#     ret, frame = cap.read()
#     if not ret:
#         break
#     feature = get_frame_features(frame, model, transform, device)
#     features.append(feature)
# cap.release()

# # Visualize
# features_array = np.array(features)
# visualize_features_with_pca(features_array, save_path='pca_plot.png')

import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms
from torchvision.models import ResNet18_Weights
from torchvision.utils import make_grid
import matplotlib.pyplot as plt


class VisualFeatureExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        # Using weights to load pre-trained models
        weights = ResNet18_Weights.DEFAULT
        base_model = models.resnet18(weights=weights)
        self.features = nn.Sequential(
            *list(base_model.children())[:-2]
        )  # Use features up to the second last layer

    def forward(self, x):
        x = self.features(x)
        return x


def preprocess_image(image_path):
    """Load and preprocess the image."""
    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
            ),
        ]
    )
    image = Image.open(image_path)  # Open the image file
    image = transform(image)  # Apply the transformations
    return image.unsqueeze(0)  # Add batch dimension


def visualize_feature_maps(feature_maps, num_maps=16):
    """Visualize feature maps from the CNN."""
    feature_maps = feature_maps.cpu()  # Ensure the tensor is on CPU
    num_maps = min(
        num_maps, feature_maps.shape[1]
    )  # Limit to the first 'num_maps' feature maps

    fig, axes = plt.subplots(
        nrows=4, ncols=4, figsize=(12, 12)
    )  # Adjust grid size based on your needs
    axes = axes.flatten()

    for idx, ax in enumerate(axes):
        if idx < num_maps:
            # Extract the specific feature map and remove batch dimension
            feature_map = feature_maps[0, idx].detach().numpy()
            ax.imshow(feature_map, cmap='gray')  # Display as grayscale
            ax.axis('off')
        else:
            ax.axis('off')  # Hide unused subplots

    plt.tight_layout()
    plt.savefig('feature_maps.png', dpi=300)
    plt.show()


# Load and preprocess the image
image_path = 'data/asd1.jpg'
image = preprocess_image(image_path)

# Initialize the model and move to appropriate device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
extractor = VisualFeatureExtractor().to(device)
image = image.to(device)

# Extract features
with torch.no_grad():
    feature_maps = extractor(image)

# Visualize some of the feature maps
visualize_feature_maps(feature_maps, num_maps=16)

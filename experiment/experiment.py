import os
import sys
import pandas as pd
import numpy as np
import random
from torch.utils.data import Subset
from collections import defaultdict
import matplotlib.pyplot as plt
import argparse
import torch.optim
import shutil
import torch.backends.cudnn as cudnn
import torch
import torch.optim as optim
import random
import torch.nn as nn
from torch.utils.data import Subset, DataLoader, Dataset
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from scipy.sparse import csr_matrix
import time
import math
from PIL import Image
from tqdm import tqdm
from sklearn.metrics.cluster import normalized_mutual_info_score, silhouette_score
from sklearn.decomposition import PCA
import multiprocessing
from torch.optim import Adam
from torch.nn import CrossEntropyLoss
from torch.profiler import profile, record_function, ProfilerActivity
from torch.cuda.amp import autocast


# Define custom dataset class for Xray images
class ChestXrayDataset(torch.utils.data.Dataset):
    """
    Dataset for Chest X-ray images.

    Args:
        root_dir (str): Path to the directory containing images.
        transform (callable, optional): Transformation pipeline for images.
    """
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = [os.path.join(root_dir, fname) for fname in os.listdir(root_dir) if fname.endswith(('.png', '.jpg', '.jpeg'))]

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        img = Image.open(img_path).convert("L")  # Convert to grayscale
        if self.transform:
            img = self.transform(img)
        return img, idx  # Return image and its index as a pseudo-label

# Initial transform for grayscale images
initial_transform = transforms.Compose([
    transforms.Resize((128, 128)), 
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

# Load dataset for grayscale images
dataset_path = "/gpfs/workdir/islamm/rotated_datasets_without_NoFindings" 
dataset = ChestXrayDataset(root_dir=dataset_path, transform=initial_transform)

# Select 20% of the dataset randomly
dataset_size = len(dataset)
subset_size = int(0.05 * dataset_size)  # 20% of the dataset
random_indices = random.sample(range(dataset_size), subset_size)
subset_dataset = Subset(dataset, random_indices)

# DataLoader for the subset dataset
dataloader = DataLoader(subset_dataset, batch_size=64, shuffle=False, num_workers=4, pin_memory=True)

print(f"Original dataset loaded with {len(dataset)} images.")
print(f"Subset dataset created with {len(subset_dataset)} images.")


# AlexNet model definition
__all__ = [ 'AlexNet', 'alexnet']
 
# (number of filters, kernel size, stride, pad)
CFG = {
    '2012': [(96, 11, 4, 2), 'M', (256, 5, 1, 2), 'M', (384, 3, 1, 1), (384, 3, 1, 1), (256, 3, 1, 1), 'M']
}

class AlexNet(nn.Module):
    def __init__(self, features, num_classes, sobel):
        super(AlexNet, self).__init__()
        self.features = features
        self.classifier = nn.Sequential(nn.Dropout(0.5),
                            nn.Linear(256 * 6 * 6, 4096),
                            nn.ReLU(inplace=True),
                            nn.Dropout(0.5),
                            nn.Linear(4096, 4096),
                            nn.ReLU(inplace=True))

        self.top_layer = nn.Linear(4096, num_classes)
        self._initialize_weights()

        if sobel:
            sobel_filter = nn.Conv2d(1, 2, kernel_size=3, stride=1, padding=1) 
            sobel_filter.weight.data[0, 0].copy_(
                torch.FloatTensor([[1, 0, -1], [2, 0, -2], [1, 0, -1]])
            )
            sobel_filter.weight.data[1, 0].copy_(
                torch.FloatTensor([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
            )
            sobel_filter.bias.data.zero_()
            self.sobel = sobel_filter
            for p in self.sobel.parameters():
                p.requires_grad = False
        else:
            self.sobel = None
# print dimension of all the tensors
    def forward(self, x, return_features=True):
        if self.sobel:
            x = self.sobel(x)
        x = self.features(x)
        if return_features:
            # check the shape of the return features (32*256*6*6, and after flatten the feature is 32*9216) 
            # Kmeans expects column vector as input but the return features are matrix so we need to flatten it
            return x.view(x.size(0), -1) 
        x = x.view(x.size(0), 256 * 6 * 6)
        x = self.classifier(x)
        if self.top_layer:
            x = self.top_layer(x)
        return x


# The x is the input batch of images which is (32*2*224*224) tensor, not a single 1D 

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()

def make_layers_features(cfg, input_dim, bn):
    layers = []
    in_channels = input_dim
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=3, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v[0], kernel_size=v[1], stride=v[2], padding=v[3])
            if bn:
                layers += [conv2d, nn.BatchNorm2d(v[0]), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v[0]
    return nn.Sequential(*layers)

def alexnet(sobel=True, bn=True, out=100):
    dim = 2 + int(not sobel)
    model = AlexNet(make_layers_features(CFG['2012'], dim, bn=bn), out, sobel)
    return model


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def compute_features(dataloader, model, N, device):
    """
    Compute features for the entire dataset and store in a pre-allocated numpy array.
    Args:
        dataloader (DataLoader): DataLoader for the dataset.
        model (nn.Module): CNN model used to extract features.
        N (int): Total number of samples in the dataset.
        device (torch.device): GPU or CPU device.
    Returns:
        np.ndarray: Numpy array of extracted features.
    """
    print("Computing features...")
    model.eval()
    batch_time = AverageMeter()
    end = time.time()

    # Pre-allocate numpy array for features
    for i, (input_tensor, _) in enumerate(tqdm(dataloader, desc="Feature Extraction")):
        input_tensor = input_tensor.to(device)  
        with torch.no_grad():
            aux = model(input_tensor).cpu().numpy()

        # Initialize feature matrix on the first batch
        if i == 0:
            features = np.zeros((N, aux.shape[1]), dtype='float32')

        # Save extracted features
        if i < len(dataloader) - 1:
            features[i * dataloader.batch_size: (i + 1) * dataloader.batch_size] = aux
        else:
            features[i * dataloader.batch_size:] = aux

        batch_time.update(time.time() - end)
        end = time.time()

    print(f"Feature extraction completed. Total time: {batch_time.sum:.2f} seconds")
    return features


def preprocess_features(features, pca_dim):
    """
    Preprocess features using PCA, whitening, and L2 normalization.

    Args:
        features (torch.Tensor): Raw features extracted from the model.
        pca_dim (int): Target dimensionality after PCA.

    Returns:
        torch.Tensor: Preprocessed features ready for clustering.
    """
    device = features.device
    features = features.to(device)

    # Center features
    mean = features.mean(dim=0, keepdim=True)
    features -= mean

    # Compute covariance matrix and eigen decomposition
    cov = features.t().mm(features) / (features.size(0) - 1)
    eigenvalues, eigenvectors = torch.linalg.eigh(cov)

    # Sort eigenvalues and eigenvectors in descending order
    sorted_indices = torch.argsort(eigenvalues, descending=True)
    eigenvalues = eigenvalues[sorted_indices]
    eigenvectors = eigenvectors[:, sorted_indices]

    # Select top components and apply whitening
    components = eigenvectors[:, :pca_dim]
    whitening = torch.diag(1.0 / torch.sqrt(eigenvalues[:pca_dim] + 1e-5))
    features = features.mm(components).mm(whitening)

    # L2 normalization
    features = features / torch.norm(features, dim=1, keepdim=True)

    return features



def kmeans_pytorch(features, num_clusters, num_iters=100, tol=1e-4, verbose=False):
    """
    Custom GPU-based k-means implementation using PyTorch.
    
    Args:
        features (torch.Tensor): Input features of shape (N, D) where N is the number of samples and D is the feature dimension.
        num_clusters (int): Number of clusters.
        num_iters (int): Maximum number of iterations for k-means.
        tol (float): Convergence tolerance for centroids.
        verbose (bool): Whether to print intermediate results.
    
    Returns:
        torch.Tensor: Cluster assignments of shape (N,).
        torch.Tensor: Final centroids of shape (num_clusters, D).
    """
    device = features.device  # Use the same device as features
    N, D = features.shape

    # Initialize centroids randomly from the features
    centroids = features[torch.randint(0, N, (num_clusters,))]

    for i in range(num_iters):
        # Compute distances between features and centroids
        distances = torch.cdist(features, centroids)

        # Assign clusters based on the nearest centroid
        cluster_assignments = torch.argmin(distances, dim=1)

        # Update centroids
        new_centroids = torch.zeros_like(centroids)
        for k in range(num_clusters):
            points_in_cluster = features[cluster_assignments == k]
            if len(points_in_cluster) > 0:
                new_centroids[k] = points_in_cluster.mean(dim=0)
        
        # Check for convergence
        if torch.allclose(centroids, new_centroids, atol=tol):
            if verbose:
                print(f"K-means converged at iteration {i + 1}")
            break
        
        centroids = new_centroids

    return cluster_assignments, centroids


class ReassignedDataset(torch.utils.data.Dataset):
    """
    Dataset with images assigned pseudo-labels from clustering.

    Args:
        subset (torch.utils.data.Subset): Original subset dataset.
        pseudo_labels (list): Cluster assignments (pseudo-labels).
        transform (callable, optional): Transformation pipeline for images.
    """
    def __init__(self, subset, pseudo_labels, transform=None):
        self.subset = subset
        self.pseudo_labels = pseudo_labels
        self.transform = transform

    def __len__(self):
        return len(self.subset)

    def __getitem__(self, idx):
        img, _ = self.subset[idx]  # Get the image from the subset
        pseudo_label = self.pseudo_labels[idx]
        
        # Apply transform only if img is not already a tensor
        if self.transform and not isinstance(img, torch.Tensor):
            img = self.transform(img)
        
        return img, pseudo_label


def cluster_assign(images_lists, dataset, transform=None):
    """
    Creates a dataset from clustering, with clusters as labels.

    Args:
        images_lists (list of list): For each cluster, the list of image indexes belonging to this cluster.
        dataset (torch.utils.data.Subset): Subset dataset used for clustering.
        transform (callable, optional): Image transformation pipeline.

    Returns:
        ReassignedDataset: A dataset with clusters as labels (pseudo-labels).
    """
    assert images_lists is not None

    # Initialize lists for storing image indices and their corresponding pseudo-labels
    pseudolabels = []
    image_indexes = []

    # Iterate through clusters and assign pseudo-labels
    for cluster, images in enumerate(images_lists):
        image_indexes.extend(images)
        pseudolabels.extend([cluster] * len(images))

    # If no transform is provided, use the dataset's transform
    transform = transform or dataset.dataset.transform

    # Create and return the ReassignedDataset with pseudo-labels
    return ReassignedDataset(dataset, pseudolabels, transform)
   
# Define constants
NUM_EPOCHS = 20  # Total number of epochs
NUM_CLUSTERS = 100  # Number of clusters for k-means
PCA_DIM = 128  # Dimensionality for PCA
BATCH_SIZE = 64  # Batch size for pseudo-labeled training
LEARNING_RATE = 0.05  # Learning rate for SGD
WEIGHT_DECAY = 1e-5  # Weight decay for regularization

# Instantiate the model
model = alexnet(sobel=True, bn=True, out=100) 
torch.backends.cudnn.benchmark = True  # Enable CuDNN for performance optimization
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = nn.DataParallel(model)
model = model.to(device)

# Define optimizer and loss
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=0.9, weight_decay=WEIGHT_DECAY)

# Initialize metrics collection
training_metrics = {
    "epoch": [],
    "nmi": [],
    "average_loss": [],
}

# Initialize previous cluster assignments for NMI calculation
prev_cluster_assignments = None
# Save model weights at the end of training loop
model_save_path = "/gpfs/workdir/islamm/alexnet_trained.pth"

for epoch in range(NUM_EPOCHS):
    print(f"\n### Epoch {epoch + 1}/{NUM_EPOCHS} ###")

    # Extract features from the dataset
    print("Extracting features...")
    features = compute_features(dataloader, model, len(subset_dataset), device)  # Get features from hidden layers
    features = torch.tensor(features, device=device, dtype=torch.float32)
    np.save("/gpfs/workdir/islamm/database_features.npy", features.cpu().numpy())
    print("Database features saved to '/gpfs/workdir/islamm/database_features.npy'")

    # Preprocess features
    print("Preprocessing features...")
    preprocessed_features = preprocess_features(features, pca_dim=PCA_DIM)

    # Perform custom k-means clustering
    print(f"Clustering into {NUM_CLUSTERS} clusters...")
    cluster_assignments, _ = kmeans_pytorch(preprocessed_features, NUM_CLUSTERS, num_iters=100, tol=1e-4, verbose=True)

    # Convert cluster assignments to images_lists
    images_lists = [[] for _ in range(NUM_CLUSTERS)]
    for idx, cluster_id in enumerate(cluster_assignments):
        images_lists[cluster_id].append(idx)

    # Assign pseudo-labels to dataset
    print("Assigning pseudo-labels...")
    pseudo_dataset = cluster_assign(images_lists, subset_dataset, transform=initial_transform)
    pseudo_dataloader = DataLoader(pseudo_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

    # Train CNN with pseudo-labeled dataset
    print("Training CNN with pseudo-labels...")
    model.train()
    running_loss = 0.0
    for imgs, pseudo_labels in tqdm(pseudo_dataloader, desc="Training"):
        imgs, pseudo_labels = imgs.to(device), pseudo_labels.to(device)

        # Forward pass (using features from the conv5 layer)
        features = model(imgs, return_features=True)  # Get hidden features (shape: batch_size, 256, 6, 6)

        # Flatten the conv5 features
        features = features.view(features.size(0), -1) 

        # Compute the loss using the flattened features
        loss = criterion(features, pseudo_labels)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    avg_loss = running_loss / len(pseudo_dataloader)
    print(f"Epoch {epoch + 1} Average Loss: {avg_loss:.4f}")
    training_metrics["average_loss"].append(avg_loss)  # Store average loss

    # Evaluate clustering performance
    if prev_cluster_assignments is not None:
        # Calculate NMI between current and previous cluster assignments
        nmi = normalized_mutual_info_score(prev_cluster_assignments.cpu().numpy(), cluster_assignments.cpu().numpy())
        print(f"Epoch {epoch + 1} NMI: {nmi:.4f}")
        training_metrics["nmi"].append(nmi)  # Store NMI
    else:
        training_metrics["nmi"].append(None)

    # Save the current epoch
    training_metrics["epoch"].append(epoch + 1)

    # Save current cluster assignments for the next epoch
    prev_cluster_assignments = cluster_assignments

# Save metrics for visualization
metrics_df = pd.DataFrame(training_metrics)
metrics_df.to_csv("/gpfs/workdir/islamm/training_metrics.csv", index=False)
print("Training metrics saved to 'training_metrics.csv'")

# Save the trained model
torch.save(model.state_dict(), model_save_path)
print(f"Model weights saved to {model_save_path}")

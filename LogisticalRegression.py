import os
import sys
import pandas as pd
import numpy as np
import random
import faiss
import torch.distributed as dist
from torch.utils.data import Subset
from collections import defaultdict
import matplotlib.pyplot as plt
import argparse
import shutil
import torch.backends.cudnn as cudnn
import torch
import cv2
import torch.optim as optim
import random
import torch.nn as nn
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Subset, DistributedSampler, Dataset
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
from torch.cuda.amp import autocast, GradScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

class ChestXrayDataset(Dataset):
    """
    Chest X-ray dataset with:
    - Sobel filtering (2-channel output)
    - Random rotations (-R° to +R° augmentation)
    
    Args:
        root_dir (str): Path to dataset folder.
        use_sobel (bool): Apply Sobel filtering.
        num_rotations (int): Number of rotated versions per image.
        rotation_range (float): Maximum absolute rotation angle in degrees.
    """
    
    def __init__(self, root_dir, use_sobel=True, num_rotations=5, rotation_range=360):
        self.root_dir = root_dir
        self.use_sobel = use_sobel
        self.num_rotations = num_rotations
        self.rotation_range = rotation_range  # Maximum rotation angle
        self.image_paths = [os.path.join(root_dir, fname) for fname in os.listdir(root_dir) 
                            if fname.endswith(('.png', '.jpg', '.jpeg'))]
    
    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # Load image & convert to grayscale
        img_path = self.image_paths[idx]
        img = Image.open(img_path).convert("L")  # Convert to grayscale

        # Resize to 64x64
        img = img.resize((64, 64), Image.BILINEAR)

        # Apply random rotations
        rotated_images = self.apply_random_rotations(img)  # List of rotated PIL images

        # Apply Sobel filter or convert to tensor
        if self.use_sobel:
            processed_images = [self.apply_sobel_filter(rotated) for rotated in rotated_images]  # (C=2, 64, 64)
        else:
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5], std=[0.5])
            ])
            processed_images = [transform(rotated) for rotated in rotated_images]  # (C=1, 64, 64)

        # Stack all processed images together (num_rotations+1, C, 64, 64)
        stacked_images = torch.stack(processed_images)  # Shape: (num_rotations+1, C, 64, 64)
        
        return stacked_images, idx  # Returns tensor (num_rotations+1 images) with ID

    def apply_random_rotations(self, img):
        """
        Generate multiple rotated versions of the image.
        
        Args:
            img (PIL.Image): Input image.
            
        Returns:
            list: List of rotated PIL images, including the original.
        """
        angles = [random.uniform(-self.rotation_range, self.rotation_range) for _ in range(self.num_rotations)]
        return [img] + [img.rotate(angle, resample=Image.BILINEAR) for angle in angles]  # Include original image

    def apply_sobel_filter(self, img):
        """
        Apply Sobel filter to generate a 2-channel edge-detected image.
        
        Args:
            img (PIL.Image): Input image.
            
        Returns:
            Tensor: Sobel-filtered 2-channel image (C=2, H=64, W=64).
        """
        img_np = np.array(img, dtype=np.float32) / 255.0  # Normalize to range [0, 1]

        # Apply Sobel filters
        sobel_x = cv2.Sobel(img_np, cv2.CV_32F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(img_np, cv2.CV_32F, 0, 1, ksize=3)

        # Normalize gradients to range [-1, 1]
        sobel_x = (sobel_x - sobel_x.mean()) / (sobel_x.std() + 1e-6)
        sobel_y = (sobel_y - sobel_y.mean()) / (sobel_y.std() + 1e-6)

        # Stack into 2-channel tensor
        sobel_filtered = np.stack((sobel_x, sobel_y), axis=0)  # Shape: (2, 64, 64)
        return torch.tensor(sobel_filtered, dtype=torch.float32)

# -------------------
# AlexNet Model Definition
# -------------------

CFG = {
    '2012': [(96, 11, 4, 2), 'M', (256, 5, 1, 2), 'M', 
             (384, 3, 1, 1), (384, 3, 1, 1), (256, 3, 1, 1), 'M']
}

class AlexNet(nn.Module):
    def __init__(self, features, num_classes, input_dim=2): 
        super(AlexNet, self).__init__()
        self.features = features
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True)
        )
        self.top_layer = nn.Linear(4096, num_classes)
        self._initialize_weights()

    # def forward(self, x, return_features=True):
    #     x = self.features(x)  # Ensure the first conv layer correctly handles 2-channel input
    #     if return_features:
    #         return x.view(x.size(0), -1)  # Flatten features for clustering
    #     x = x.view(x.size(0), 256 * 6 * 6)
    #     x = self.classifier(x)
    #     if self.top_layer:
    #         x = self.top_layer(x)
    #     return x

    def forward(self, x, return_features=True):
        if x.dim() == 5: 
            batch_size, num_rotations, C, H, W = x.shape
            x = x.view(batch_size * num_rotations, C, H, W)  

        print(f"Input image shape: {x.shape}") 
        x = self.features(x)
        print(f"Feature map shape before view(): {x.shape}")

        if return_features:
            return x.view(x.size(0), -1) 

        x = x.view(x.size(0), 256 * 6 * 6)
        x = self.classifier(x)

        if self.top_layer:
            x = self.top_layer(x)

        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, (2. / n) ** 0.5)
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
    in_channels = input_dim  # Ensure the correct number of input channels (2 for Sobel)
    
    for v in cfg:
        if v == 'M':
            layers.append(nn.MaxPool2d(kernel_size=3, stride=2))
        else:
            conv2d = nn.Conv2d(in_channels, v[0], kernel_size=v[1], stride=v[2], padding=v[3])
            if bn:
                layers.extend([conv2d, nn.BatchNorm2d(v[0]), nn.ReLU(inplace=True)])
            else:
                layers.extend([conv2d, nn.ReLU(inplace=True)])
            in_channels = v[0]  # Update in_channels for the next layer

    return nn.Sequential(*layers)

def alexnet(sobel=True, bn=True, out=200):
    """
    Create an AlexNet model with optional Sobel filtering.

    Args:
        sobel (bool): Apply Sobel filtering before passing to AlexNet.
        bn (bool): Use Batch Normalization.
        out (int): Number of output classes.

    Returns:
        AlexNet model with optional Sobel filtering.
    """
    input_dim = 2 if sobel else 1 
    model = AlexNet(make_layers_features(CFG['2012'], input_dim, bn=bn), out, sobel)
    return model


# # AlexNet model definition
# __all__ = [ 'AlexNet', 'alexnet']
 
# # (number of filters, kernel size, stride, pad)
# CFG = {
#     '2012': [(96, 11, 4, 2), 'M', (256, 5, 1, 2), 'M', (384, 3, 1, 1), (384, 3, 1, 1), (256, 3, 1, 1), 'M']
# }

# class AlexNet(nn.Module):
#     def __init__(self, features, num_classes, sobel):
#         super(AlexNet, self).__init__()
#         self.features = features
#         self.classifier = nn.Sequential(nn.Dropout(0.5),
#                             nn.Linear(256 * 6 * 6, 4096),
#                             nn.ReLU(inplace=True),
#                             nn.Dropout(0.5),
#                             nn.Linear(4096, 4096),
#                             nn.ReLU(inplace=True))

#         self.top_layer = nn.Linear(4096, num_classes)
#         self._initialize_weights()

#         if sobel:
#             sobel_filter = nn.Conv2d(1, 2, kernel_size=3, stride=1, padding=1) 
#             sobel_filter.weight.data[0, 0].copy_(
#                 torch.FloatTensor([[1, 0, -1], [2, 0, -2], [1, 0, -1]])
#             )
#             sobel_filter.weight.data[1, 0].copy_(
#                 torch.FloatTensor([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
#             )
#             sobel_filter.bias.data.zero_()
#             self.sobel = sobel_filter
#             for p in self.sobel.parameters():
#                 p.requires_grad = False
#         else:
#             self.sobel = None
# # print dimension of all the tensors
#     def forward(self, x, return_features=True):
#         if self.sobel:
#             x = self.sobel(x)
#         x = self.features(x)
#         if return_features:
#             # check the shape of the return features (32*256*6*6, and after flatten the feature is 32*9216) 
#             # Kmeans expects column vector as input but the return features are matrix so we need to flatten it
#             return x.view(x.size(0), -1) 
#         x = x.view(x.size(0), 256 * 6 * 6)
#         x = self.classifier(x)
#         if self.top_layer:
#             x = self.top_layer(x)
#         return x


# # The x is the input batch of images which is (32*2*224*224) tensor, not a single 1D 

#     def _initialize_weights(self):
#         for m in self.modules():
#             if isinstance(m, nn.Conv2d):
#                 n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
#                 m.weight.data.normal_(0, math.sqrt(2. / n))
#                 if m.bias is not None:
#                     m.bias.data.zero_()
#             elif isinstance(m, nn.BatchNorm2d):
#                 m.weight.data.fill_(1)
#                 m.bias.data.zero_()
#             elif isinstance(m, nn.Linear):
#                 m.weight.data.normal_(0, 0.01)
#                 m.bias.data.zero_()

# def make_layers_features(cfg, input_dim, bn):
#     layers = []
#     in_channels = input_dim
#     for v in cfg:
#         if v == 'M':
#             layers += [nn.MaxPool2d(kernel_size=3, stride=2)]
#         else:
#             conv2d = nn.Conv2d(in_channels, v[0], kernel_size=v[1], stride=v[2], padding=v[3])
#             if bn:
#                 layers += [conv2d, nn.BatchNorm2d(v[0]), nn.ReLU(inplace=True)]
#             else:
#                 layers += [conv2d, nn.ReLU(inplace=True)]
#             in_channels = v[0]
#     return nn.Sequential(*layers)

# def alexnet(sobel=True, bn=True, out=100):
#     dim = 2 + int(not sobel)
#     model = AlexNet(make_layers_features(CFG['2012'], dim, bn=bn), out, sobel)
#     return model


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

# def compute_features(dataloader, model, N, device):
#     model.eval()
#     features = []
#     with torch.no_grad():
#         for i, (imgs, _) in enumerate(tqdm(dataloader, desc=f"Feature Extraction (GPU {device})")):
#             print(f"GPU {device}: Processing batch {i + 1}/{len(dataloader)}") 
#             imgs = imgs.to(device, non_blocking=True)

#             # Reshape input from (batch_size, 6, C, H, W) → (batch_size * 6, C, H, W)
#             batch_size, num_rotations, C, H, W = imgs.shape  # Expecting (batch, 6, 2, 64, 64)
#             imgs = imgs.view(batch_size * num_rotations, C, H, W)  # Flatten rotations into batch

#             # Use model.module.features() if using DistributedDataParallel (DDP)
#             features_extracted = model.module.features(imgs).cpu() if isinstance(model, DDP) else model.features(imgs).cpu()
            
#             features.append(features_extracted)

#     return torch.cat(features)

def compute_features(dataloader, model, N, device):
    """
    Extracts features from the model and stores them in a tensor.
    Saves the latest feature extraction results like in DeepCluster.

    Args:
        dataloader (DataLoader): DataLoader for dataset.
        model (nn.Module): Model used to extract features.
        N (int): Total number of samples in the dataset.
        device (torch.device): GPU or CPU device.

    Returns:
        torch.Tensor: Extracted features tensor of shape (N, feature_dim).
    """
    model.eval()
    features = []
    
    with torch.no_grad():
        for i, (imgs, _) in enumerate(tqdm(dataloader, desc=f"Feature Extraction (GPU {device})")):
            print(f"GPU {device}: Processing batch {i + 1}/{len(dataloader)}") 
            
            imgs = imgs.to(device, non_blocking=True)

            # Reshape input from (batch_size, 6, C, H, W) → (batch_size * 6, C, H, W)
            batch_size, num_rotations, C, H, W = imgs.shape
            imgs = imgs.view(batch_size * num_rotations, C, H, W) 

            # Use model.module.features() if using DDP
            features_extracted = model.module.features(imgs) if isinstance(model, DDP) else model.features(imgs)

            # Flatten the feature maps from (B, C, H, W) → (B, C*H*W)
            features_extracted = features_extracted.view(features_extracted.shape[0], -1).cpu()

            features.append(features_extracted)

    # Concatenate all extracted features
    features = torch.cat(features)

    # Ensure all GPUs finish before saving
    dist.barrier()

    # Save only the latest features
    if dist.get_rank() == 0:  
        np.save("/gpfs/workdir/islamm/alexnet_features.npy", features.numpy())
        print(f"GPU {device}: Saved extracted features to '/gpfs/workdir/islamm/alexnet_features.npy'")

    return features

# def compute_features(dataloader, model, N, device):
#     """
#     Compute features for the entire dataset and store in a pre-allocated numpy array.
#     Args:
#         dataloader (DataLoader): DataLoader for the dataset.
#         model (nn.Module): CNN model used to extract features.
#         N (int): Total number of samples in the dataset.
#         device (torch.device): GPU or CPU device.
#     Returns:
#         np.ndarray: Numpy array of extracted features.
#     """
#     print("Computing features...")
#     model.eval()
#     batch_time = AverageMeter()
#     end = time.time()

#     # Pre-allocate numpy array for features
#     for i, (input_tensor, _) in enumerate(tqdm(dataloader, desc="Feature Extraction")):
#         input_tensor = input_tensor.to(device)  
#         with torch.no_grad():
#             aux = model(input_tensor).cpu().numpy()

#         # Initialize feature matrix on the first batch
#         if i == 0:
#             features = np.zeros((N, aux.shape[1]), dtype='float32')

#         # Save extracted features
#         if i < len(dataloader) - 1:
#             features[i * dataloader.batch_size: (i + 1) * dataloader.batch_size] = aux
#         else:
#             features[i * dataloader.batch_size:] = aux

#         batch_time.update(time.time() - end)
#         end = time.time()

#     print(f"Feature extraction completed. Total time: {batch_time.sum:.2f} seconds")
#     return features


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

# def kmeans_pytorch(features, num_clusters, num_iters=100, tol=1e-4, verbose=False):
#     """
#     Custom GPU-based k-means implementation using PyTorch.
    
#     Args:
#         features (torch.Tensor): Input features of shape (N, D) where N is the number of samples and D is the feature dimension.
#         num_clusters (int): Number of clusters.
#         num_iters (int): Maximum number of iterations for k-means.
#         tol (float): Convergence tolerance for centroids.
#         verbose (bool): Whether to print intermediate results.
    
#     Returns:
#         torch.Tensor: Cluster assignments of shape (N,).
#         torch.Tensor: Final centroids of shape (num_clusters, D).
#     """
#     device = features.device # Get the device (GPU or CPU) of the features
#     N, D = features.shape

#     # Initialize centroids randomly from the features
#     centroids = features[torch.randint(0, N, (num_clusters,))]

#     for i in range(num_iters):
#         # Compute distances between features and centroids
#         distances = torch.cdist(features, centroids)

#         # Assign clusters based on the nearest centroid
#         cluster_assignments = torch.argmin(distances, dim=1)

#         # Update centroids
#         new_centroids = torch.zeros_like(centroids)
#         for k in range(num_clusters):
#             points_in_cluster = features[cluster_assignments == k]
#             if len(points_in_cluster) > 0:
#                 new_centroids[k] = points_in_cluster.mean(dim=0)
        
#         # Check for convergence
#         if torch.allclose(centroids, new_centroids, atol=tol):
#             if verbose:
#                 print(f"K-means converged at iteration {i + 1}")
#             break
        
#         centroids = new_centroids

#     return cluster_assignments, centroids

def run_kmeans(x, nmb_clusters, verbose=False):
    """
    Runs FAISS-GPU k-means clustering.

    Args:
        x (torch.Tensor or numpy.ndarray): Feature matrix `[num_samples, feature_dim]`
        nmb_clusters (int): Number of clusters.
        verbose (bool): Print FAISS loss evolution.

    Returns:
        torch.Tensor: Cluster assignments `[num_samples]`.
        float: Final k-means loss.
    """
    if isinstance(x, torch.Tensor):
        x = x.cpu().numpy()  # Ensure numpy array format
    
    n_data, d = x.shape

    # Initialize FAISS-GPU resources
    res = faiss.StandardGpuResources()

    # FAISS k-means clustering settings
    clus = faiss.Clustering(d, nmb_clusters)
    clus.seed = np.random.randint(1234)
    clus.niter = 20  # Number of iterations
    clus.max_points_per_centroid = 10000000  # Handle large datasets

    # Configure GPU for FAISS
    flat_config = faiss.GpuIndexFlatConfig()
    flat_config.useFloat16 = False
    flat_config.device = torch.cuda.current_device()

    # Create GPU FAISS Index
    index = faiss.GpuIndexFlatL2(res, d, flat_config)

    # Perform clustering
    clus.train(x, index)

    # Assign clusters
    _, I = index.search(x, 1)  # Find nearest centroids

    # Compute clustering loss manually (Sum of squared distances)
    centroids = faiss.vector_to_array(clus.centroids).reshape(nmb_clusters, d)
    final_loss = np.mean(np.linalg.norm(x - centroids[I.squeeze()], axis=1) ** 2)

    if verbose:
        print(f"Final K-Means Loss: {final_loss}")

    return torch.tensor(I.flatten(), dtype=torch.long), final_loss  

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




# -------------------
# Training Pipeline
# -------------------


# # Define constants
# NUM_EPOCHS = 20  # Total number of epochs
# NUM_CLUSTERS = 200  # Number of clusters for k-means
# PCA_DIM = 128  # Dimensionality for PCA
# BATCH_SIZE = 256  # Batch size for pseudo-labeled training
# LEARNING_RATE = 0.001  # Learning rate for Adam
# WEIGHT_DECAY = 1e-5  # Weight decay for regularization
# SUBSET_SIZE = 10000

# # Load Full Dataset
# dataset_path = "/gpfs/workdir/islamm/datasets"
# full_dataset = ChestXrayDataset(root_dir=dataset_path, use_sobel=True)  # Use Sobel filtering

# # Take a random subset of the dataset
# subset_indices = random.sample(range(len(full_dataset)), SUBSET_SIZE)
# subset_dataset = Subset(full_dataset, subset_indices)

# # Load Data
# dataloader = DataLoader(
#     subset_dataset, 
#     batch_size=BATCH_SIZE, 
#     shuffle=True, 
#     num_workers=4,  
#     pin_memory=True, 
#     persistent_workers=True,  
#     prefetch_factor=4  
# )

# print(f"Subset dataset loaded with {len(subset_dataset)} images. After augmentation: {len(subset_dataset) * 6} images.")
# # Define Model
# model = alexnet(sobel=True, bn=True, out=NUM_CLUSTERS)
# torch.backends.cudnn.benchmark = True
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# # Use DataParallel for multi-GPU support
# if torch.cuda.device_count() > 1:
#     print(f"Using {torch.cuda.device_count()} GPUs for training.")
#     model = nn.DataParallel(model)

# model = model.to(device)

# # Optimizer, Loss, and Mixed Precision
# criterion = nn.CrossEntropyLoss()
# optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)  # Changed to Adam
# scaler = GradScaler()  # For mixed precision

# # Metrics collection
# training_metrics = {
#     "epoch": [],
#     "nmi": [],
#     "average_loss": [],
# }

# prev_cluster_assignments = None
# model_save_path = "/gpfs/workdir/islamm/alexnet_trained.pth"

# # Start GPU Profiling
# with profile(
#     activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
#     schedule=torch.profiler.schedule(wait=1, warmup=1, active=5, repeat=1),
#     on_trace_ready=torch.profiler.tensorboard_trace_handler("/gpfs/workdir/islamm/profiler_logs"),
#     record_shapes=True,
#     profile_memory=True,
#     with_stack=True
# ) as prof:

#     # Training Loop
#     for epoch in range(NUM_EPOCHS):
#         print(f"\n### Epoch {epoch + 1}/{NUM_EPOCHS} ###")

#         # Feature Extraction Profiling
#         with record_function("Feature Extraction"):
#             print("Extracting features...")
#             features = compute_features(dataloader, model, len(subset_dataset), device)
#             features = torch.tensor(features, device=device, dtype=torch.float32)
#             np.save("/gpfs/workdir/islamm/alexnet_features.npy", features.cpu().numpy())
#             print("Database features saved to '/gpfs/workdir/islamm/alexnet_features.npy'")

#         # Preprocess features before clustering
#         print("Preprocessing features...")
#         features = preprocess_features(features, pca_dim=PCA_DIM)

#         # Perform FAISS K-means Clustering
#         print(f"Clustering {features.shape[0]} samples into {NUM_CLUSTERS} clusters using FAISS...")
#         cluster_assignments, kmeans_loss = run_kmeans(features.cpu().numpy(), NUM_CLUSTERS, verbose=True)

#         # Convert cluster assignments to tensors
#         cluster_assignments = torch.tensor(cluster_assignments, dtype=torch.long, device=device)

#         # Convert cluster assignments to pseudo-labels for training
#         images_lists = [[] for _ in range(NUM_CLUSTERS)]
#         for idx, cluster_id in enumerate(cluster_assignments):
#             images_lists[cluster_id].append(idx)

#         pseudo_dataset = cluster_assign(images_lists, subset_dataset, transform=transform)
#         pseudo_dataloader = DataLoader(pseudo_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)

#         # Training Profiling
#         with record_function("CNN Training"):
#             print("Training CNN with pseudo-labels...")
#             model.train()
#             running_loss = 0.0

#             scaler = torch.cuda.amp.GradScaler()  # Helps stabilize training

#             for imgs, _ in tqdm(dataloader, desc="Training"):
#                 imgs = imgs.view(-1, 1, 64, 64).to(device)

#                 with autocast():  # Enables mixed precision
#                     features = model.features(imgs)  
#                     features = features.view(features.shape[0], -1)
#                     outputs = model.classifier(features)
#                     loss = criterion(outputs, torch.randint(0, NUM_CLUSTERS, (outputs.shape[0],)).to(device))

#                 optimizer.zero_grad()
#                 scaler.scale(loss).backward()
#                 scaler.step(optimizer)
#                 scaler.update()

#                 running_loss += loss.item()

#             avg_loss = running_loss / len(pseudo_dataloader)
#             print(f"Epoch {epoch + 1} Average Loss: {avg_loss:.4f}")
#             training_metrics["average_loss"].append(avg_loss)

#         # Evaluate Clustering Performance
#         with record_function("Evaluating Clustering"):
#             if prev_cluster_assignments is not None:
#                 nmi = normalized_mutual_info_score(prev_cluster_assignments.cpu().numpy(), cluster_assignments.cpu().numpy())
#                 print(f"Epoch {epoch + 1} NMI: {nmi:.4f}")
#                 training_metrics["nmi"].append(nmi)
#             else:
#                 training_metrics["nmi"].append(None)

#             training_metrics["epoch"].append(epoch + 1)
#             prev_cluster_assignments = cluster_assignments

#         # Step profiler (capture next iteration)
#         prof.step()

# # Save Metrics and Model
# metrics_df = pd.DataFrame(training_metrics)
# metrics_df.to_csv("/gpfs/workdir/islamm/training_metrics1.csv", index=False)
# print("Training metrics saved to '/gpfs/workdir/islamm/training_metrics1.csv'")

# torch.save(model.state_dict(), model_save_path)
# print(f"Model weights saved to {model_save_path}")


# -------------------
# Training Parameters
# -------------------
NUM_EPOCHS = 100
NUM_CLUSTERS = 200
PCA_DIM = 128
BATCH_SIZE = 256
LEARNING_RATE = 0.001
WEIGHT_DECAY = 1e-5
WORLD_SIZE = torch.cuda.device_count() 

# -------------------
# Distributed Training Setup
# -------------------
def setup(rank, world_size):
    """ Initialize the distributed training environment. """
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = "29500"
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

def cleanup():
    """ Cleanup distributed training resources. """
    dist.destroy_process_group()


# -------------------
# Training Function
# -------------------
def train(rank, world_size):
    setup(rank, world_size)

    # Load Full Dataset
    dataset_path = "/gpfs/workdir/islamm/datasets"
    full_dataset = ChestXrayDataset(root_dir=dataset_path, use_sobel=True, num_rotations=5)

    # Take the first 20% of the dataset
    subset_size = int(0.2 * len(full_dataset))  # Compute 20% of the dataset
    subset_indices = list(range(subset_size))   # Select first 20% of the dataset
    subset_dataset = Subset(full_dataset, subset_indices)

    print(f"Using first 20% of dataset: {subset_size} samples.")

    # Create Distributed Sampler for Multi-GPU Efficiency
    train_sampler = DistributedSampler(subset_dataset, num_replicas=world_size, rank=rank, shuffle=True)

    dataloader = DataLoader(
        subset_dataset,
        batch_size=BATCH_SIZE // world_size,  
        shuffle=False,
        num_workers=4,  
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2,
        sampler=train_sampler
    )

    # Load Model
    model = alexnet(sobel=True, bn=True, out=NUM_CLUSTERS).to(rank)
    model = DDP(model, device_ids=[rank], output_device=rank)

    # Optimizer & Loss Function
    criterion = nn.CrossEntropyLoss().to(rank)
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scaler = torch.amp.GradScaler("cuda") 

    # Training Metrics
    training_metrics = {"epoch": [], "nmi": [], "average_loss": []}
    prev_cluster_assignments = None
    model_save_path = "/gpfs/workdir/islamm/alexnet_trained.pth"

    # Start GPU Profiling
    with profile(
        activities=[ProfilerActivity.xCPU, ProfilerActivity.CUDA],
        schedule=torch.profiler.schedule(wait=1, warmup=1, active=5, repeat=1),
        on_trace_ready=torch.profiler.tensorboard_trace_handler("/gpfs/workdir/islamm/profiler_logs"),
        record_shapes=True,
        profile_memory=True,
        with_stack=True
    ) as prof:
        accuracies = []
        for epoch in range(NUM_EPOCHS):
            print(f"\n### Epoch {epoch + 1}/{NUM_EPOCHS} (GPU {rank}) ###")
            train_sampler.set_epoch(epoch)

            # Feature Extraction
            dist.barrier()
            with record_function("Feature Extraction"):
                print(f"GPU {rank}: Extracting features...")
                with torch.no_grad(): 
                    features = compute_features(dataloader, model, len(subset_dataset), rank)
                features = features.view(features.shape[0], -1)  # Flatten features for clustering

            dist.barrier()
            if rank == 0:
                np.save(f"/gpfs/workdir/islamm/alexnet_features.npy", features.cpu().numpy())
                print(f"GPU {rank}: Feature extraction saved!")

            # Preprocess Features
            features = preprocess_features(features, pca_dim=PCA_DIM)

            # Perform FAISS K-means Clustering
            dist.barrier()
            if rank == 0:
                print(f"GPU {rank}: Running FAISS K-means clustering...")
                cluster_assignments_np, _ = run_kmeans(features.cpu().numpy(), NUM_CLUSTERS, verbose=False)
                cluster_assignments = torch.tensor(cluster_assignments_np, dtype=torch.long, device=rank)
            else:
                cluster_assignments = torch.full((features.shape[0],), -1, dtype=torch.long, device=rank)  # ✅ Initialize properly

            # 🔹 Synchronize across all GPUs
            dist.broadcast(cluster_assignments, src=0)

            # 逻辑回归部分
            if rank == 0:
                # 加载特征数据
                features = np.load("/gpfs/workdir/islamm/alexnet_features.npy")
                # 准备标签数据
                labels = cluster_assignments.cpu().numpy()

                # 划分训练集和测试集
                X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

                # 创建逻辑回归模型
                logreg = LogisticRegression(max_iter=1000)
                # 训练模型
                logreg.fit(X_train, y_train)

                # 在测试集上进行预测
                y_pred = logreg.predict(X_test)
                # 计算准确率
                accuracy = accuracy_score(y_test, y_pred)
                print(f"logistic regression accurancy: {accuracy:.4f}")

            # Convert to pseudo-labels
            images_lists = [[] for _ in range(NUM_CLUSTERS)]
            for idx, cluster_id in enumerate(cluster_assignments.cpu().numpy()):
                images_lists[cluster_id].append(idx)

            dist.barrier()
            print(f"GPU {rank}: Preparing pseudo-labeled dataset...")
            pseudo_dataset = cluster_assign(images_lists, subset_dataset, transform=transforms.ToTensor())
            pseudo_dataloader = DataLoader(pseudo_dataset, batch_size=BATCH_SIZE // world_size, shuffle=True,
                                           num_workers=2, pin_memory=True)

            # CNN Training
            with record_function("CNN Training"):
                print(f"GPU {rank}: Training CNN with pseudo-labels...")
                model.train()
                running_loss = 0.0

                for imgs, pseudo_labels in tqdm(pseudo_dataloader, desc=f"Training on GPU {rank}"):
                    imgs, pseudo_labels = imgs.to(rank), pseudo_labels.to(rank)

                    with autocast():  # Mixed Precision
                        outputs = model(imgs, return_features=False)
                        loss = criterion(outputs, pseudo_labels)

                    optimizer.zero_grad()
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()

                    running_loss += loss.item()

                avg_loss = running_loss / len(pseudo_dataloader)
                print(f"GPU {rank}: Epoch {epoch + 1} Average Loss: {avg_loss:.4f}")
                training_metrics["average_loss"].append(avg_loss)

            training_metrics["epoch"].append(epoch + 1)
            prev_cluster_assignments = cluster_assignments
            prof.step()

    cleanup()

# -------------------
# Multi-GPU Training Launcher
# -------------------
if __name__ == "__main__":
    WORLD_SIZE = torch.cuda.device_count()
    print(f"Launching multi-GPU training on {WORLD_SIZE} GPUs...")
    torch.multiprocessing.set_start_method("spawn", force=True)

    mp.spawn(train, args=(WORLD_SIZE,), nprocs=WORLD_SIZE, join=True)
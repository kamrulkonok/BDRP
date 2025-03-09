import os
import sys
import cv2
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
import wandb
import time
import math

from collections import defaultdict
import argparse
import shutil

from scipy.sparse import csr_matrix
from PIL import Image
from tqdm import tqdm
import multiprocessing

from sklearn.metrics.cluster import normalized_mutual_info_score, silhouette_score
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

import torch
import torch.distributed as dist
from torch.utils.data import Subset
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.nn as nn
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler, Dataset
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.optim import Adam
from torch.nn import CrossEntropyLoss
from torch.profiler import profile, record_function, ProfilerActivity
from torch.cuda.amp import autocast, GradScaler

# ---------------------------
# Global Results Directory
# ---------------------------
RESULT_DIR = "/gpfs/workdir/islamm/results"
if not os.path.exists(RESULT_DIR):
    os.makedirs(RESULT_DIR)

# Set random seed for reproducibility
RANDOM_SEED = 42
random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

# ---------------------------
# Dataset Definition
# ---------------------------
class ChestXrayDataset(torch.utils.data.Dataset):
    """
    Dataset for Chest X-ray images with random rotations.

    Args:
        root_dir (str): Path to the directory containing images.
        transform (callable, optional): Transformation pipeline for images.
        num_rotations (int): Number of random rotations to apply to each image.
    """
    def __init__(self, root_dir, transform=None, num_rotations=5):
        self.root_dir = root_dir
        self.transform = transform
        self.num_rotations = num_rotations
        self.image_paths = [os.path.join(root_dir, fname) for fname in os.listdir(root_dir) if fname.endswith(('.png', '.jpg', '.jpeg'))]

    def __len__(self):
        return len(self.image_paths) * self.num_rotations

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        img = Image.open(img_path).convert("L")  # Convert to grayscale

        # Apply 5 random rotations between -360 and +360 degrees
        rotated_images = []
        for _ in range(self.num_rotations):
            angle = random.uniform(-360, 360)
            rotated_img = transforms.functional.rotate(img, angle)
            if self.transform:
                rotated_img = self.transform(rotated_img)
            rotated_images.append(rotated_img)
        rotated_images = torch.stack(rotated_images)

        return rotated_images, idx

# Initial transform for grayscale images
initial_transform = transforms.Compose([
    transforms.Resize((224, 224)), 
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])
 
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
        print(f"Shape of x before view: {x.shape}")  # Debug
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

# ---------------------------
# Utility: AverageMeter for timing
# ---------------------------
class AverageMeter(object):
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

def compute_features(dataloader, model, N, device, num_rotations=5):
    print("Computing features...")
    model.eval()
    batch_time = AverageMeter()
    end = time.time()

    features = None  # Initialize as None
    start_idx = 0  # Track cumulative samples

    for i, (input_tensor, _) in enumerate(tqdm(dataloader)):
        # Reshape and move to device
        B, num_rotations, C, H, W = input_tensor.shape
        input_tensor = input_tensor.view(B * num_rotations, C, H, W).to(device)

        # Extract features
        with torch.no_grad():
            aux = model(input_tensor).cpu().numpy()
            # aux = model(input_tensor).detach()

        # Debug
        print(f"Input tensor shape: {input_tensor.shape}")
        print(f"Model output shape: {aux.shape}")

        # Initialize feature matrix on the first batch
        if features is None:
            features = np.zeros((N * num_rotations, aux.shape[1]), dtype='float32')
            # features = torch.empty((N * num_rotations, aux.shape[1]), device=device, dtype=torch.float32)
            print(f"Feature matrix shape: {features.shape}")

        # Save extracted features
        end_idx = start_idx + aux.shape[0]
        features[start_idx:end_idx] = aux
        start_idx = end_idx 

        batch_time.update(time.time() - end)
        end = time.time()

    print(f"Feature extraction completed. Total time: {batch_time.sum:.2f} seconds")
    return features

def preprocess_features(features, pca_dim):
    """
    Preprocess features using PCA with whitening and L2 normalization.
    """
    features_np = features.cpu().detach().numpy()
    pca = PCA(n_components=pca_dim, whiten=True, svd_solver='full')
    features_pca = pca.fit_transform(features_np)
    norms = np.linalg.norm(features_pca, axis=1, keepdims=True)
    features_normalized = features_pca / (norms + 1e-10)
    return torch.tensor(features_normalized, device=features.device, dtype=torch.float32)

def kmeans_pytorch(features, num_clusters, num_iters=100, tol=1e-4, verbose=False):
    """
    Custom GPU-based k-means implementation.
    """
    device = features.device
    N, D = features.shape
    centroids = features[torch.randint(0, N, (num_clusters,))]

    for i in range(num_iters):
        distances = torch.cdist(features, centroids)
        cluster_assignments = torch.argmin(distances, dim=1)
        new_centroids = torch.zeros_like(centroids)
        for k in range(num_clusters):
            points_in_cluster = features[cluster_assignments == k]
            if len(points_in_cluster) > 0:
                new_centroids[k] = points_in_cluster.mean(dim=0)
        if torch.allclose(centroids, new_centroids, atol=tol):
            if verbose:
                print(f"K-means converged at iteration {i + 1}")
            break
        centroids = new_centroids

    return cluster_assignments, centroids

class ReassignedDataset(torch.utils.data.Dataset):
    """
    Dataset with pseudo-labels from clustering.
    """
    def __init__(self, subset, pseudo_labels, transform=None):
        self.subset = subset
        self.pseudo_labels = pseudo_labels
        self.transform = transform

    def __len__(self):
        return len(self.subset)

    def __getitem__(self, idx):
        img, _ = self.subset[idx]
        pseudo_label = self.pseudo_labels[idx]
        if self.transform and not isinstance(img, torch.Tensor):
            img = self.transform(img)
        return img, pseudo_label

def cluster_assign(images_lists, dataset, transform=None):
    """
    Creates a dataset from clustering with clusters as labels.
    """
    pseudolabels = []
    image_indexes = []
    for cluster, images in enumerate(images_lists):
        image_indexes.extend(images)
        pseudolabels.extend([cluster] * len(images))
    transform = transform or dataset.dataset.transform
    return ReassignedDataset(dataset, pseudolabels, transform)

def save_cluster_assignments(cluster_assignments, output_file=None):
    """
    Save cluster assignments to a CSV file with a unique identifier for each sample.
    """
    if output_file is None:
        output_file = os.path.join(RESULT_DIR, "cluster_assignments.csv")
    # Convert cluster assignments to a NumPy array
    cluster_assignments = cluster_assignments.cpu().numpy().flatten()
    # Create a DataFrame with a unique ID for each sample
    df = pd.DataFrame({
        "ID": np.arange(len(cluster_assignments)),
        "Cluster Assignment": cluster_assignments
    })
    df.to_csv(output_file, index=False)
    print(f"Cluster assignments saved to {output_file}")

# ---------------------------
# Training Parameters
# ---------------------------
NUM_EPOCHS = 50
NUM_CLUSTERS = 200
PCA_DIM = 64
BATCH_SIZE = 256
LEARNING_RATE = 0.001
WEIGHT_DECAY = 1e-5
WORLD_SIZE = torch.cuda.device_count()

# ---------------------------
# Distributed Setup Utilities
# ---------------------------
def setup(rank, world_size):
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = "29500"
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

def cleanup():
    dist.destroy_process_group()

# ---------------------------
# Training Function
# ---------------------------
def train(rank, world_size):
    setup(rank, world_size)

    torch.cuda.empty_cache()
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False

    start_time = time.time()

    # Initialize wandb on rank 0
    if rank == 0:
        wandb.init(project="alexnet_deepcluster", config={
            "num_epochs": NUM_EPOCHS,
            "num_clusters": NUM_CLUSTERS,
            "pca_dim": PCA_DIM,
            "batch_size": BATCH_SIZE,
            "learning_rate": LEARNING_RATE,
            "weight_decay": WEIGHT_DECAY,
            "num_rotations": 5
        })

    # Prepare dataset and dataloader
    dataset_path = "/gpfs/workdir/islamm/datasets"
    full_dataset = ChestXrayDataset(root_dir=dataset_path, transform=initial_transform, num_rotations=5)
    subset_size = int(0.1 * len(full_dataset))  # Use 10% of the dataset
    subset_indices = list(range(subset_size)) 
    subset_dataset = Subset(full_dataset, subset_indices)
    if rank == 0:
        print(f"Using first {subset_size} samples (10% of the dataset).")

    # Distributed sampler and dataloader
    train_sampler = DistributedSampler(subset_dataset, num_replicas=world_size, rank=rank, shuffle=True)
    dataloader = DataLoader(
        subset_dataset,
        batch_size=BATCH_SIZE // world_size,
        shuffle=False,
        num_workers=8,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2,
        sampler=train_sampler
    )
    print(f"GPU {rank}: DataLoader and Dataset prepared.")  # Debug

    # Initialize model, loss, optimizer, scaler, and scheduler
    model = alexnet(sobel=True, bn=True, out=NUM_CLUSTERS).to(rank)
    model = DDP(model, device_ids=[rank], output_device=rank)
    criterion = nn.CrossEntropyLoss().to(rank)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scaler = GradScaler()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

    # Initialize metrics dictionary to track time, loss, and NMI
    training_metrics = {
        "epoch": [],
        "avg_loss": [],
        "nmi": [],
        "epoch_time": [],
        "feature_extraction_time": [],
        "clustering_time": [],
        "cnn_train_time": []
    }

    # Path for saving model checkpoints
    checkpoint_base = os.path.join(RESULT_DIR, "alexnet_checkpoint_epoch")

    # Variable to store previous cluster assignments for NMI calculation
    prev_cluster_assignments = None

    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        schedule=torch.profiler.schedule(wait=1, warmup=1, active=5, repeat=1),
        on_trace_ready=torch.profiler.tensorboard_trace_handler(os.path.join(RESULT_DIR, "profiler_logs")),
        record_shapes=True,
        profile_memory=True,
        with_stack=True
    ) as prof:
        for epoch in range(NUM_EPOCHS):
            epoch_start_time = time.time()
            print(f"\n### Epoch {epoch + 1}/{NUM_EPOCHS} (GPU {rank}) ###")
            train_sampler.set_epoch(epoch)

            # Feature Extraction
            feature_start_time = time.time()
            dist.barrier()
            with record_function("Feature Extraction"):
                with torch.no_grad():
                    features = compute_features(dataloader, model, len(dataloader.dataset), rank, num_rotations=5)

                    # print("Before empty_cache:")
                    # print(f"GPU memory allocated: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
                    # print(f"GPU memory cached: {torch.cuda.memory_reserved() / 1e9:.2f} GB")

                    # torch.cuda.empty_cache()

                    # print("After empty_cache:")
                    # print(f"GPU memory allocated: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
                    # print(f"GPU memory cached: {torch.cuda.memory_reserved() / 1e9:.2f} GB")

                    print(f"Type of features after compute_features: {type(features)}")
                    print(f"Shape of features after compute_features: {features.shape}")

                    # Convert to tensor and reshape
                    features = torch.tensor(features, device=rank)
                    features = features.view(features.shape[0], -1)
                    print(f"Type of features after conversion: {type(features)}")
                    print(f"Shape of features after reshaping: {features.shape}")
            feature_end_time = time.time()
            feat_time = feature_end_time - feature_start_time
            print(f"GPU {rank}: Feature Extraction Time: {feat_time:.2f} seconds")

            # Save features (rank 0)
            if rank == 0:
                features_path = os.path.join(RESULT_DIR, "alexnet_features.npy")
                np.save(features_path, features.cpu().numpy())
                print(f"GPU {rank}: Features saved to '{features_path}'")

            # Preprocess features with PCA and normalization
            features = preprocess_features(features, pca_dim=PCA_DIM)

            # Clustering (K-means)
            cluster_start_time = time.time()
            dist.barrier()
            if rank == 0:
                print(f"GPU {rank}: Running PyTorch K-means clustering...")
                cluster_assignments, _ = kmeans_pytorch(features, NUM_CLUSTERS, num_iters=100, tol=1e-4, verbose=True)
                print(f"GPU {rank}: Cluster assignments device before moving to GPU: {cluster_assignments.device}")  # Debug
                cluster_assignments = cluster_assignments.to(rank)
                print(f"GPU {rank}: Cluster assignments device after moving to GPU: {cluster_assignments.device}")  # Debug
            else:
                cluster_assignments = torch.full((features.shape[0],), -1, dtype=torch.long, device=rank)
                print(f"GPU {rank}: Cluster assignments device (non-rank 0): {cluster_assignments.device}")  # Debug
            save_cluster_assignments(cluster_assignments)
            dist.broadcast(cluster_assignments, src=0)
            cluster_end_time = time.time()
            clust_time = cluster_end_time - cluster_start_time
            print(f"GPU {rank}: K-Means Clustering Time: {clust_time:.2f} seconds")

            # ---------------------------
            # NMI Score Calculation
            # ---------------------------
            nmi = 0.0 

            if prev_cluster_assignments is not None and rank == 0:
                nmi = normalized_mutual_info_score(
                    prev_cluster_assignments.cpu().numpy(),
                    cluster_assignments.cpu().numpy()
                )

            if rank == 0:
                print(f"Epoch {epoch+1}: NMI Score: {nmi:.4f}")

            # Create pseudo-label dataset based on clustering
            images_lists = [[] for _ in range(NUM_CLUSTERS)]
            for idx, cluster_id in enumerate(cluster_assignments.cpu().numpy()):
                images_lists[cluster_id].append(idx)
            pseudo_dataset = cluster_assign(images_lists, subset_dataset, transform=transforms.ToTensor())
            pseudo_dataloader = DataLoader(
                pseudo_dataset,
                batch_size=BATCH_SIZE // world_size,
                shuffle=True,
                num_workers=8,
                pin_memory=True,
                drop_last=True
            )
            print(f"GPU {rank}: Pseudo-label dataset created.")  # Debug

            # CNN Training Phase on pseudo-labels
            model.train()
            running_loss = 0.0
            cnn_train_start_time = time.time()
            for imgs, pseudo_labels in tqdm(pseudo_dataloader, desc=f"Training on GPU {rank}"):
                imgs, pseudo_labels = imgs.to(rank), pseudo_labels.to(rank)

                # Reshape: (B, 5, C, H, W) → (B*5, C, H, W)
                B, num_rotations, C, H, W = imgs.shape
                imgs = imgs.view(B * num_rotations, C, H, W)

                # Expand pseudo-labels to match the flattened batch
                pseudo_labels = pseudo_labels.repeat_interleave(num_rotations)

                # Debugging output
                print(f"GPU {rank}: Input images device: {imgs.device}, Pseudo-labels device: {pseudo_labels.device}")  # Debug

                # Forward pass with mixed precision
                with autocast():
                    outputs = model(imgs, return_features=False)
                    loss = criterion(outputs, pseudo_labels)

                # Backward pass and optimization
                optimizer.zero_grad()
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

                # Update running loss
                running_loss += loss.item()

            # Compute average loss for the epoch
            avg_loss = running_loss / len(pseudo_dataloader)
            cnn_train_end_time = time.time()
            cnn_train_time = cnn_train_end_time - cnn_train_start_time
            print(f"GPU {rank}: CNN Training Time: {cnn_train_time:.2f} seconds")
            print(f"GPU {rank}: Epoch {epoch + 1} Average Loss: {avg_loss:.4f}")

            # Step the scheduler using the average loss
            scheduler.step(avg_loss)
            current_lr = optimizer.param_groups[0]['lr']
            print(f"GPU {rank}: Learning Rate after epoch {epoch + 1}: {current_lr:.6f}")

            # Record epoch timing
            epoch_end_time = time.time()
            epoch_time = epoch_end_time - epoch_start_time

            # Log metrics and update CSV file on rank 0 after each epoch
            if rank == 0:
                wandb.log({
                    "epoch": epoch + 1,
                    "avg_loss": avg_loss,
                    "nmi": nmi,
                    "epoch_time": epoch_time,
                    "feature_extraction_time": feat_time,
                    "clustering_time": clust_time,
                    "cnn_train_time": cnn_train_time,
                    "learning_rate": current_lr
                })

                # Save checkpoint
                checkpoint = {
                    "epoch": epoch + 1,
                    "model_state_dict": model.module.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scaler_state_dict": scaler.state_dict(),
                    "avg_loss": avg_loss
                }
                checkpoint_path = f"{checkpoint_base}_{epoch + 1}.pth"
                torch.save(checkpoint, checkpoint_path)
                wandb.save(checkpoint_path)
                print(f"GPU {rank}: Checkpoint saved to '{checkpoint_path}'")

                # Update training metrics dictionary
                training_metrics["epoch"].append(epoch + 1)
                training_metrics["avg_loss"].append(avg_loss)
                training_metrics["nmi"].append(nmi)
                training_metrics["epoch_time"].append(epoch_time)
                training_metrics["feature_extraction_time"].append(feat_time)
                training_metrics["clustering_time"].append(clust_time)
                training_metrics["cnn_train_time"].append(cnn_train_time)

                # Save the training metrics CSV file after every epoch
                csv_path = os.path.join(RESULT_DIR, "alexnet_training_metrics.csv")
                df_metrics = pd.DataFrame(training_metrics)
                df_metrics.to_csv(csv_path, index=False)
                print(f"Epoch {epoch + 1}: Training metrics saved to '{csv_path}'")

                # Update previous cluster assignments for the next epoch's NMI calculation
                prev_cluster_assignments = cluster_assignments.clone()

            prof.step()
            print(f"GPU {rank}: Epoch {epoch + 1} Time: {epoch_time:.2f} seconds")

    # End of training
    total_training_time = time.time() - start_time
    print(f"GPU {rank}: Total Training Time: {total_training_time / 60:.2f} minutes")
    if rank == 0:
        wandb.finish()
    cleanup()

# ---------------------------
# Multi-GPU Training Launcher
# ---------------------------
def main():
    world_size = torch.cuda.device_count()
    print(f"Launching distributed training on {world_size} GPUs...")
    mp.spawn(train, args=(world_size,), nprocs=world_size, join=True)

if __name__ == "__main__":
    main()

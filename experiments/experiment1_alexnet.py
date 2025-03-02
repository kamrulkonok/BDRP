import os
import sys
import pandas as pd
import numpy as np
import random
import wandb
import torch.distributed as dist
from torch.utils.data import Subset
from collections import defaultdict
import matplotlib.pyplot as plt
import argparse
import shutil
import torch.backends.cudnn as cudnn
import torch
import cv2
from sklearn.decomposition import PCA
import torch.optim as optim
import torch.nn as nn
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler, Dataset
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
class ChestXrayDataset(Dataset):
    """
    Chest X-ray dataset with:
    - Sobel filtering (2-channel output)
    - Random rotations (-R° to +R° augmentation)
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
        """
        angles = [random.uniform(-self.rotation_range, self.rotation_range) for _ in range(self.num_rotations)]
        return [img] + [img.rotate(angle, resample=Image.BILINEAR) for angle in angles]  # Include original image

    def apply_sobel_filter(self, img):
        """
        Apply Sobel filter to generate a 2-channel edge-detected image.
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

# ---------------------------
# AlexNet Model Definition
# ---------------------------
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

    def forward(self, x, return_features=True):
        if x.dim() == 5: 
            batch_size, num_rotations, C, H, W = x.shape
            x = x.view(batch_size * num_rotations, C, H, W)  

        x = self.features(x)

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
            in_channels = v[0]

    return nn.Sequential(*layers)

def alexnet(sobel=True, bn=True, out=200):
    """
    Create an AlexNet model with optional Sobel filtering.
    """
    input_dim = 2 if sobel else 1 
    model = AlexNet(make_layers_features(CFG['2012'], input_dim, bn=bn), out, sobel)
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

def compute_features(dataloader, model, N, device):
    """
    Extract features from the model and save them.
    """
    model.eval()
    features = []
    
    with torch.no_grad():
        for i, (imgs, _) in enumerate(tqdm(dataloader, desc=f"Feature Extraction (GPU {device})")):
            imgs = imgs.to(device, non_blocking=True)
            batch_size, num_rotations, C, H, W = imgs.shape
            imgs = imgs.view(batch_size * num_rotations, C, H, W) 
            features_extracted = model.module.features(imgs) if isinstance(model, DDP) else model.features(imgs)
            features_extracted = features_extracted.view(features_extracted.shape[0], -1).cpu()
            features.append(features_extracted)

    features = torch.cat(features)

    # Save features on rank 0
    dist.barrier()
    if dist.get_rank() == 0:  
        features_path = os.path.join(RESULT_DIR, "alexnet_features.npy")
        np.save(features_path, features.numpy())
        print(f"GPU {device}: Saved extracted features to '{features_path}'")

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
NUM_EPOCHS = 100
NUM_CLUSTERS = 200
PCA_DIM = 128
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

# ---------------------------
# Training Function
# ---------------------------
def train(rank, world_size):
    setup(rank, world_size)
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

    # Prepare dataset and dataloader (using only first 20% of the dataset)
    dataset_path = "/gpfs/workdir/islamm/datasets"
    full_dataset = ChestXrayDataset(root_dir=dataset_path, use_sobel=True, num_rotations=5)
    subset_size = int(0.002 * len(full_dataset))
    subset_indices = list(range(subset_size))
    subset_dataset = Subset(full_dataset, subset_indices)
    if rank == 0:
        print(f"Using first 20% of dataset: {subset_size} samples.")
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

    # Initialize model, loss, optimizer, scaler, and scheduler
    model = alexnet(sobel=True, bn=True, out=NUM_CLUSTERS).to(rank)
    model = DDP(model, device_ids=[rank], output_device=rank)
    criterion = nn.CrossEntropyLoss().to(rank)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scaler = GradScaler()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

    # Initialize metrics dictionary to track time and loss
    training_metrics = {
        "epoch": [],
        "avg_loss": [],
        "epoch_time": [],
        "feature_extraction_time": [],
        "clustering_time": [],
        "cnn_train_time": []
    }

    # Path for saving model checkpoints
    checkpoint_base = os.path.join(RESULT_DIR, "alexnet_checkpoint_epoch")

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
                    features = compute_features(dataloader, model, len(subset_dataset), rank)
                features = features.view(features.shape[0], -1)
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
            else:
                cluster_assignments = torch.full((features.shape[0],), -1, dtype=torch.long, device=rank)
            save_cluster_assignments(cluster_assignments)
            dist.broadcast(cluster_assignments, src=0)
            cluster_end_time = time.time()
            clust_time = cluster_end_time - cluster_start_time
            print(f"GPU {rank}: K-Means Clustering Time: {clust_time:.2f} seconds")

            # Create pseudo-label dataset based on clustering
            images_lists = [[] for _ in range(NUM_CLUSTERS)]
            for idx, cluster_id in enumerate(cluster_assignments.cpu().numpy()):
                images_lists[cluster_id].append(idx)
            pseudo_dataset = cluster_assign(images_lists, subset_dataset, transform=transforms.ToTensor())
            pseudo_dataloader = DataLoader(
                pseudo_dataset,
                batch_size=BATCH_SIZE // world_size,
                shuffle=True,
                num_workers=2,
                pin_memory=True
            )

            # CNN Training Phase on pseudo-labels
            model.train()
            running_loss = 0.0
            cnn_train_start_time = time.time()
            for imgs, pseudo_labels in tqdm(pseudo_dataloader, desc=f"Training on GPU {rank}"):
                imgs, pseudo_labels = imgs.to(rank), pseudo_labels.to(rank)
                with autocast():
                    outputs = model(imgs, return_features=False)
                    loss = criterion(outputs, pseudo_labels)
                optimizer.zero_grad()
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                running_loss += loss.item()
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
                training_metrics["epoch_time"].append(epoch_time)
                training_metrics["feature_extraction_time"].append(feat_time)
                training_metrics["clustering_time"].append(clust_time)
                training_metrics["cnn_train_time"].append(cnn_train_time)

                # Save the training metrics CSV file after every epoch
                csv_path = os.path.join(RESULT_DIR, "alexnet_training_metrics.csv")
                df_metrics = pd.DataFrame(training_metrics)
                df_metrics.to_csv(csv_path, index=False)
                print(f"Epoch {epoch + 1}: Training metrics saved to '{csv_path}'")

            prof.step()
            print(f"GPU {rank}: Epoch {epoch + 1} Time: {epoch_time:.2f} seconds")

    # End of training
    total_training_time = time.time() - start_time
    print(f"GPU {rank}: Total Training Time: {total_training_time / 60:.2f} minutes")
    if rank == 0:
        wandb.finish()
    dist.destroy_process_group()

# ---------------------------
# Multi-GPU Training Launcher
# ---------------------------
if __name__ == "__main__":
    print(f"Launching multi-GPU training on {WORLD_SIZE} GPUs...")
    torch.multiprocessing.set_start_method("spawn", force=True)
    mp.spawn(train, args=(WORLD_SIZE,), nprocs=WORLD_SIZE, join=True)
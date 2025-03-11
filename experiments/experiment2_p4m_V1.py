import time
import os
import sys
import random
import math
import numpy as np
import pandas as pd
import csv
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import DataLoader, Dataset, DistributedSampler
from torchsummary import summary

import torchvision.transforms as transforms
from PIL import Image
from tqdm import tqdm
from sklearn.metrics.cluster import normalized_mutual_info_score
from sklearn.decomposition import PCA

from e2resnet import E2ResNet, E2BasicBlock
from e2cnn import gspaces, nn as e2nn

import wandb
from torch.profiler import profile, record_function, ProfilerActivity

# ---------------------------
# Global Result Directory
# ---------------------------
RESULT_DIR = "/gpfs/workdir/islamm/results"
if not os.path.exists(RESULT_DIR):
    os.makedirs(RESULT_DIR)

# ---------------------------
# Training Parameters
# ---------------------------
NUM_EPOCHS = 100
NUM_CLUSTERS = 200
PCA_DIM = 64
BATCH_SIZE = 64
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4

CHECKPOINT_PATH = os.path.join(RESULT_DIR, "se2_checkpoint.pth")
FEATURES_SAVE_PATH = os.path.join(RESULT_DIR, "se2_features.npy")

# Set random seed for reproducibility
RANDOM_SEED = 42
random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

# ---------------------------
# Dataset Definition
# ---------------------------
class ChestXrayDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = [
            os.path.join(root_dir, fname)
            for fname in os.listdir(root_dir)
            if fname.endswith(('.png', '.jpg', '.jpeg'))
        ]

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        img = Image.open(img_path).convert("L") 
        if self.transform:
            img = self.transform(img)
        return img, idx 

# ---------------------------
# P4M Model Definition
# ---------------------------
class P4MClusteringCNN(nn.Module):
    def __init__(self, n_features=200):
        super(P4MClusteringCNN, self).__init__()

        # P4M Group: Rotations + Reflections
        self.p4m_act = gspaces.FlipRot2dOnR2(N=4)  # 4 rotations + mirror flips

        # Input Type
        in_type = e2nn.FieldType(self.p4m_act, 1 * [self.p4m_act.trivial_repr])
        self.input_type = in_type

        # Block 1
        out_type1 = e2nn.FieldType(self.p4m_act, 64 * [self.p4m_act.regular_repr])
        self.block1 = e2nn.SequentialModule(
            e2nn.R2Conv(in_type, out_type1, kernel_size=5, padding=2, bias=False),
            e2nn.InnerBatchNorm(out_type1),
            e2nn.ReLU(out_type1, inplace=True),
            e2nn.PointwiseAvgPoolAntialiased(out_type1, sigma=0.66, stride=2)
        )

        # Block 2
        out_type2 = e2nn.FieldType(self.p4m_act, 128 * [self.p4m_act.regular_repr])
        self.block2 = e2nn.SequentialModule(
            e2nn.R2Conv(self.block1.out_type, out_type2, kernel_size=3, padding=1, bias=False),
            e2nn.InnerBatchNorm(out_type2),
            e2nn.ReLU(out_type2, inplace=True),
            e2nn.PointwiseAvgPoolAntialiased(out_type2, sigma=0.66, stride=2)
        )

        # Block 3
        out_type3 = e2nn.FieldType(self.p4m_act, 256 * [self.p4m_act.regular_repr])
        self.block3 = e2nn.SequentialModule(
            e2nn.R2Conv(self.block2.out_type, out_type3, kernel_size=3, padding=1, bias=False),
            e2nn.InnerBatchNorm(out_type3),
            e2nn.ReLU(out_type3, inplace=True),
            e2nn.PointwiseAvgPoolAntialiased(out_type3, sigma=0.66, stride=2)
        )

        # Block 4
        out_type4 = e2nn.FieldType(self.p4m_act, 512 * [self.p4m_act.regular_repr])
        self.block4 = e2nn.SequentialModule(
            e2nn.R2Conv(self.block3.out_type, out_type4, kernel_size=3, padding=1, bias=False),
            e2nn.InnerBatchNorm(out_type4),
            e2nn.ReLU(out_type4, inplace=True),
            e2nn.PointwiseAvgPoolAntialiased(out_type4, sigma=0.66, stride=2)
        )

        # Group Pooling: Pool across rotations and reflections
        self.gpool = e2nn.GroupPooling(out_type4)

        # Adaptive Global Pooling
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)

        # Fully Connected Layer for final embedding
        self.fc = nn.Linear(512, n_features)

    def forward(self, x, return_features=True):
        x = e2nn.GeometricTensor(x, self.input_type)
        x = self.block1(x)
        x = self.block2(x) 
        x = self.block3(x)
        x = self.block4(x)
        x = self.gpool(x) 
        x = x.tensor
        x = self.global_avg_pool(x)
        x = x.view(x.size(0), -1)
        if return_features:
            return x
        return self.fc(x)

# ---------------------------
# Utility: AverageMeter for timing
# ---------------------------
class AverageMeter(object):
    def __init__(self):
        self.reset()
    def reset(self):
        self.val = 0
        self.sum = 0
        self.count = 0
        self.avg = 0
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

# ---------------------------
# Feature Extraction
# ---------------------------
def compute_features(dataloader, model, N, device, return_features=True):
    print("Computing features...")
    model.eval()
    features = None

    for i, (input_tensor, _) in enumerate(tqdm(dataloader, desc="Feature Extraction")):
        if input_tensor is None:
            raise ValueError(f"Batch {i} returned None. Check dataset and DataLoader.")
        
        input_tensor = input_tensor.to(device)
        with torch.no_grad():
            output = model(input_tensor, return_features=return_features)
            if isinstance(output, e2nn.GeometricTensor):
                aux = output.tensor.cpu().numpy()
            else:
                aux = output.cpu().numpy()

        if features is None:
            features = np.zeros((N, aux.shape[1]), dtype='float32')

        start_idx = i * dataloader.batch_size
        end_idx = start_idx + aux.shape[0]
        features[start_idx:end_idx] = aux

    if features is None or len(features) == 0:
        raise ValueError("No features extracted. Check dataset and DataLoader.")
    
    print("Feature extraction completed.")
    return features

# ---------------------------
# Feature Preprocessing
# ---------------------------

def preprocess_features(features, pca_dim):
    """
    Preprocess features using scikit-learn PCA with whitening and L2 normalization.
    """
    features_np = features.cpu().detach().numpy()
    pca = PCA(n_components=pca_dim, whiten=True, svd_solver='full')
    features_pca = pca.fit_transform(features_np)
    norms = np.linalg.norm(features_pca, axis=1, keepdims=True)
    features_normalized = features_pca / (norms + 1e-10)
    return torch.tensor(features_normalized, device=features.device, dtype=torch.float32)

# ---------------------------
# PyTorch K-Means Clustering
# ---------------------------
def kmeans_pytorch(features, num_clusters, num_iters=100, tol=1e-4, verbose=False):
    """
    Custom GPU-based k-means implementation using PyTorch.
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

# ---------------------------
# Reassigned Dataset for Pseudo-labels
# ---------------------------
class ReassignedDataset(Dataset):
    def __init__(self, dataset, pseudo_labels, transform=None):
        self.dataset = dataset
        self.pseudo_labels = pseudo_labels
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img, _ = self.dataset[idx]
        pseudo_label = self.pseudo_labels[idx]
        if self.transform and not isinstance(img, torch.Tensor):
            img = self.transform(img)
        return img, pseudo_label

def cluster_assign(images_lists, dataset, transform=None):
    pseudolabels = []
    image_indexes = []
    for cluster, images in enumerate(images_lists):
        image_indexes.extend(images)
        pseudolabels.extend([cluster] * len(images))
    transform = transform or dataset.transform
    return ReassignedDataset(dataset, pseudolabels, transform)

# ---------------------------
# save cluster_assignments
# ---------------------------
def save_cluster_assignments(cluster_assignments, output_file=None):
    if output_file is None:
        output_file = os.path.join(RESULT_DIR, "p4m_cluster_assignments.csv")
    cluster_assignments = cluster_assignments.cpu().numpy().flatten()
    df = pd.DataFrame({
        "ID": np.arange(len(cluster_assignments)),
        "Cluster Assignment": cluster_assignments
    })
    df.to_csv(output_file, index=False)
    print(f"Cluster assignments saved to {output_file}")

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
# Distributed Training Function
# ---------------------------
def train(rank, world_size):
    setup(rank, world_size)
    
    # Initialize wandb
    if rank == 0:
        wandb.init(
            project="chest_xray_gcnn_clustering",
            config={
                "num_epochs": NUM_EPOCHS,
                "num_clusters": NUM_CLUSTERS,
                "batch_size": BATCH_SIZE,
                "learning_rate": LEARNING_RATE,
                "weight_decay": WEIGHT_DECAY,
                "pca_dim": PCA_DIM,
            }
        )

    # Define transforms and load dataset
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])
    dataset_path = "/gpfs/workdir/islamm/datasets"
    full_dataset = ChestXrayDataset(root_dir=dataset_path, transform=transform)

    # Use the first 20% of the dataset for training
    subset_size = int(0.2 * len(full_dataset))  
    subset_indices = list(range(subset_size))
    subset_dataset = torch.utils.data.Subset(full_dataset, subset_indices)

    if rank == 0:
        print(f"Using first {subset_size} samples.")

    # Create a Distributed Sampler for multi-GPU training
    sampler = DistributedSampler(subset_dataset, num_replicas=world_size, rank=rank, shuffle=True)
    dataloader = DataLoader(
        subset_dataset,
        batch_size=BATCH_SIZE // world_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        sampler=sampler,
        drop_last=True
    )

    device = torch.device(f"cuda:{rank}")

     # Initialize P4M 
    model = P4MClusteringCNN(n_features=NUM_CLUSTERS).to(device)
    
    # Print Model Summary
    if rank == 0:
        print("Model Summary:")
        summary(model, input_size=(1, 224, 224), device=str(device))

    # Optimizer, Loss, and Scheduler
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scaler = GradScaler()
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

    torch.backends.cudnn.benchmark = True

    # Start tracking total training time
    total_start_time = time.time()
    training_metrics = {"epoch": [], "nmi": [], "average_loss": [], "epoch_time": [], "feature_extraction_time": [], "clustering_time": []}
    prev_cluster_assignments = None

    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        schedule=torch.profiler.schedule(wait=1, warmup=1, active=5, repeat=1),
        on_trace_ready=torch.profiler.tensorboard_trace_handler(f"/gpfs/workdir/islamm/profiler_logs_rank{rank}"),
        record_shapes=True,
        profile_memory=True,
        with_stack=True
    ) as prof:
        for epoch in range(NUM_EPOCHS):
            epoch_start_time = time.time()
            print(f"\n### Epoch {epoch+1}/{NUM_EPOCHS} (Rank {rank}) ###")
            sampler.set_epoch(epoch)

            # ---------------------------
            # Feature Extraction
            # ---------------------------
            feature_extraction_start_time = time.time()
            dist.barrier()
            with record_function("Feature Extraction"):
                features_np = compute_features(dataloader, model, subset_size, device, return_features=True)
            feature_extraction_time = time.time() - feature_extraction_start_time
            if rank == 0:
                np.save(FEATURES_SAVE_PATH, features_np)
                print(f"Saved extracted features to {FEATURES_SAVE_PATH}")

            if features_np is None:
                raise ValueError("Feature extraction returned None.")

            features_tensor = preprocess_features(torch.tensor(features_np), pca_dim=PCA_DIM).to(device)

            # ---------------------------
            # K-Means Clustering
            # ---------------------------
            clustering_start_time = time.time()
            dist.barrier()
            if rank == 0:
                print(f"GPU {rank}: Running PyTorch K-means clustering...")
                cluster_assignments, _ = kmeans_pytorch(features_tensor, NUM_CLUSTERS, num_iters=100, tol=1e-4, verbose=True)
            else:
                cluster_assignments = torch.full((features_tensor.shape[0],), -1, dtype=torch.long, device=device)
            save_cluster_assignments(cluster_assignments)
            dist.broadcast(cluster_assignments, src=0)
            clustering_time = time.time() - clustering_start_time

            if prev_cluster_assignments is not None and rank == 0:
                nmi = normalized_mutual_info_score(
                    prev_cluster_assignments.cpu().numpy(),
                    cluster_assignments.cpu().numpy()
                )
            else:
                nmi = 0
            if rank == 0:
                print(f"Epoch {epoch+1}: NMI Score: {nmi:.4f}")

            # ---------------------------
            # CNN Training Phase
            # ---------------------------
            model.train()
            running_loss = 0.0
            with record_function("CNN Training"):
                for imgs, pseudo_labels in tqdm(dataloader, desc=f"Training Rank {rank}"):
                    imgs = imgs.to(device)
                    pseudo_labels = pseudo_labels.to(device)
                    if pseudo_labels.min() < 0 or pseudo_labels.max() >= NUM_CLUSTERS:
                        #print(f"Remapping targets: min={pseudo_labels.min().item()}, max={pseudo_labels.max().item()}")
                        pseudo_labels = pseudo_labels % NUM_CLUSTERS

                    with autocast():
                        outputs = model(imgs, return_features=False)
                        if outputs.shape[1] != NUM_CLUSTERS:
                            print(f"Warning: Model output dim {outputs.shape[1]} != NUM_CLUSTERS {NUM_CLUSTERS}")
                        loss = criterion(outputs, pseudo_labels)

                    optimizer.zero_grad()
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                    running_loss += loss.item()

            avg_loss = running_loss / len(dataloader)
            if rank == 0:
                print(f"Epoch {epoch+1}: Average Loss: {avg_loss:.4f}")
            
            # Save model checkpoint
            if rank == 0:
                torch.save({
                    'epoch': epoch + 1,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': avg_loss,
                }, CHECKPOINT_PATH)
                print(f"Model checkpoint saved to {CHECKPOINT_PATH}")

            epoch_time = time.time() - epoch_start_time
            training_metrics["epoch"].append(epoch + 1)
            training_metrics["nmi"].append(nmi)
            training_metrics["average_loss"].append(avg_loss)
            training_metrics["epoch_time"].append(epoch_time)
            training_metrics["feature_extraction_time"].append(feature_extraction_time)
            training_metrics["clustering_time"].append(clustering_time)
            scheduler.step(avg_loss)
            prev_cluster_assignments = cluster_assignments.clone()
            prof.step()

    if rank == 0:
        total_training_time = time.time() - total_start_time
        csv_file = "p4m_training_time_metrics.csv"
        with open(csv_file, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["epoch", "nmi", "average_loss", "epoch_time", "feature_extraction_time", "clustering_time"])
            writer.writeheader()
            for ep, nmi_val, loss_val, ep_time, fe_time, cl_time in zip(
                training_metrics["epoch"], training_metrics["nmi"], training_metrics["average_loss"],
                training_metrics["epoch_time"], training_metrics["feature_extraction_time"], training_metrics["clustering_time"]
            ):
                writer.writerow({
                    "epoch": ep, "nmi": nmi_val, "average_loss": loss_val,
                    "epoch_time": ep_time, "feature_extraction_time": fe_time, "clustering_time": cl_time
                })
        print(f"Training time metrics saved to {csv_file}")
        print(f"Total Training Time: {total_training_time / 60:.2f} minutes")
        wandb.finish()
    cleanup()

def main():
    world_size = torch.cuda.device_count()
    print(f"Launching distributed training on {world_size} GPUs...")
    mp.spawn(train, args=(world_size,), nprocs=world_size, join=True)

if __name__ == "__main__":
    main()
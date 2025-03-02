import os
import numpy as np
import shutil
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.models import alexnet
from sklearn.decomposition import PCA
import random
import matplotlib.pyplot as plt
from PIL import Image


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

# Define R-MAC Feature Extraction
class RMACExtractor:
    def __init__(self, model, levels=3):
        self.model = model
        self.levels = levels

    @staticmethod
    def get_rmac_region_coordinates(H, W, L):
        steps = np.array((2, 3, 4, 5, 6, 7), dtype=np.float32)
        w = np.minimum(H, W)
        b = (np.maximum(H, W) - w) / (steps - 1)
        idx = np.argmin(np.abs(((w**2 - w * b) / w**2) - 0.4)) + 1

        Wd = idx if H < W else 0
        Hd = idx if H > W else 0

        regions = []
        for l in range(1, L + 1):
            wl = np.floor(2 * w / (l + 1))
            wl2 = np.floor(wl / 2 - 1)
            if l + Wd - 1 > 0:
                b = (W - wl) / (l + Wd - 1)
            else:
                b = 0
            cenW = np.floor(wl2 + b * np.arange(l - 1 + Wd + 1)) - wl2
            if l + Hd - 1 > 0:
                b = (H - wl) / (l + Hd - 1)
            else:
                b = 0
            cenH = np.floor(wl2 + b * np.arange(l - 1 + Hd + 1)) - wl2
            for i in cenH:
                for j in cenW:
                    regions.append([j, i, wl, wl])

        for region in regions:
            for j in range(4):
                region[j] = int(round(region[j]))
        return np.array(regions)

    @staticmethod
    def normalize_L2(features):
        return features / np.linalg.norm(features, axis=1, keepdims=True)

    def extract_rmac_features(self, feature_maps):
        _, C, H, W = feature_maps.size()
        regions = self.get_rmac_region_coordinates(H, W, self.levels)

        rmac_descriptors = []
        for x, y, w, h in regions:
            region = feature_maps[:, :, y:y + h, x:x + w]
            pooled = torch.max(torch.max(region, dim=2)[0], dim=2)[0]
            rmac_descriptors.append(pooled)

        rmac_descriptors = torch.stack(rmac_descriptors, dim=1)
        aggregated = torch.sum(rmac_descriptors, dim=1)
        return self.normalize_L2(aggregated.cpu().numpy())


# Create Query Dataset
def create_query_dataset(original_dataset_path, query_path, query_size=100):
    if not os.path.exists(query_path):
        os.makedirs(query_path)

    all_images = [
        fname for fname in os.listdir(original_dataset_path)
        if fname.endswith((".png", ".jpg", ".jpeg"))
    ]
    np.random.shuffle(all_images)
    query_images = all_images[:query_size]

    for img_name in query_images:
        src = os.path.join(original_dataset_path, img_name)
        dst = os.path.join(query_path, img_name)
        shutil.copy(src, dst)

    print(f"Query dataset created with {len(query_images)} images.")
    
# Paths
dataset_path = "/gpfs/workdir/islamm/rotated_datasets_without_NoFindings"
query_path = "/gpfs/workdir/islamm/query"
create_query_dataset(dataset_path, query_path, query_size=10)

# Define Transform
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

# Load Datasets
dataset = ChestXrayDataset(root_dir=dataset_path, transform=transform)
query_dataset = ChestXrayDataset(root_dir=query_path, transform=transform)

# Define Dataloaders
dataloader = DataLoader(dataset, batch_size=64, shuffle=False, num_workers=4)
query_dataloader = DataLoader(query_dataset, batch_size=64, shuffle=False, num_workers=4)

# Load precomputed database features
database_features = np.load("/gpfs/workdir/islamm/database_features.npy")

# Extract Query Features
# Load the pre-trained AlexNet model
model = alexnet()

# Modify the first convolutional layer to accept grayscale input
# The original AlexNet first layer takes 3 channels (RGB), here we change it to 1 channel
original_weights = model.features[0].weight.clone() 
model.features[0] = torch.nn.Conv2d(
    in_channels=1,  # Grayscale images have 1 channel
    out_channels=model.features[0].out_channels,
    kernel_size=model.features[0].kernel_size,
    stride=model.features[0].stride,
    padding=model.features[0].padding
)

# Initialize the new weights by averaging the original weights across the channel dimension
model.features[0].weight.data = original_weights.mean(dim=1, keepdim=True)

# Transfer the model to the GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

rmac_extractor = RMACExtractor(model=model, levels=3)
query_features = []

for images, _ in query_dataloader:
    images = images.cuda()
    with torch.no_grad():
        feature_maps = model.features(images)
        rmac_features = rmac_extractor.extract_rmac_features(feature_maps)
        query_features.append(rmac_features)

query_features = np.vstack(query_features)

# Apply PCA
pca_dim = 128
def preprocess_features(features, pca_dim):
    pca = PCA(n_components=pca_dim)
    features = pca.fit_transform(features)
    return features

database_features = preprocess_features(database_features, pca_dim)
query_features = preprocess_features(query_features, pca_dim)

# Compute Similarity
similarity_matrix = np.dot(query_features, database_features.T)

# Retrieve Top-k Matches
k = 3
rankings = {}
for i, row in enumerate(similarity_matrix):
    top_k_indices = np.argsort(-row)[:k]
    rankings[f"Query {i}"] = top_k_indices.tolist()
    print(f"Query {i} top {k} matches: {top_k_indices}")
def visualize_single_query(query_idx, query_path, dataset_path, rankings, k=3, save_path="query_matches.png"):
    """
    Visualizes a single query image and its top-k matches from the database.

    Args:
        query_idx (int): Index of the query to visualize.
        query_path (str): Path to the directory containing query images.
        dataset_path (str): Path to the directory containing database images.
        rankings (dict): Dictionary containing rankings for all queries.
        k (int): Number of top matches to visualize.
        save_path (str): Path to save the output visualization image.
    """
    query_images = [os.path.join(query_path, fname) for fname in os.listdir(query_path)]
    database_images = [os.path.join(dataset_path, fname) for fname in os.listdir(dataset_path)]

    # Get the top-k matches for the selected query
    top_k_indices = rankings[f"Query {query_idx}"]

    # Create the plot
    fig, axes = plt.subplots(1, k + 1, figsize=(15, 5))

    # Load and display the query image
    query_image = Image.open(query_images[query_idx])
    axes[0].imshow(query_image, cmap='gray')
    axes[0].set_title("Query Image")
    axes[0].axis('off')

    # Load and display the top-k matches
    for i, match_idx in enumerate(top_k_indices):
        match_image = Image.open(database_images[match_idx])
        axes[i + 1].imshow(match_image, cmap='gray')
        axes[i + 1].set_title(f"Match {i + 1}")
        axes[i + 1].axis('off')

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close(fig) 
    print(f"Visualization saved to {save_path}")


query_idx = 10 
save_path = "/gpfs/workdir/islamm/output/query_10_matches.png" 

visualize_single_query(query_idx, query_path, dataset_path, rankings, k=3, save_path=save_path)

import torch
import torch.nn as nn
import torchvision.models as models
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import roc_auc_score

# Load your pretrained DeepCluster model (modify according to your setup)
pretrained_model = torch.load("/gpfs/workdir/islamm/alexnet_trainedV1.pth")  
backbone = pretrained_model.features  # Extract feature extractor

# Define a new classifier head (for NIH ChestX-ray, there are 14 disease labels)
num_classes = 14  
classifier = nn.Linear(in_features=backbone[-1].out_channels, out_features=num_classes)

# Combine backbone with new classifier
model = nn.Sequential(
    backbone,
    nn.AdaptiveAvgPool2d((1, 1)),  # Global Average Pooling
    nn.Flatten(),  
    classifier
)


for param in backbone.parameters():
    param.requires_grad = False  # Freeze pretrained layers
for param in classifier.parameters():
    param.requires_grad = True  # Train only the classifier



criterion = nn.BCEWithLogitsLoss()  # For multi-label classification

# Optimizer choice (adjust learning rate based on fine-tuning strategy)
optimizer = optim.Adam(model.parameters(), lr=1e-4)  # Full fine-tuning
# optimizer = optim.Adam(classifier.parameters(), lr=1e-3)  # Linear probing

# Learning rate scheduler (optional)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)


# Define transformations (for training & evaluation)
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),  
    transforms.RandomHorizontalFlip(),  
    transforms.ToTensor(),  
    transforms.Normalize(mean=[0.5], std=[0.5])  # Adjust based on dataset statistics
])

test_transform = transforms.Compose([
    transforms.Resize((224, 224)),  
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

# Define DataLoader
train_loader = DataLoader(ChestXrayDataset(train_data, transform=train_transform), batch_size=32, shuffle=True)
val_loader = DataLoader(ChestXrayDataset(val_data, transform=test_transform), batch_size=32, shuffle=False)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

num_epochs = 20
for epoch in range(num_epochs):
    model.train()  # Set to training mode
    running_loss = 0.0
    
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()

    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader)}")

    # Adjust learning rate
    scheduler.step()


model.eval()  # Set model to evaluation mode
y_true, y_pred = [], []

with torch.no_grad():
    for images, labels in val_loader:
        images = images.to(device)
        outputs = model(images)
        preds = torch.sigmoid(outputs).cpu().numpy()  # Convert logits to probabilities
        
        y_true.append(labels.numpy())
        y_pred.append(preds)

# Compute AUROC for each disease label
y_true = np.vstack(y_true)
y_pred = np.vstack(y_pred)

auc_scores = roc_auc_score(y_true, y_pred, average=None)
print(f"AUROC per class: {auc_scores}")
print(f"Mean AUROC: {auc_scores.mean()}")

torch.save(model.state_dict(), "fine_tuned_chestxray.pth")
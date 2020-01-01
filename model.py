import torch
from torchvision import datasets, models, transforms

import os


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

# Configure paths
data_base_path = "data"
train_path = os.path.join(data_base_path, "train")
val_path = os.path.join(data_base_path, "validation")

# Configure datasets
transform = transforms.Compose([
    transforms.RandomResizedCrop(size=224),
    transforms.RandomRotation(degrees=15),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

train_dataset = datasets.ImageFolder(train_path, transform = transform)
val_dataset = datasets.ImageFolder(val_path, transform = transform)

batch_size = 8
num_workers = 4
dataloaders = {
    'train': torch.utils.data.DataLoader(train_dataset, batch_size = batch_size, shuffle = True, num_workers = num_workers),
    'val': torch.utils.data.DataLoader(val_dataset, batch_size = batch_size, shuffle = True, num_workers = num_workers)
}

print(f"Samples in Train: {len(train_dataset)}")
print(f"Samples in Validation: {len(val_dataset)}")
print(f"Total samples: {len(train_dataset) + len(val_dataset)}")
print(f"Number of classes: {len(train_dataset.classes)}")
print(f"Classes: {train_dataset.classes}")


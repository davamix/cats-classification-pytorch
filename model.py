import torch
import torch.nn as nn
import torchvision
from torchvision import datasets, models, transforms
import torch.optim as optim
from torch.optim import lr_scheduler
import matplotlib.pyplot as plt

import os
import time
import datetime
import copy

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

# Configure paths
data_base_path = "data"
train_path = os.path.join(data_base_path, "train")
val_path = os.path.join(data_base_path, "validation")
model_path = 'cat-model.pth.tar'
experiments_path = "experiments" # Path to save the current values of hyperparameters and metrics for each experiment

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

dataset_sizes = {
    'train': len(train_dataset),
    'val': len(val_dataset)
}

print(f"Samples in Train: {len(train_dataset)}")
print(f"Samples in Validation: {len(val_dataset)}")
print(f"Total samples: {len(train_dataset) + len(val_dataset)}")
print(f"Number of classes: {len(train_dataset.classes)}")
print(f"Classes: {train_dataset.classes}")


'''
Create a file with the values of the hyperparameters
name: name of the experiment (datetime)
hyperparameters: dictionary. {"lr":"0.0.1", ...}
'''
def save_experiment(exp_path, hyperparameters):
    if not os.path.exists(exp_path):
        os.makedirs(ex_path)

    params_path = os.path.join(ex_path, "parameters.txt")
    f = open(params_path, "w")
    
    for hp in hyperparameters:
        f.write(f"{hp}: {hyperparameters[hp]}\n")
    
    f.close()

# Save metrics
def save_metrics(exp_path, t_losses, v_losses, t_accs, v_accs):
    if not os.path.exists(exp_path):
        os.makedirs(ex_path)

    fig = plt.figure(figsize=(15,5))
    plt.subplot(1,2,1)
    plt.plot(t_losses, color = 'blue', label = 'Train')
    plt.plot(v_losses, color = 'red', label = 'Validation')
    plt.title('Loss')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend()

    plt.subplot(1,2,2)
    plt.plot(t_accs, color = 'blue', label = 'Train')
    plt.plot(v_accs, color = 'red', label = 'Validation')
    plt.title('Accuracy')
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.legend()

    metrics_path = os.path.join(exp_path, "metrics.png")
    fig.savefig(metrics_path)



def train_model(model, criterion, optimizer, scheduler, num_epochs=10):
    train_losses = []
    train_accuracies = []
    val_losses = []
    val_accuracies = []
    
    timer = time.time()

    best_model = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print(f"Epoch {epoch + 1}/{num_epochs}")
        print("-" * 10)

        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()
            
            running_loss = 0.0
            running_corrects = 0

            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print(f"{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}")

            if phase == 'train':
                train_losses.append(epoch_loss)
                train_accuracies.append(epoch_acc)
            else:
                val_losses.append(epoch_loss)
                val_accuracies.append(epoch_acc)
            
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model = copy.deepcopy(model.state_dict())

    time_elapsed = time.time() - timer
    print(f"Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s")
    print(f"Best Val acc: {best_acc:.4f}")

    model.load_state_dict(best_model)
    
    torch.save({'state_dict': model.state_dict()}, model_path) 

    return model, train_losses, train_accuracies, val_losses, val_accuracies


# Hyperparameters
hyperparameters = {
    'epochs': 5,
    'opt_lr': 0.05,
    'opt_momentum': 0.9,
    'sch_gamma': 0.15,
    'sch_step': 3
}


# Create model ResNet
model_res = models.resnet18(pretrained = True)
for param in model_res.parameters():
    param.requires_grad = False

filters = model_res.fc.in_features
model_res.fc = nn.Linear(filters, len(train_dataset.classes))

model_res.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model_res.fc.parameters(), lr = hyperparameters['opt_lr'], momentum = hyperparameters['opt_momentum'])
scheduler = lr_scheduler.StepLR(optimizer, step_size = hyperparameters['sch_step'], gamma = hyperparameters['sch_gamma'])

# Train the model
model_res, t_losses, t_accs, v_losses, v_accs = train_model(model_res, criterion, optimizer, scheduler, hyperparameters['epochs'])

# Save experiment
experiment_name = datetime.datetime.utcnow().strftime('%Y%m%d-%H%M%S')
ex_path = os.path.join(experiments_path, experiment_name)
save_experiment(ex_path, hyperparameters)
save_metrics(ex_path, t_losses, v_losses, t_accs, v_accs)
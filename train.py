import torch
import torch.nn as nn
import torchvision
from torchvision import datasets, models, transforms
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np

import os
import time
import datetime
import copy

from utils import save_hyperparameters, save_metrics, save_predictions, fix_colormap

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

# Configure paths
data_base_path = "data"
train_path = os.path.join(data_base_path, "train")
val_path = os.path.join(data_base_path, "validation")
model_path = 'cat-model.pth.tar'
experiments_path = "experiments" # Path to save the current values of hyperparameters and metrics for each experiment

# Configure datasets
mean_norm = [0.485, 0.456, 0.406]
std_norm = [0.229, 0.224, 0.225]

transform = transforms.Compose([
    transforms.Resize(720),
    transforms.RandomResizedCrop(size=224),
    transforms.RandomRotation(degrees=15),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean_norm, std_norm)
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

# Make some predictions over the validation dataset
def validate_model(model, num_samples=1):
    predictions = []
    model.eval()
    
    with torch.no_grad():
        for i, (inputs, labels) in enumerate(dataloaders['val']):
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)

            # Get probabilities
            sm = nn.Softmax(dim=1)
            probs = sm(outputs)

            _, preds = torch.max(outputs, 1)

            for x in range(num_samples):
                sample_prob = probs[x].cpu().detach().numpy()[preds[x]] * 100
                #print(f"Predicted:{train_dataset.classes[preds[x]]} ({sample_prob:.2f}%)\nExpected: {train_dataset.classes[labels[x]]}")

                fixed_image = fix_colormap(inputs.cpu().data[x], mean_norm, std_norm)
                
                # (image, class_predicted, probability, class_expected)
                prediction = (fixed_image, train_dataset.classes[preds[x]], sample_prob, train_dataset.classes[labels[x]])
                predictions.append(prediction)

                # Exit the loop when we have the predictions required
                if len(predictions) == num_samples: break

            if len(predictions) == num_samples: break
    
    return predictions

## Start training and save the results of the experiment
def run_experiment(model, hyperparameters):
    # Train the model
    print("\nStart training process...\n")
    model, t_losses, t_accs, v_losses, v_accs = train_model(model, criterion, optimizer, scheduler, hyperparameters['epochs'])
    print("\nTraining process finished\n")

    # Validate model
    print("\nValidating model...", end='')
    predictions = validate_model(model, num_samples=6)
    print("OK")

    # Save experiment
    experiment_name = datetime.datetime.utcnow().strftime('%Y%m%d-%H%M%S')
    ex_path = os.path.join(experiments_path, experiment_name)
    print(f"Saving experiment results in {ex_path}...", end='')
    save_hyperparameters(ex_path, hyperparameters)
    save_metrics(ex_path, t_losses, v_losses, t_accs, v_accs)
    save_predictions(ex_path, predictions)
    print("OK")
    

# Hyperparameters configuration
hyperparameters = {
    'epochs': 1,
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

# Start experiment
run_experiment(model_res, hyperparameters)



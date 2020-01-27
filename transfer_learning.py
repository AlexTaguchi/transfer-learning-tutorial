# Modules
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy

# Parameters
directory = 'data'
epochs = 20

# Normalize data and augmentation for training
data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

# Read image data directory
image_datasets = {x: datasets.ImageFolder(os.path.join(directory, x), data_transforms[x])
                  for x in ['train', 'val']}
dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
class_names = image_datasets['train'].classes

# Initialize dataloader
dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=10, shuffle=True, num_workers=4)
               for x in ['train', 'val']}

# Run on GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Train model
def train_model(model, criterion, optimizer, scheduler, epochs=10):

    # Record start time of training
    start_time = time.time()

    # Preallocate best model weights and accuracy
    best_model_weights = copy.deepcopy(model.state_dict())
    best_accuracy = 0.0

    # Train over multiple epochs
    for epoch in range(1, epochs + 1):
        print(f'Epoch {epoch}/{epochs}')
        print('-' * 10)

        # Start with train phase then switch to validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            # Preallocate the running loss and number of correct predictions
            running_loss = 0.0
            running_corrects = 0

            # Pass inputs through model
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # Zero the parameter gradients
                optimizer.zero_grad()

                # Remember gradients for backpropagation only in training phase
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # Optimize model only in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # Increment running statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
            
            # Schedule optimizer learning rate
            if phase == 'train':
                scheduler.step()

            # Report loss and accuracy of current phase
            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_accuracy = running_corrects.double() / dataset_sizes[phase]
            print(f'{phase.capitalize()} Loss: {epoch_loss:.4f} Accuracy: {epoch_accuracy:.4f}')

            # Keep track of the best running model on the validation set
            if phase == 'val' and epoch_accuracy > best_accuracy:
                best_accuracy = epoch_accuracy
                best_model_weights = copy.deepcopy(model.state_dict())
        print()

    # Report final model performance
    time_elapsed = time.time() - start_time
    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best Validation Accuracy: {best_accuracy:4f}')

    # Return best model weights
    model.load_state_dict(best_model_weights)
    return model

# Visualize model
def visualize_model(model):

    # Remember model training mode and switch to evaluation mode
    training_mode = model.training
    model.eval()

    # Pass a batch of validation inputs without recording gradients
    plt.figure()
    with torch.no_grad():
        for i, (inputs, labels) in enumerate(dataloaders['val']):

            # Predict labels from inputs
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            # Plot images and predicted labels
            if i < 9:
                ax = plt.subplot(3, 3, i + 1)
                ax.axis('off')
                ax.set_title('Predicted: {}'.format(class_names[preds[i]]))
                inp = inputs.cpu().data[i].numpy().transpose((1, 2, 0))
                mean = np.array([0.485, 0.456, 0.406])
                std = np.array([0.229, 0.224, 0.225])
                inp = inp * std + mean
                inp = np.clip(inp, 0, 1)
                plt.imshow(inp)
                plt.pause(0.001)
            else:
                break

        # Return model back to original training mode
        model.train(mode=training_mode)

# Load pretrained model
model = models.resnet18(pretrained=True)

# Replace the final fully connected layer for one with the correct number of outputs
features = model.fc.in_features
model.fc = nn.Linear(features, len(class_names))
model = model.to(device)

# Set up loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

# Decay learning rate by a factor of 0.1 every 5 epochs
scheduler = lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

# Train model
model = train_model(model, criterion, optimizer, scheduler, epochs=epochs)

# Visualize model performance
visualize_model(model)
plt.show()

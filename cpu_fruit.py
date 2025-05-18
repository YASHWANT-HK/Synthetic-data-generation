import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import time
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, precision_score, f1_score
import pandas as pd
from torchvision import models

# Force CPU usage
device = torch.device("cpu")

# Define dataset path
dataset_dir = r"C:/Yashwanth/Mini Project/dataset/New dataset_2/Apple Golden 2"  # Change this to your dataset location

# Define Data Augmentation & Normalization
transform = transforms.Compose([
    transforms.Resize((224, 224)),  
    transforms.RandomHorizontalFlip(),  
    transforms.RandomRotation(30),  
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),  
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# Load dataset
train_dataset = datasets.ImageFolder(root=dataset_dir, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)

# Load Pretrained ResNet Model
model = models.resnet18(pretrained=True)  
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, len(train_dataset.classes))  # Modify last layer for classification
model = model.to(device)

# Loss and Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001, weight_decay=1e-4)

# Training Function
def train_model(epochs=10):
    model.to(device)
    start_time = time.time()
    
    for epoch in range(epochs):
        model.train()
        all_preds, all_labels = [], []
        
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Store predictions and labels for metrics
            _, predicted = torch.max(outputs, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

        # Calculate accuracy after each epoch
        epoch_accuracy = accuracy_score(all_labels, all_preds)
        print(f"Epoch [{epoch+1}/{epochs}], Accuracy: {epoch_accuracy:.4f}")

    end_time = time.time()
    
    # Compute final metrics
    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average='weighted', zero_division=1)
    f1 = f1_score(all_labels, all_preds, average='weighted', zero_division=1)
    processing_time = end_time - start_time

    return accuracy, precision, f1, processing_time

# Run training on CPU
cpu_results = train_model(epochs=100)  # Train for 15 epochs

# Display results
results_df = pd.DataFrame({
    "Device": ["CPU"],
    "Accuracy": [cpu_results[0]],
    "Precision": [cpu_results[1]],
    "F1-Score": [cpu_results[2]],
    "Processing Time (s)": [cpu_results[3]]
})

print(results_df)
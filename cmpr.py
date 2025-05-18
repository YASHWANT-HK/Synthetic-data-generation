import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import time
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, precision_score, f1_score
import pandas as pd
from torchvision import models

# Force CPU usage
device = torch.device("cpu")
print(f"Using device: {device}")

# Define dataset path (Update this to your dataset location)
dataset_dir =r"C:/Yashwanth/Mini Project/dataset/New dataset_2/Apple Golden 2"
# Define Data Augmentation & Normalization
transform = transforms.Compose([
    transforms.Resize((224, 224)),  
     
    transforms.RandomHorizontalFlip(),  
    transforms.RandomRotation(20),  
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),  
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
]) 

# Load dataset
train_dataset = datasets.ImageFolder(root=dataset_dir, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)

# Define Models for Comparison
models_dict = {
    "MobileNetV2": models.mobilenet_v2(pretrained=True),
    "ResNet-18": models.resnet18(pretrained=True),
    "EfficientNet-B0": models.efficientnet_b0(pretrained=True)
}

# Modify output layers for classification
num_classes = len(train_dataset.classes)

for model_name, model in models_dict.items():
    if model_name == "MobileNetV2" or model_name == "EfficientNet-B0":
        num_ftrs = model.classifier[1].in_features  # Correct for MobileNet & EfficientNet
        model.classifier[1] = nn.Linear(num_ftrs, num_classes)
    else:  # ResNet-18 uses 'fc'
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, num_classes)


# Loss function
criterion = nn.CrossEntropyLoss()

# Training Function
def train_model(model, model_name, epochs=10):
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.0005, weight_decay=1e-4)
    
    print(f"\nTraining {model_name}...\n")
    start_time = time.time()
    epoch_accuracies = []

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
        epoch_accuracies.append(epoch_accuracy)
        print(f"{model_name} - Epoch [{epoch+1}/{epochs}], Accuracy: {epoch_accuracy:.4f}")

    end_time = time.time()
    
    # Compute final metrics
    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average='weighted', zero_division=1)
    f1 = f1_score(all_labels, all_preds, average='weighted', zero_division=1)
    processing_time = end_time - start_time

    return accuracy, precision, f1, processing_time, epoch_accuracies

# Train and Evaluate All Models
results = []
all_accuracies = {}  # Store accuracy per epoch for visualization

for model_name, model in models_dict.items():
    acc, prec, f1, time_taken, epoch_accs = train_model(model, model_name, epochs=10)
    results.append([model_name, acc, prec, f1, time_taken])
    all_accuracies[model_name] = epoch_accs  # Store per-epoch accuracy

# Display results in a DataFrame
results_df = pd.DataFrame(results, columns=["Model", "Accuracy", "Precision", "F1-Score", "Processing Time (s)"])
print("\nFinal Comparison Report:")
print(results_df)

# Plot Training Curves
plt.figure(figsize=(10, 6))
for model_name, accs in all_accuracies.items():
    plt.plot(range(1, len(accs) + 1), accs, label=model_name)

plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Training Accuracy per Epoch")
plt.legend()
plt.show()
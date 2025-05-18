import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import time
import pandas as pd
from torch.utils.data import DataLoader, random_split
from sklearn.metrics import accuracy_score, precision_score, f1_score
from torchvision import models
from PIL import Image
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Ensure the model only runs on GPU
if not torch.cuda.is_available():
    raise RuntimeError("CUDA is not available. Please run the script on a system with a GPU.")

device = torch.device("cuda")
logging.info(f"Using device: {device}")

# Define dataset path
dataset_dir = r"C:/Yashwanth/Mini Project/dataset/New dataset_2/Apple Golden 2"  # Update this path if needed

# Define Data Augmentation & Normalization
transform = transforms.Compose([
    transforms.Resize((224, 224)),  
    transforms.RandomHorizontalFlip(),  
    transforms.RandomRotation(30),  
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),  
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# Load dataset and split into train and validation sets
full_dataset = datasets.ImageFolder(root=dataset_dir, transform=transform)
train_size = int(0.8 * len(full_dataset))  # 80% Training, 20% Validation
val_size = len(full_dataset) - train_size

train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, pin_memory=True)

# Load Pretrained ResNet Model
model = models.resnet18(pretrained=True).to(device)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, len(full_dataset.classes)).to(device)  # Modify last layer for classification

# Loss and Optimizer
criterion = nn.CrossEntropyLoss().to(device)
optimizer = optim.Adam(model.parameters(), lr=0.0001, weight_decay=1e-4)

# Training Function
def train_model(epochs=100):  # Updated to 100 epochs
    model.train()
    start_time = time.time()
    
    for epoch in range(epochs):
        all_preds, all_labels = [], []
        running_loss = 0.0
        
        for images, labels in train_loader:
            images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
        
        epoch_accuracy = accuracy_score(all_labels, all_preds)
        logging.info(f"Epoch [{epoch+1}/{epochs}], Loss: {running_loss/len(train_loader):.4f}, Accuracy: {epoch_accuracy:.4f}")

    end_time = time.time()
    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average='weighted', zero_division=1)
    f1 = f1_score(all_labels, all_preds, average='weighted', zero_division=1)
    processing_time = end_time - start_time

    torch.save(model.state_dict(), "gpu_trained_model.pth")
    logging.info("Model saved as 'gpu_trained_model.pth'")

    return accuracy, precision, f1, processing_time

# Run training on GPU
logging.info("Training Model...")
train_results = train_model(epochs=100)

# Validation Function
def validate_model():
    model.eval()
    all_preds, all_labels = [], []
    start_time = time.time()

    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    end_time = time.time()
    val_accuracy = accuracy_score(all_labels, all_preds)
    val_precision = precision_score(all_labels, all_preds, average='weighted', zero_division=1)
    val_f1 = f1_score(all_labels, all_preds, average='weighted', zero_division=1)
    validation_time = end_time - start_time

    return val_accuracy, val_precision, val_f1, validation_time

# Run Validation
logging.info("Validating Model...")
val_accuracy, val_precision, val_f1, val_time = validate_model()

# Display Training & Validation Results
results_df = pd.DataFrame({
    "Metric": ["Accuracy", "Precision", "F1-Score", "Processing Time (s)"],
    "Training Score": [train_results[0], train_results[1], train_results[2], train_results[3]],
    "Validation Score": [val_accuracy, val_precision, val_f1, val_time]
})

logging.info("Training vs Validation Results:")
logging.info(f"\n{results_df}")

# Prediction Function
def predict_flower(image_path):
    model.load_state_dict(torch.load("gpu_trained_model.pth", map_location=device))
    model.eval()
    
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    try:
        image = Image.open(image_path).convert("RGB")
        image = preprocess(image).unsqueeze(0).to(device, non_blocking=True)
        
        start_time = time.time()
        with torch.no_grad():
            outputs = model(image)
            probabilities = nn.functional.softmax(outputs, dim=1)
            predicted_class = torch.argmax(probabilities, dim=1).item()
        end_time = time.time()

        class_names = full_dataset.classes  # Uses dataset class names
        logging.info(f"Predicted Flower: {class_names[predicted_class]}")
        logging.info(f"Prediction Time: {end_time - start_time:.4f} seconds")
    except Exception as e:
        logging.error(f"Error: {e}. Please provide a valid image path!")
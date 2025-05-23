# Synthetic-data-generation
Synthetic image data generation of apple fruit  using nvidia isaac sim.


Dataset:  https://drive.google.com/drive/folders/1QFVp-7KVCJVHMZZjs65M5vS9GTiHrkZf?usp=drive_link

INSTRUCTION:

 Deep Learning Model Comparison on Custom Image Dataset (GPU-Accelerated)
This project compares the performance of MobileNetV2, ResNet-18, and EfficientNet-B0 on a custom image classification dataset using PyTorch. It includes training, evaluation, and visualization of model accuracy per epoch.

📁 Project Structure
bash
Copy
Edit
project-root/
│
├── train.py                  # Main training and evaluation script
├── requirements.txt          # Dependencies
└── dataset/
    └── class1/
    └── class2/
    └── ...
📦 Requirements
Ensure you have Python 3.7 or later installed.

Install Dependencies
bash
Copy
Edit
pip install -r requirements.txt
requirements.txt:

txt
Copy
Edit
torch
torchvision
matplotlib
pandas
scikit-learn
tabulate
✅ If using a GPU, install the CUDA-enabled version of PyTorch from https://pytorch.org/get-started/locally

💾 Dataset Setup
Your dataset should be structured like this:

Copy
Edit
dataset/
├── cats/
│   ├── cat1.jpg
│   └── ...
├── dogs/
│   ├── dog1.jpg
│   └── ...
Update the path in train.py:

python
Copy
Edit
dataset_dir = r"C:\path\to\your\dataset"
Or use a relative path:

python
Copy
Edit
dataset_dir = "./dataset"
🛠️ How to Run (VS Code Instructions)
1. Open Project in VS Code
Launch VS Code.

Open this folder: File > Open Folder > project-root

2. Select Python Interpreter
Press Ctrl + Shift + P → Select Python: Select Interpreter

Choose the Python environment where your dependencies are installed.

3. Run the Script
Option 1: Using the Terminal
bash
Copy
Edit
python train.py
Option 2: Using the VS Code Run Button
Open train.py

Click ▶️ Run Python File at the top right

🖥️ Training Overview
Trains 3 models (MobileNetV2, ResNet-18, EfficientNet-B0) for 10 epochs each

Uses GPU (if available)

Displays:

Accuracy

Precision

F1-score

Time taken per model

Visualizes training accuracy per epoch

📊 Output Example
Terminal shows final results in a tabulated format

A matplotlib plot displays training accuracy curves

✅ Sample Output (Tabulated)
sql
Copy
Edit
 Final GPU Model Performance Report:
+------------------+------------+------------+-----------+----------------------+
| Model            | Accuracy   | Precision  | F1-Score  | Processing Time (s) |
+------------------+------------+------------+-----------+----------------------+
| MobileNetV2      | 0.8750     | 0.8801     | 0.8732    | 24.32                |
| ResNet-18        | 0.8650     | 0.8705     | 0.8610    | 27.91                |
| EfficientNet-B0  | 0.8900     | 0.8923     | 0.8881    | 29.15                |
+------------------+------------+------------+-----------+----------------------+
📈 Accuracy per Epoch
A plot will be shown at the end of training that looks like this:

X-axis: Epochs (1–10)

Y-axis: Accuracy

Lines: Each model’s performance across epochs

🧠 Notes
Uses data augmentation (RandomHorizontalFlip, RandomRotation, ColorJitter)

All models are pre-trained on ImageNet

Final classifier layers are replaced to match the number of dataset classes

⚙️ Optional: Virtual Environment Setup
bash
Copy
Edit
python -m venv venv
source venv/bin/activate  # macOS/Linux
venv\Scripts\activate     # Windows

pip install -r requirements.txt

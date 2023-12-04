import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torchvision import transforms
import os
import pickle
import numpy as np
from tqdm import tqdm

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
BATCH_SIZE = 32
pwd = os.getcwd()

# Load the dataset directly from the pickle file
dataset_file_path = pwd + "/CNN_road_sign/road_signs_dataset.pkl"

# Verify that the dataset file exists
if not os.path.exists(dataset_file_path):
    raise FileNotFoundError(f"Dataset file not found: {dataset_file_path}")

# Load the dataset directly from the pickle file
with open(dataset_file_path, 'rb') as file:
    dataset = pickle.load(file)

# Verify that the dataset contains images and labels
images = [data['image'] for data in dataset]
labels = [data['label'] for data in dataset]

# Convert to PyTorch tensors
images = torch.stack([torch.from_numpy(np.array(img)).unsqueeze(0) for img in images])
labels = torch.tensor(labels)

# Ensure all images are single-channel (grayscale)
if images.size(1) > 1:
    # Convert RGB to grayscale
    images = images.mean(dim=1, keepdim=True)

# Create a PyTorch dataset
torch_dataset = TensorDataset(images, labels)

# Create a data loader
train_loader = DataLoader(torch_dataset, batch_size=BATCH_SIZE, shuffle=True)

class RoadSignCNN(nn.Module):
    def __init__(self, num_classes):
        super(RoadSignCNN, self).__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),

            nn.Conv2d(32, 64, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),

            nn.Flatten(),
            nn.Linear(64 * (224 // 4) * (224 // 4), 256),
            nn.Linear(256, 256),
            nn.Linear(256, num_classes)
        )
        self.loss = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.parameters(), lr=0.001)

    def forward(self, inputs):
        return self.layers(inputs)

    def train_step(self, inputs, labels):
        self.optimizer.zero_grad()
        pred = self.forward(inputs)

        # Ensure the adjusted labels are within the correct range [0, C-1]
        # num_classes = self.layers[-1].out_features
        # adjusted_labels = torch.clamp(labels, 0, num_classes - 1)
        adjusted_labels = torch.clamp(labels, 0, 3)

        loss = self.loss(pred, adjusted_labels)
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def predict(self, input):
        with torch.no_grad():
            pred = self.forward(input)
            return torch.argmax(pred, axis=-1)

    def save_model(self, model_path):
        torch.save(self.state_dict(), model_path)
        print(f"Model saved to {model_path}")

    def load_model(self, model_path):
        self.load_state_dict(torch.load(model_path))
        print(f"Model loaded from {model_path}")

# Instantiate the model using the number of classes from the training set
# model = RoadSignCNN(num_classes=len(torch.unique(labels)))
model = RoadSignCNN(num_classes=len(torch.unique(labels)))
model.to(DEVICE)

# Check if the model has already been trained
model_saved_path = pwd + "/CNN_road_sign/road_sign_model.pth"
if os.path.exists(model_saved_path):
    # Load the trained model if it exists
    model.load_model(model_saved_path)
else:
    # Train the model if it has not been trained yet
    EPOCHS = 10
    for epoch in range(EPOCHS):
        total_loss = 0
        for inputs, labels in tqdm(train_loader):
            inputs = inputs.to(DEVICE)
            labels = labels.to(DEVICE)

            loss = model.train_step(inputs, labels)
            total_loss += loss
        print("EPOCH:", epoch + 1, ": ", total_loss)

    # Save the trained model
    model.save_model(model_saved_path)

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torchvision import transforms
import os
import pickle
import numpy as np
from sklearn.model_selection import train_test_split
from tqdm import tqdm

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
BATCH_SIZE = 32
pwd = os.getcwd()

# Load the dataset directly from the pickle file
train_dataset_file_path = pwd + "/CNN_road_sign/road_signs_dataset.pkl"

# Verify that the dataset file exists
if not os.path.exists(train_dataset_file_path):
    raise FileNotFoundError(f"Dataset file not found: {train_dataset_file_path}")

# Load the dataset directly from the pickle file
with open(train_dataset_file_path, 'rb') as file:
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

# Split the dataset into train and eval sets
train_images, eval_images, train_labels, eval_labels = train_test_split(
    images, labels, test_size=0.1, random_state=42
)

# Create PyTorch datasets for train and eval sets
train_dataset = TensorDataset(train_images, train_labels)
eval_dataset = TensorDataset(eval_images, eval_labels)

# Create data loaders for train and eval sets
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
eval_loader = DataLoader(eval_dataset, batch_size=BATCH_SIZE, shuffle=False)

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
        adjusted_labels = torch.clamp(labels, 0, 3)

        loss = self.loss(pred, adjusted_labels)
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def predict(self, input):
        with torch.no_grad():
            pred = self.forward(input)
            return torch.argmax(pred, dim=-1)

    def evaluate(self, dataloader):
        self.eval()  # Set the model to evaluation mode
        correct_predictions = 0
        total_samples = 0

        with torch.no_grad():
            for inputs, labels in tqdm(dataloader):
                inputs = inputs.to(DEVICE)
                labels = labels.to(DEVICE)

                predictions = self.predict(inputs)
                correct_predictions += torch.sum(predictions == labels).item()
                total_samples += labels.size(0)

        accuracy = correct_predictions / total_samples
        print("Accuracy: {:.2%}".format(accuracy))
        return accuracy

    def save_model(self, model_path):
        torch.save(self.state_dict(), model_path)
        print(f"Model saved to {model_path}")

    def load_model(self, model_path):
        checkpoint = torch.load(model_path)

        # Adjust the model architecture if needed
        if 'layers.9.weight' in checkpoint and checkpoint['layers.9.weight'].shape[0] != self.layers[9].weight.shape[0]:
            # Modify the model architecture to match the checkpoint
            # Add or remove layers as needed
            num_classes = checkpoint['layers.9.weight'].shape[0]
            self.layers[9] = nn.Linear(checkpoint['layers.9.weight'].shape[1], num_classes)

        self.load_state_dict(checkpoint)
        print(f"Model loaded from {model_path}")

# Instantiate the model using the number of classes from the training set
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

    # Evaluate the model on the evaluation set
    model.evaluate(eval_loader)

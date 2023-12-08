import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torchvision import transforms
from torch.optim.lr_scheduler import ReduceLROnPlateau
import os
import pickle
import numpy as np
from sklearn.model_selection import train_test_split
from tqdm import tqdm

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# DEVICE = torch.device('cpu')
BATCH_SIZE = 32
pwd = os.getcwd()

# Load the dataset directly from the pickle file
train_dataset_file_path = pwd + "/CNN_road_sign/road_signs_dataset.pkl"
# train_dataset_file_path = pwd + "/road_signs_dataset.pkl"

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

my_transform = transforms.Compose([
    transforms.RandomRotation(10),
    transforms.RandomHorizontalFlip(),
])

# Create PyTorch datasets for train and eval sets
train_dataset = TensorDataset(my_transform(train_images), train_labels)
eval_dataset = TensorDataset(my_transform(eval_images), eval_labels)

# Create data loaders for train and eval sets
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
eval_loader = DataLoader(eval_dataset, batch_size=BATCH_SIZE, shuffle=False)

class RoadSignCNN(nn.Module):
    def __init__(self, num_classes):
        super(RoadSignCNN, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),

            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
        )

        # Dummy input to calculate the output size of the convolutional layers
        dummy_input = torch.rand(1, 1, 224, 224)
        conv_output_size = self._get_conv_output_size(dummy_input)

        self.fc_layers = nn.Sequential(
            nn.Linear(conv_output_size, 256),
            nn.ReLU(),
            nn.Dropout(0.5),

            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Dropout(0.5),

            nn.Linear(256, num_classes)
        )

        self.num_classes = num_classes  # Added attribute to store the number of classes
        self.loss = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.parameters(), lr=0.001)
        self.scheduler = ReduceLROnPlateau(self.optimizer, mode='max', patience=5, verbose=True)

    def _get_conv_output_size(self, x):
        x = self.conv_layers(x)
        return x.view(x.size(0), -1).shape[1]

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)
        x = self.fc_layers(x)
        return x

    def train_step(self, inputs, labels):
        self.optimizer.zero_grad()
        pred = self.forward(inputs)

        # Ensure the adjusted labels are within the correct range [0, C-1]
        adjusted_labels = torch.clamp(labels, 0, self.num_classes - 1)

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
        # Save additional information about the number of classes
        checkpoint = {
            'num_classes': self.num_classes,
            'model_state_dict': self.state_dict()
        }
        torch.save(checkpoint, model_path)
        print(f"Model saved to {model_path}")

    def set_num_classes(self, num_classes):
        # Dynamically adjust the fully connected layers based on the output size of conv_layers
        dummy_input = torch.rand(1, 1, 224, 224)
        conv_output_size = self._get_conv_output_size(dummy_input.to(DEVICE))

        self.fc_layers[0] = nn.Linear(conv_output_size, 256)
        self.fc_layers[-1] = nn.Linear(256, num_classes)

        self.num_classes = num_classes  # Update the number of classes

        # Update the loss and optimizer to reflect the new architecture
        self.loss = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.parameters(), lr=0.001)
        self.scheduler = ReduceLROnPlateau(self.optimizer, mode='max', patience=5, verbose=True)

    def load_model(self, model_path):
        checkpoint = torch.load(model_path)

        # Extract the number of classes from the checkpoint
        num_classes = checkpoint.get('num_classes', None)

        if num_classes is None:
            raise ValueError("Unable to determine the number of classes from the checkpoint.")

        # Update the model to reflect the new number of classes
        self.set_num_classes(num_classes)

        # Load the model state dict
        self.load_state_dict(checkpoint['model_state_dict'])
        self.to(DEVICE)
        print(f"Model loaded from {model_path}")

# Instantiate the model using the number of classes from the training set
num_classes = len(torch.unique(labels)) - 1  # Subtract 1 to match the correct number of classes
dummy_input = torch.rand(1, 1, 224, 224)  # Use a dummy input
model = RoadSignCNN(num_classes=num_classes)
model.to(DEVICE)

# Check if the model has already been trained
model_saved_path = pwd + "/CNN_road_sign/road_sign_model.pth"
# model_saved_path = pwd + "/road_sign_model.pth"
if os.path.exists(model_saved_path):
    # Load the trained model if it exists
    print("I AM HERE")
    model.load_model(model_saved_path)
    model.to(DEVICE)
else:
    # Train the model if it has not been trained yet
    EPOCHS = 10
    print("DEVICE IS",DEVICE)
    for epoch in range(EPOCHS):
        total_loss = 0
        for inputs, labels in tqdm(train_loader):
            inputs = inputs.to(DEVICE)
            labels = labels.to(DEVICE)

            loss = model.train_step(inputs, labels)
            total_loss += loss
        print("EPOCH:", epoch + 1, ": ", total_loss)

        # model.evaluate(eval_loader)
    # Save the trained model
    model.save_model(model_saved_path)

    # Evaluate the model on the evaluation set

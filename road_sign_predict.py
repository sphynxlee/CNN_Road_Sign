import torch
from torchvision import transforms
from PIL import Image
import os
from road_sign_CNN import RoadSignCNN  # Assuming your model class is defined in road_sign_CNN.py

pwd = os.getcwd()

# Instantiate the model using the RoadSignCNN class
model = RoadSignCNN(num_classes=4)

# Load the trained model
model_path = pwd + "/CNN_road_sign/road_sign_model.pth"
model.load_model(model_path)

# Prepare input data
# For example, load an image using PIL and convert it to the required format
# input_image_path = pwd + "/CNN_road_sign/test_images/stop01.png"

# Modified -> function to call with an image path
def Road_Sign_Predict(input_image_path):
    input_image = Image.open(input_image_path).convert("L")  # Convert to grayscale
    input_image = transforms.ToTensor()(input_image).unsqueeze(0)  # Convert to tensor and add batch dimension

    # Perform prediction
    with torch.no_grad():
        model.eval()  # Set the model to evaluation mode
        prediction = model.predict(input_image)

    print("Predicted Class:", prediction.item())
    return prediction.item()
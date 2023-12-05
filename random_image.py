import os
import random
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

folder_path = "road_signs_img"

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


# List all files and find png files from road_signs_img folder
def random_image():
    file_list = os.listdir(folder_path)
    png_files = [file for file in file_list if file.lower().endswith(".png")]

    random_file_path = None  # Initialize the variable

    if png_files:
        random_file = random.choice(png_files)
        random_file_path = os.path.join(folder_path, random_file)
        print("chosen file:", random_file_path)
    else:
        print("No PNG files")

    if random_file_path is not None:
        prediction = Road_Sign_Predict(random_file_path)
        print("predicted sign is: ", prediction)
        return prediction
    else:
        # Handle the case where random_file_path is not defined
        print("Unable to make a prediction because no PNG files were found.")
        return None

random_image()
import os
import random
import torch
from torchvision import transforms
from PIL import Image
from road_sign_CNN import RoadSignCNN  # Assuming your model class is defined in road_sign_CNN.py

pwd = os.getcwd()

TAG = '=======================random_image ===================='

# Adjust the number of classes accordingly
model = RoadSignCNN(num_classes=4)
# model.to('cuda')

# Load the trained model
# windows platform:
# model_path = pwd + "/CNN_road_sign/road_sign_model.pth"
model_path = os.path.join(pwd, 'road_sign_model.pth')
# model.load_model(model_path)
# model.to("cuda")

# windows platform:
# folder_path = pwd + "/CNN_road_sign/road_signs_img/"
folder_path = os.path.join(pwd, 'road_signs_img')

# Prepare input data
# For example, load an image using PIL and convert it to the required format
# input_image_path = pwd + "/CNN_road_sign/test_images/stop01.png"

def Road_Sign_Predict(input_image_path):
    input_image = Image.open(input_image_path).convert("L")  # Convert to grayscale

    # Convert the PIL Image to a PyTorch tensor
    input_image = transforms.ToTensor()(input_image)

    # Add batch dimension
    input_image = input_image.unsqueeze(0)

    # Resize the image to the expected size (224x224)
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
    ])
    input_image = transform(input_image)

    # Perform prediction
    with torch.no_grad():
        # model.eval()  # Set the model to evaluation mode
        prediction = model.predict(input_image)

    print(TAG, "Predicted Class:", prediction.item())
    return prediction.item()

# List all files and find png files from road_signs_img folder
def random_image():
    sign_folder_list = os.listdir(folder_path)

    for folder in sign_folder_list:
        folder_path_full = os.path.join(folder_path, folder)
        png_files = [file for file in os.listdir(folder_path_full) if file.lower().endswith(".png")]
        # print(TAG, "PNG files:", png_files)

        random_file_path = None  # Initialize the variable

        if png_files:
            random_file = random.choice(png_files)
            random_file_path = os.path.join(folder_path_full, random_file)
            print(TAG, "chosen file:", random_file_path)

            # Perform prediction
            prediction = Road_Sign_Predict(random_file_path)
            print(TAG, "predicted sign is: ", prediction)
            return prediction
        else:
            print(TAG, "No PNG files in folder:", folder)

    # Handle the case where no PNG files were found
    print(TAG, "Unable to make a prediction because no PNG files were found.")
    return None

# receive an image path and return the prediction
def predict_image(image_path):
    # Perform prediction
    prediction = Road_Sign_Predict(image_path)
    print(TAG, "predicted sign is: ", prediction)
    return prediction

# windows platform:
# # for i in range(20):
# predict_image(fr"C:\Users\Avina\OneDrive - Nova Scotia Community College\NSCC_ProgrammingForAI\GroupAssignment\gitmainfolder\CNN_Road_Sign\road_signs_img\2 speedlimit\40_01.png")
# predict_image(fr"C:\Users\Avina\OneDrive - Nova Scotia Community College\NSCC_ProgrammingForAI\GroupAssignment\gitmainfolder\CNN_Road_Sign\road_signs_img\0 TrafficLight\road3_trafficlight.png")

random_image()
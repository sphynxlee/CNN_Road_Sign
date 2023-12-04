# https://stackoverflow.com/questions/53754662/mnist-dataset-structure
# I take the above link as a reference to create the following code.
# It is used to load and normalize images in the same way as the MNIST dataset.

import os
import pickle
import numpy as np
from PIL import Image

def resize_images_in_folder(input_folder, output_folder, target_size):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for class_folder in os.listdir(input_folder):
        class_path = os.path.join(input_folder, class_folder)
        output_class_path = os.path.join(output_folder, class_folder)

        if not os.path.exists(output_class_path):
            os.makedirs(output_class_path)

        for image_name in os.listdir(class_path):
            image_path = os.path.join(class_path, image_name)
            output_path = os.path.join(output_class_path, "resized_" + image_name)

            img = Image.open(image_path)
            img_resized = img.resize(target_size, Image.Resampling.LANCZOS)

            img_resized.save(output_path)

def preprocess_and_save_dataset(input_folder, output_file):
    dataset = []

    for class_folder in os.listdir(input_folder):
        class_path = os.path.join(input_folder, class_folder)

        for image_name in os.listdir(class_path):
            image_path = os.path.join(class_path, image_name)

            # Load and resize image
            img = Image.open(image_path).convert("L")  # Convert to grayscale

            # Convert image to NumPy array and normalize pixel values to [0, 1]
            img_array = np.array(img, dtype=np.float32) / 255.0

            # Extract label from folder name
            label_str = class_folder.split()[0]
            label = int(label_str)

            # Combine image and label into a dictionary
            data = {"image": img_array, "label": label}

            # Append to the dataset
            dataset.append(data)

    # Save the dataset to a file using pickle
    with open(output_file, 'wb') as file:
        pickle.dump(dataset, file)

# Paths
input_folder_resize = os.getcwd() + '/CNN_road_sign/road_signs_img'
output_folder_resize = os.getcwd() + '/CNN_road_sign/resized_images'
output_file = os.getcwd() + '/CNN_road_sign/road_signs_dataset.pkl'
target_size = (224, 224)

# Resize Images
resize_images_in_folder(input_folder_resize, output_folder_resize, target_size)

# Preprocess and Save Dataset
preprocess_and_save_dataset(output_folder_resize, output_file)

import os
import random

folder_path = "test_images"

# List all files and find png files
file_list = os.listdir(folder_path)
png_files = [file for file in file_list if file.lower().endswith(".png")]

if png_files:
    random_file = random.choice(png_files)
    random_file_path = os.path.join(folder_path, random_file)

    print("chosen file:", random_file_path)
else:
    print("No PNG files")

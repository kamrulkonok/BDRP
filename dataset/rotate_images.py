import os
import cv2
import numpy as np
from pathlib import Path

# Specify input and output paths
input_path = "/gpfs/workdir/islamm/datasets_without_NoFindings"
output_path = "/gpfs/workdir/islamm/rotated_datasets_without_NoFindings"

# Create the output folder if it doesn't exist
os.makedirs(output_path, exist_ok=True)

# Define the range of rotation angles and the number of rotations
angles_range = range(-15, 16)  # From -15 to +15 degrees
num_rotations = 5  # Number of different random rotations per image

def rotate_image(image, angle):
    """
    Rotate image function
    :param image: Input image
    :param angle: Rotation angle
    :return: Rotated image
    """
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    # Calculate the rotation matrix
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    # Perform affine transformation with black padding (border constant filled with black)
    rotated = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))
    return rotated

# Traverse all subfolders and images in the input directory
for root, dirs, files in os.walk(input_path):
    for file_name in files:
        if file_name.lower().endswith(('.png', '.jpg', '.jpeg')):  # Check if it's an image file
            # Construct the file path
            file_path = os.path.join(root, file_name)
            
            # Read the image
            image = cv2.imread(file_path)
            if image is not None:
                # Randomly choose num_rotations different angles
                random_angles = np.random.choice(angles_range, num_rotations, replace=False)
                for angle in random_angles:
                    # Rotate the image
                    rotated_image = rotate_image(image, angle)

                    # Construct the relative path for saving
                    relative_path = os.path.relpath(root, input_path)
                    output_folder = os.path.join(output_path, relative_path)
                    os.makedirs(output_folder, exist_ok=True)

                    # Save the rotated image
                    output_file_name = f"{os.path.splitext(file_name)[0]}_rotated_{angle}.png"
                    output_file_path = os.path.join(output_folder, output_file_name)
                    cv2.imwrite(output_file_path, rotated_image)

print("All images have been rotated and saved!")


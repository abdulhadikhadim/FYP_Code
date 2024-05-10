from PIL import Image
import numpy as np
import os

def process_mask_images(input_folder, output_folder):
    # Get a list of all files in the input folder
    all_files = os.listdir(input_folder)

    # Filter out files that are not PNG images
    mask_files = [file for file in all_files if file.lower().endswith('.png')]

    saved_mask_paths = []
    for file in mask_files:
        # Construct the full file path
        file_path = os.path.join(input_folder, file)

        # Load the mask image
        mask = Image.open(file_path)

        # Convert the mask image to a numpy array
        mask_array = np.array(mask)

        # Create an RGB image with the same width and height as the mask, filled with black pixels
        color_mask_image = np.zeros((mask_array.shape[0], mask_array.shape[1], 3), dtype=np.uint8)

        # Assign red to one class and blue to another class
        # Assuming class 1 is red and class 2 is blue
        color_mask_image[mask_array == 1] = [255, 0, 0]  # Red for class 1
        color_mask_image[mask_array == 2] = [0, 0, 255]  # Blue for class 2

        # Convert the numpy array back to an image
        color_mask_image_pil = Image.fromarray(color_mask_image)

        # Ensure the output folder exists
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        # Save the colorized mask image with the same file name
        visible_mask_path = os.path.join(output_folder, file)
        color_mask_image_pil.save(visible_mask_path)
        saved_mask_paths.append(visible_mask_path)
        print(f"Color mask image {file} saved to {visible_mask_path}.")

    return saved_mask_paths

# Specify the folders
input_folder = r"E:\FYP\FInal-Dataset\mask"
output_folder = r"E:\FYP\FInal-Dataset\masks"

# Process the mask images and save the colorized versions
paths_of_saved_masks = process_mask_images(input_folder, output_folder)
print("Paths of saved color masks:", paths_of_saved_masks)

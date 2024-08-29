# import cv2
# import numpy as np
# import matplotlib.pyplot as plt
# from PIL import Image
# import os

# # Load the pre-trained HED model
# #hed_net = cv2.dnn.readNetFromCaffe('deploy.prototxt', 'hed_pretrained_bsds.caffemodel')

# # Create the path for the 'updatedDataset' directory
# updated_dataset_dir = os.path.join("", "updatedDataset")
# # Create the 'updatedDataset' directory if it doesn't exist
# os.makedirs(updated_dataset_dir, exist_ok=True)
# # Print the list of files and directories in the 'updatedDataset' directory
# print(os.listdir(updated_dataset_dir))

# # Path to the dataset folder
# dataset_dir = os.path.join("", "dataset")

# def get_jpg_image_names(folder_path, format=".jpg"):
#     """
#     Gets the names of files with the specified format in the given folder.

#     Args:
#     folder_path (str): The path to the folder where the files are located.
#     format (str): The file format to look for (default is ".jpg").

#     Returns:
#     list: A list of file names with the specified format.
#     """
#     # Check if the format is ".jpg"
#     if format == ".jpg":
#         # List comprehension to find all files ending with '.jpg' in the specified folder
#         jpg_images = [file_name for file_name in os.listdir(folder_path) if file_name.endswith('.jpg')]
#         # Return the list of jpg file names
#         return jpg_images
#     else:
#         return 


# def resize_with_aspect_ratio(image, target_size=(512, 512)):
#     """
#     Resize an image while preserving its aspect ratio.
    
#     Args:
#     image (numpy.ndarray): The input image to resize.
#     target_size (tuple): The desired size to resize the input image to.
    
#     Returns:
#     numpy.ndarray: The resized image with padding to fit the target size.
#     """
#     original_aspect_ratio = image.shape[1] / image.shape[0]
#     target_aspect_ratio = target_size[0] / target_size[1]
    
#     if original_aspect_ratio > target_aspect_ratio:
#         new_width = target_size[0]
#         new_height = int(new_width / original_aspect_ratio)
#     else:
#         new_height = target_size[1]
#         new_width = int(new_height * original_aspect_ratio)
    
#     resized_image = cv2.resize(image, (new_width, new_height))
    
#     # Create a new image with the target size and paste the resized image onto it
#     new_image = np.zeros((target_size[1], target_size[0], 3), dtype=np.uint8)
#     new_image[(target_size[1] - new_height) // 2:(target_size[1] - new_height) // 2 + new_height, 
#               (target_size[0] - new_width) // 2:(target_size[0] - new_width) // 2 + new_width] = resized_image
    
#     return new_image

# # Function to detect edges using HED (Holistically-Nested Edge Detection)
# def hed_edge_detection(image):
#     """
#     Detects edges in an image using the HED model.

#     Args:
#     image (numpy.ndarray): The input image in which edges are to be detected.

#     Returns:
#     numpy.ndarray: The output image with edges detected.
#     """
#     # Prepare the image for the HED model by creating a blob.
#     # Using the original size of the image
#     height, width = image.shape[:2]
#     blob = cv2.dnn.blobFromImage(image, scalefactor=1.0, size=(width, height), mean=(104.00698793, 116.66876762, 122.67891434), swapRB=False, crop=False)
    
#     # Set the input to the HED network.
#     hed_net.setInput(blob)
    
#     # Perform a forward pass to compute the edges.
#     hed_output = hed_net.forward()
    
#     # Extract the first (and only) channel from the output.
#     hed_output = hed_output[0, 0]
    
#     # Scale the output to the range [0, 255] and convert to uint8.
#     hed_output = (255 * hed_output).astype("uint8")
    
#     # Return the final edge-detected image.
#     return hed_output

# # Step 1: Read the image
# image_path = 'synt_1.jpg'
# image = cv2.imread(image_path)

# # Load the pre-trained HED model (ensure you have the HED model files)
# hed_net = cv2.dnn.readNetFromCaffe('deploy.prototxt', 'hed_pretrained_bsds.caffemodel')

# # Step 2: Detect edges using HED
# edges = hed_edge_detection(image)

# # Step 3: Save the edge-detected image to the 'updatedDataset' directory
# output_path = os.path.join(updated_dataset_dir, os.path.splitext(os.path.basename(image_path))[0] + '_edges.png')
# cv2.imwrite(output_path, edges)
# # Step 3: Save the edge-detected image to the same directory as the original image
# #output_path = os.path.splitext(image_path)[0] + '_edges.png'
# #cv2.imwrite(output_path, edges)

# # Step 4: Display the result
# plt.figure(figsize=(10, 10))
# plt.subplot(1, 2, 1)
# plt.title('Original Image')
# plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))  # Convert BGR to RGB for correct color display

# plt.subplot(1, 2, 2)
# plt.title('Edge Detected Image (HED)')
# plt.imshow(edges, cmap='gray')

# plt.show()

# #resize an image
# def resize_with_aspect_ratio(image, target_size=(512, 512)):
#     """
#     Resize an image while preserving its aspect ratio.
    
#     Args:
#     image (numpy.ndarray): The input image to resize.
#     target_size (tuple): The desired size to resize the input image to.
    
#     Returns:
#     numpy.ndarray: The resized image with padding to fit the target size.
#     """
#     original_aspect_ratio = image.shape[1] / image.shape[0]
#     target_aspect_ratio = target_size[0] / target_size[1]
    
#     if original_aspect_ratio > target_aspect_ratio:
#         new_width = target_size[0]
#         new_height = int(new_width / original_aspect_ratio)
#     else:
#         new_height = target_size[1]
#         new_width = int(new_height * original_aspect_ratio)
    
#     resized_image = cv2.resize(image, (new_width, new_height))
    
#     # Create a new image with the target size and paste the resized image onto it
#     new_image = np.zeros((target_size[1], target_size[0], 3), dtype=np.uint8)
#     new_image[(target_size[1] - new_height) // 2:(target_size[1] - new_height) // 2 + new_height, 
#               (target_size[0] - new_width) // 2:(target_size[0] - new_width) // 2 + new_width] = resized_image
    
#     return new_image

# image_path = 'synt_1_edges.png'
# image = cv2.imread(image_path)
# edges_resized = resize_with_aspect_ratio(image)

# # Step 5: Save the resized edge-detected image to the 'updatedDataset' directory
# output_path_resized = os.path.join(updated_dataset_dir, os.path.splitext(os.path.basename(image_path))[0] + '_resized.png')
# cv2.imwrite(output_path_resized, edges_resized)
# # Step 3: Save the edge-detected image to the same directory as the original image
# #output_path = os.path.splitext(image_path)[0] + '_resized.png'
# #cv2.imwrite(output_path, edges_resized)

import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os

# Create the path for the 'updatedDataset' directory
updated_dataset_dir = os.path.join("", "updatedDataset")
# Create the 'updatedDataset' directory if it doesn't exist
os.makedirs(updated_dataset_dir, exist_ok=True)
# Print the list of files and directories in the 'updatedDataset' directory
print(os.listdir(updated_dataset_dir))

# Path to the dataset folder
dataset_dir = os.path.join("", "dataset")

def get_jpg_image_names(folder_path, format=".jpg"):
    """
    Gets the names of files with the specified format in the given folder.

    Args:
    folder_path (str): The path to the folder where the files are located.
    format (str): The file format to look for (default is ".jpg").

    Returns:
    list: A list of file names with the specified format.
    """
    # Check if the format is ".jpg"
    if format == ".jpg":
        # List comprehension to find all files ending with '.jpg' in the specified folder
        jpg_images = [file_name for file_name in os.listdir(folder_path) if file_name.endswith('.jpg')]
        # Return the list of jpg file names
        return jpg_images
    else:
        return

def resize_with_aspect_ratio(image, target_size=(512, 512)):
    """
    Resize an image while preserving its aspect ratio.

    Args:
    image (numpy.ndarray): The input image to resize.
    target_size (tuple): The desired size to resize the input image to.

    Returns:
    numpy.ndarray: The resized image with padding to fit the target size.
    """
    original_aspect_ratio = image.shape[1] / image.shape[0]
    target_aspect_ratio = target_size[0] / target_size[1]

    if original_aspect_ratio > target_aspect_ratio:
        new_width = target_size[0]
        new_height = int(new_width / original_aspect_ratio)
    else:
        new_height = target_size[1]
        new_width = int(new_height * original_aspect_ratio)

    resized_image = cv2.resize(image, (new_width, new_height))

    # Create a new image with the target size and paste the resized image onto it
    if len(image.shape) == 3:  # If the image has 3 channels (RGB)
        new_image = np.zeros((target_size[1], target_size[0], 3), dtype=np.uint8)
    else:  # If the image is grayscale (single channel)
        new_image = np.zeros((target_size[1], target_size[0]), dtype=np.uint8)

    new_image[(target_size[1] - new_height) // 2:(target_size[1] - new_height) // 2 + new_height,
              (target_size[0] - new_width) // 2:(target_size[0] - new_width) // 2 + new_width] = resized_image

    return new_image

# Function to detect edges using HED (Holistically-Nested Edge Detection)
def hed_edge_detection(image):
    """
    Detects edges in an image using the HED model.

    Args:
    image (numpy.ndarray): The input image in which edges are to be detected.

    Returns:
    numpy.ndarray: The output image with edges detected.
    """
    # Prepare the image for the HED model by creating a blob.
    # Using the original size of the image
    height, width = image.shape[:2]
    blob = cv2.dnn.blobFromImage(image, scalefactor=1.0, size=(width, height), mean=(104.00698793, 116.66876762, 122.67891434), swapRB=False, crop=False)

    # Set the input to the HED network.
    hed_net.setInput(blob)

    # Perform a forward pass to compute the edges.
    hed_output = hed_net.forward()

    # Extract the first (and only) channel from the output.
    hed_output = hed_output[0, 0]

    # Scale the output to the range [0, 255] and convert to uint8.
    hed_output = (255 * hed_output).astype("uint8")

    # Return the final edge-detected image.
    return hed_output

# Load the pre-trained HED model (ensure you have the HED model files)
hed_net = cv2.dnn.readNetFromCaffe('deploy.prototxt', 'hed_pretrained_bsds.caffemodel')

# Get the list of JPG images in the dataset folder
jpg_images = get_jpg_image_names(dataset_dir)

# Process each image in the dataset folder
for image_name in jpg_images:
    # Step 1: Read the image
    image_path = os.path.join(dataset_dir, image_name)
    image = cv2.imread(image_path)

    # Step 2: Detect edges using HED
    edges = hed_edge_detection(image)

    # Step 3: Save the edge-detected image to the 'updatedDataset' directory
    output_path = os.path.join(updated_dataset_dir, os.path.splitext(image_name)[0] + '_edges.png')
    #cv2.imwrite(output_path, edges)

    # Step 4: Display the result (optional, for the first image only)
    if image_name == jpg_images[0]:
        plt.figure(figsize=(10, 10))
        plt.subplot(1, 2, 1)
        plt.title('Original Image')
        plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))  # Convert BGR to RGB for correct color display

        plt.subplot(1, 2, 2)
        plt.title('Edge Detected Image (HED)')
        plt.imshow(edges, cmap='gray')

        plt.show()

    # Resize the edge-detected image
    edges_resized = resize_with_aspect_ratio(edges)

    # Step 5: Save the resized edge-detected image to the 'updatedDataset' directory
    output_path_resized = os.path.join(updated_dataset_dir, os.path.splitext(image_name)[0] + '_resized.png')
    cv2.imwrite(output_path_resized, edges_resized)



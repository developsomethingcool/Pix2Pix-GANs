import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os

# Load the pre-trained HED model
hed_net = cv2.dnn.readNetFromCaffe('deploy.prototxt', 'hed_pretrained_bsds.caffemodel')


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
    new_image = np.zeros((target_size[1], target_size[0], 3), dtype=np.uint8)
    new_image[(target_size[1] - new_height) // 2:(target_size[1] - new_height) // 2 + new_height, 
              (target_size[0] - new_width) // 2:(target_size[0] - new_width) // 2 + new_width] = resized_image
    
    return new_image

# Function to detect edges using HED (Holistically-Nested Edge Detection)
def hed_edge_detection(image, target_size=(512, 512)):
    """
    Detects edges in an image using the HED model.

    Args:
    image (numpy.ndarray): The input image in which edges are to be detected.
    target_size (tuple): The desired size to resize the input image.

    Returns:
    numpy.ndarray: The output image with edges detected.
    """
    # Resize the image to the target size while preserving aspect ratio
    #resized_image = resize_with_aspect_ratio(image, target_size)

    # Prepare the image for the HED model by creating a blob.
    blob = cv2.dnn.blobFromImage(image, scalefactor=1.0, size=target_size, mean=(104.00698793, 116.66876762, 122.67891434), swapRB=False, crop=False)
    
    # Set the input to the HED network.
    hed_net.setInput(blob)
    
    # Perform a forward pass to compute the edges.
    hed_output = hed_net.forward()
    
    # Extract the first (and only) channel from the output.
    hed_output = hed_output[0, 0]
    
    # Resize the output to match the resized image size.
    #hed_output = cv2.resize(hed_output, target_size)
    
    # Scale the output to the range [0, 255] and convert to uint8.
    hed_output = (255 * hed_output).astype("uint8")
    
    # Return the final edge-detected image.
    return hed_output

# Step 1: Read the image
image = cv2.imread('cob_1.jpg')

# Load the pre-trained HED model (ensure you have the HED model files)
hed_net = cv2.dnn.readNetFromCaffe('deploy.prototxt', 'hed_pretrained_bsds.caffemodel')

# Step 2: Detect edges using HED
edges = hed_edge_detection(image)

# Step 3: Save the edge-detected image to the same directory as the original image
output_path = os.path.splitext('cob_1')[0] + '_edges.png'
cv2.imwrite(output_path, edges)

# Step 3: Display the result
plt.figure(figsize=(10, 10))
plt.subplot(1, 2, 1)
plt.title('Original Image')
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))  # Convert BGR to RGB for correct color display

plt.subplot(1, 2, 2)
plt.title('Edge Detected Image (HED)')
plt.imshow(edges, cmap='gray')

plt.show()
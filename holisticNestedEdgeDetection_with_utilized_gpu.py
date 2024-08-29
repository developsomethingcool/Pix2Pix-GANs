import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os
import torch
import torchvision.transforms as transforms
import time

# Load the pre-trained HED model using PyTorch
def load_hed_model():
    # Assuming you have a PyTorch model for HED
    model = torch.load('hed_model.pth')
    model.eval()  # Set the model to evaluation mode
    return model

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
def hed_edge_detection(image, model, device):
    """
    Detects edges in an image using the HED model.

    Args:
    image (numpy.ndarray): The input image in which edges are to be detected.
    model (torch.nn.Module): The pre-trained HED model.
    device (torch.device): The device to use for computation (CPU or GPU).

    Returns:
    numpy.ndarray: The output image with edges detected.
    """
    # Prepare the image for the HED model by creating a blob.
    # Using the original size of the image
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    image_tensor = transform(Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)))
    image_tensor = image_tensor.unsqueeze(0)  # Add batch dimension

    # Move the model and input to the specified device
    model.to(device)
    image_tensor = image_tensor.to(device)

    # Perform a forward pass to compute the edges.
    with torch.no_grad():
        hed_output = model(image_tensor)

    # Extract the first (and only) channel from the output.
    hed_output = hed_output[0, 0].cpu().numpy()

    # Scale the output to the range [0, 255] and convert to uint8.
    hed_output = (255 * hed_output).astype("uint8")

    # Return the final edge-detected image.
    return hed_output

# Step 1: Read the image
image_path = 'synt_1.jpg'
image = cv2.imread(image_path)

# Load the pre-trained HED model (ensure you have the HED model file)
hed_model = load_hed_model()

# Step 2: Detect edges using HED on CPU
device_cpu = torch.device("cpu")
start_time_cpu = time.time()
edges_cpu = hed_edge_detection(image, hed_model, device_cpu)
end_time_cpu = time.time()
cpu_time = end_time_cpu - start_time_cpu
print(f"CPU Time: {cpu_time:.4f} seconds")

# Step 3: Detect edges using HED on GPU
device_gpu = torch.device("cuda" if torch.cuda.is_available() else "cpu")
start_time_gpu = time.time()
edges_gpu = hed_edge_detection(image, hed_model, device_gpu)
end_time_gpu = time.time()
gpu_time = end_time_gpu - start_time_gpu
print(f"GPU Time: {gpu_time:.4f} seconds")

# Step 4: Save the edge-detected image to the same directory as the original image
output_path_cpu = os.path.splitext(image_path)[0] + '_edges_cpu.png'
cv2.imwrite(output_path_cpu, edges_cpu)

output_path_gpu = os.path.splitext(image_path)[0] + '_edges_gpu.png'
cv2.imwrite(output_path_gpu, edges_gpu)

# Step 5: Display the result
plt.figure(figsize=(10, 10))
plt.subplot(1, 3, 1)
plt.title('Original Image')
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))  # Convert BGR to RGB for correct color display

plt.subplot(1, 3, 2)
plt.title('Edge Detected Image (CPU)')
plt.imshow(edges_cpu, cmap='gray')

plt.subplot(1, 3, 3)
plt.title('Edge Detected Image (GPU)')
plt.imshow(edges_gpu, cmap='gray')

plt.show()

# Resize an image
image_path = 'synt_1_edges.png'
image = cv2.imread(image_path)
edges_resized = resize_with_aspect_ratio(image)

# Step 6: Save the edge-detected image to the same directory as the original image
output_path = os.path.splitext(image_path)[0] + '_resized.png'
cv2.imwrite(output_path, edges_resized)

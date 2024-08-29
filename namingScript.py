import os
from PIL import Image
import shutil


def get_folder_names(directory):
    """
    Gets the names of all folders in the specified directory.

    Args:
    directory (str): The path to the directory.

    Returns:
    list: A list of folder names in the specified directory.
    """
    try:
        # Get the list of all entries in the directory
        entries = os.listdir(directory)
        # Filter out only directories by checking if each entry is a directory
        folders = [entry for entry in entries if os.path.isdir(os.path.join(directory, entry))]
        return folders
    except Exception as e:
        # Print an error message if an exception occurs and return an empty list
        print(f"An error occurred: {e}")
        return []

def get_image_files(directory):
    """
    Gets the names of all image files (with .jpg, .jpeg, .png extensions) in the specified directory.

    Args:
    directory (str): The path to the directory.

    Returns:
    list: A list of image file names in the specified directory.
    """
    # List to hold the image file names
    image_files = []
    
    try:
        # Iterate over all the entries in the directory
        for entry in os.listdir(directory):
            # Get the full path of the entry
            full_path = os.path.join(directory, entry)
            # Check if the entry is a file and has one of the desired image extensions
            if os.path.isfile(full_path) and entry.lower().endswith(('.jpg', '.jpeg', '.png')):
                # Add the entry to the list of image files
                image_files.append(entry)
    except Exception as e:
        # Print an error message if an exception occurs
        print(f"An error occurred: {e}")
    
    # Return the list of image files
    return image_files

def rename_image(image_path, current_name, new_name):
    """
    Renames an image file from the current name to the new name.

    Args:
    image_path (str): The path to the directory containing the image.
    current_name (str): The current name of the image file.
    new_name (str): The new name for the image file.

    Returns:
    None
    """
    try:
        # Construct the full path to the current image
        current_path = os.path.join(image_path, current_name)
        # Construct the full path to the new image name
        new_path = os.path.join(image_path, new_name)
        
        # Rename the image file
        os.rename(current_path, new_path)
        print(f"Renamed '{current_name}' to '{new_name}' successfully.")
    except Exception as e:
        # Print an error message if an exception occurs
        print(f"An error occurred while renaming the image: {e}")  

def copy_image_to_resized_folder(src_folder, dest_folder, image_name, new_image_name):
    try:
        if not os.path.exists(dest_folder):
            os.makedirs(dest_folder)
        src_path = os.path.join(src_folder, image_name)
        dest_path = os.path.join(dest_folder, new_image_name)

        # Open the image
        img = Image.open(src_path)

        # Resize the image
        new_width = 256
        new_height = 256
        resized_img = img.resize((new_width, new_height))
        print("Resized", src_path)
        # Save the resized image
        resized_img.save(dest_path)

        #shutil.copy2(src_path, dest_path)
        print(f"Copied '{src_path}' to '{dest_path}' successfully.")

    except Exception as e:
        print(f"An error occurred while copying the image: {e}")


# Rename the images
directory_path = "COB"
resized_folder_path = "COB_resized"

folder_names = get_folder_names(directory_path)
folder_name_paths = [os.path.join(directory_path, folder_name) for folder_name in folder_names]

#Naming and resizing the image
image_counter = 1
for folder in folder_name_paths:
    images = get_image_files(folder)
    for image in images:
        new_image_name = f"cob_{image_counter}.jpg"
        
        #rename_image(folder, image, new_image_name)
        #copy_image_to_resized_folder(folder, resized_folder_path, new_image_name, new_image_name)
        image_counter += 1

print(f"Renamed {image_counter - 1} images.")

#directory_path = "COB"
#folder_names = get_folder_names(directory_path)
#folder_name_paths = [os.path.join(directory_path, folder_name) for folder_name in folder_names]
#print(folder_name_paths)

copy_image_to_resized_folder(directory_path, "cob_1.jpg", image_name, new_image_name)
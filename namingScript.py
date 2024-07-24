import os

def get_folder_names(directory):
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

# Example usage
directory_path = "COB"
folder_names = get_folder_names(directory_path)
folder_name_paths = [os.path.join(directory_path, folder_name) for folder_name in folder_names]

image_counter = 1
for folder in folder_name_paths:
    images = get_image_files(folder)
    for image in images:
        new_image_name = f"image_{image_counter}.jpg"
        rename_image(folder, image, new_image_name)
        image_counter += 1

print(f"Renamed {image_counter - 1} images.")
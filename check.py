import os

def rename_images(folder_path, new_format="AS1A02_005T01_9000000948lxp1BB_level{:03d}.fits"):
    """
    Rename all images in the specified folder to a specific format.
    
    Args:
        folder_path (str): Path to the folder containing the images.
        new_format (str): Format for the new file names. Use {:03d} for numbering.
                          Example: "image_{:03d}.jpg" -> image_001.jpg, image_002.jpg
    """
    # Get a list of all files in the folder
    files = os.listdir(folder_path)
    
    # Filter for image files (you can adjust this based on your file types)
    image_files = [file for file in files if file.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp'))]
    
    # Sort the files for consistent numbering (optional)
    image_files.sort()
    
    # Loop through the files and rename them
    for index, file in enumerate(image_files, start=1):
        # Create the new file name
        new_name = new_format.format(index)
        print(new_name)
        
        # Construct full file paths
        old_path = os.path.join(folder_path, file)
        new_path = os.path.join(folder_path, new_name)
        
        # Rename the file
        os.rename(old_path, new_path)
        print(f"Renamed: {file} -> {new_name}")

# Example Usage
folder_path = "/Users/meerthikasr/Desktop/images"  # Replace with the path to your folder
rename_images(folder_path, new_format="AS1A02_005T01_9000000948lxp1BB_level{:03d}.fits")

import tarfile
import os
import shutil

# Paths
tar_dir = "/Users/federicogirolami/Downloads/CXR8/images"
target_dir = "./ChestX-ray14/images"

# Ensure the target directory exists
os.makedirs(target_dir, exist_ok=True)

# List of tar files
tar_files = [
    "images_001.tar.gz", "images_002.tar.gz", "images_003.tar.gz",
    "images_004.tar.gz", "images_005.tar.gz", "images_006.tar.gz",
    "images_007.tar.gz", "images_008.tar.gz", "images_009.tar.gz",
    "images_010.tar.gz", "images_011.tar.gz", "images_012.tar.gz"
]

# Flags for replace options
replace_all = False

def ask_replace(file_path):
    global replace_all
    if replace_all:
        return True

    response = input(f"File '{file_path}' already exists. Replace? (y = yes, n = no, a = replace all): ").strip().lower()
    if response == 'y':
        return True
    elif response == 'a':
        replace_all = True
        return True
    else:
        return False

# Process each tar file
for tar_name in tar_files:
    tar_path = os.path.join(tar_dir, tar_name)

    # Extract the tar file
    with tarfile.open(tar_path, 'r:gz') as tar:
        # Get the top-level directory inside the tar file
        base_folder = tar.getmembers()[0].name.split('/')[0]

        # Extract to a temporary location
        tar.extractall(path=tar_dir)

        # Move images to the target directory
        extracted_folder = os.path.join(tar_dir, base_folder)
        for root, _, files in os.walk(extracted_folder):
            for file in files:
                if file.endswith(('.png', '.jpg', '.jpeg')):  # Adjust extensions if necessary
                    src_path = os.path.join(root, file)
                    dest_path = os.path.join(target_dir, file)

                    # Check if file already exists
                    if os.path.exists(dest_path):
                        if not ask_replace(dest_path):
                            continue

                    shutil.move(src_path, dest_path)

        # Clean up the extracted folder
        shutil.rmtree(extracted_folder)

print("Extraction and moving of images completed.")

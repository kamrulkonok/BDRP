import os
import shutil

# Source and target directory paths
source_dir = "/gpfs/workdir/islamm/datasets_without_NoFindings"
target_dir = "/gpfs/workdir/islamm/rotated_datasets_without_NoFindings"

# Ensure the target directory exists
if not os.path.exists(target_dir):
    os.makedirs(target_dir)

# Iterate over the files in the source directory
for filename in os.listdir(source_dir):
    source_path = os.path.join(source_dir, filename)
    target_path = os.path.join(target_dir, filename)

    # Check if the file is an image (based on file extension)
    if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif', '.tiff')):
        try:
            # Copy the file to the target directory
            shutil.copy2(source_path, target_path)
            print("Copied: %s -> %s" % (filename, target_dir))

            # Remove the file from the source directory
            os.remove(source_path)
            print("Deleted: %s from %s" % (filename, source_dir))

        except Exception as e:
            print("Error processing %s: %s" % (filename, e))


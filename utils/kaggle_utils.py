import os
import subprocess
import zipfile


def download_tiny_imagenet(root_dir):
    """
    Downloads the Tiny ImageNet dataset from Kaggle if not already present.
    Assumes Kaggle API is installed and configured.
    """
    dataset_folder = os.path.join(root_dir, 'tiny-imagenet-200')
    if os.path.exists(dataset_folder):
        print("Tiny ImageNet dataset already exists at", dataset_folder)
        return

    os.makedirs(root_dir, exist_ok=True)
    print("Downloading Tiny ImageNet dataset from Kaggle...")

    # This command assumes the dataset is available at 'tylerx/tiny-imagenet'
    command = f'kaggle datasets download -d akash2sharma/tiny-imagenet -p {root_dir}'
    subprocess.run(command, shell=True, check=True)

    # Extract the downloaded zip file
    zip_path = os.path.join(root_dir, 'tiny-imagenet.zip')
    if os.path.exists(zip_path):
        print("Extracting dataset...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(root_dir)
        os.remove(zip_path)
        print("Download and extraction complete!")
    else:
        print("Download might have failed; zip file not found.")


# Example usage:
if __name__ == "__main__":
    download_tiny_imagenet(root_dir='data')
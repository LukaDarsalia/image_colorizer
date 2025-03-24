import os

import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import Dataset, DataLoader


class TinyImageNetDataset(Dataset):
    """
    Dataset class for Tiny ImageNet dataset, supporting both colorization
    (grayscale to color) and standard RGB modes.

    For GANs, we can use this dataset in multiple ways:
    1. Grayscale to color mode: Generator learns to colorize grayscale images
    2. Standard RGB mode: For traditional GANs where we generate images from noise
    """

    def __init__(self, root_dir, split='train', transform=None, mode='colorization'):
        """
        Initialize the Tiny ImageNet dataset.

        Args:
            root_dir (str): Root directory of the tiny-imagenet dataset.
            split (str): 'train' or 'val' to select the data split.
            transform (callable, optional): Optional transform to be applied on an image.
            mode (str): 'colorization' for grayscale to color, 'rgb' for standard RGB mode.
        """
        self.split = split
        self.transform = transform
        self.mode = mode

        if self.mode not in ['colorization', 'rgb']:
            raise ValueError("mode must be either 'colorization' or 'rgb'")

        # Construct path for the given split
        self.split_dir = os.path.join(root_dir, split)
        self.image_paths = []
        self.labels = []
        self.class_to_idx = {}  # Mapping from class names to indices

        # Check if the dataset directory exists
        if not os.path.exists(self.split_dir):
            raise FileNotFoundError(f"Dataset directory not found: {self.split_dir}. "
                                    "Please download the dataset first.")

        if split == 'train':
            # Each class in training is a folder under train/<class>/images/
            for idx, class_dir in enumerate(sorted(os.listdir(self.split_dir))):
                class_path = os.path.join(self.split_dir, class_dir, 'images')
                if os.path.isdir(class_path):
                    self.class_to_idx[class_dir] = idx
                    for file in os.listdir(class_path):
                        if file.lower().endswith(('.jpeg', '.jpg', '.png')):
                            self.image_paths.append(os.path.join(class_path, file))
                            self.labels.append(idx)
        elif split == 'val':
            # Build the class to idx mapping first from the train set
            train_dir = os.path.join(os.path.dirname(self.split_dir), 'train')
            if os.path.exists(train_dir):
                for idx, class_dir in enumerate(sorted(os.listdir(train_dir))):
                    if os.path.isdir(os.path.join(train_dir, class_dir)):
                        self.class_to_idx[class_dir] = idx

            # In validation, images are stored in a single folder and annotations in a txt file.
            val_images_dir = os.path.join(self.split_dir, 'images')
            annotation_file = os.path.join(self.split_dir, 'val_annotations.txt')

            # Build a mapping from image name to class label.
            mapping = {}
            with open(annotation_file, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 2:
                        mapping[parts[0]] = parts[1]

            for file in os.listdir(val_images_dir):
                if file.lower().endswith(('.jpeg', '.jpg', '.png')):
                    class_name = mapping.get(file, 'unknown')
                    class_idx = self.class_to_idx.get(class_name, -1)
                    self.image_paths.append(os.path.join(val_images_dir, file))
                    self.labels.append(class_idx)
        else:
            raise ValueError("split must be 'train' or 'val'")

    def __len__(self):
        """Return the total number of images."""
        return len(self.image_paths)

    def __getitem__(self, idx):
        """
        Get the image and target for the given index.

        For colorization mode: input is grayscale, target is RGB
        For RGB mode: input and target are both RGB
        """
        # Load image and convert to RGB.
        img_path = self.image_paths[idx]
        try:
            image = Image.open(img_path).convert('RGB')
        except (IOError, OSError) as e:
            print(f"Error loading image {img_path}: {e}")
            # Return a blank image as fallback
            image = Image.new('RGB', (64, 64), color='gray')

        # Apply transforms if provided
        if self.transform:
            image = self.transform(image)
        else:
            # Basic transform to tensor if none provided.
            image = transforms.ToTensor()(image)

        if self.mode == 'colorization':
            # Create grayscale input from the color image.
            grayscale = transforms.functional.rgb_to_grayscale(image, num_output_channels=1)
            return grayscale, image
        else:  # rgb mode
            return image, self.labels[idx]


def get_dataloaders(root_dir, batch_size=64, image_size=64, mode='colorization'):
    """
    Creates and returns training and validation dataloaders.

    Args:
        root_dir (str): Root directory of the tiny-imagenet dataset.
        batch_size (int): Batch size for the dataloader.
        image_size (int): Size to resize images to.
        mode (str): 'colorization' or 'rgb' mode.

    Returns:
        tuple: (train_loader, val_loader)
    """
    # Define transforms
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    # Create datasets
    train_dataset = TinyImageNetDataset(
        root_dir=root_dir,
        split='train',
        transform=transform,
        mode=mode
    )

    val_dataset = TinyImageNetDataset(
        root_dir=root_dir,
        split='val',
        transform=transform,
        mode=mode
    )

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    return train_loader, val_loader


# Example usage:
if __name__ == "__main__":
    # Define transforms and test the dataset
    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    try:
        # Test colorization mode
        dataset = TinyImageNetDataset(
            root_dir='data/tiny-imagenet-200',
            split='train',
            transform=transform,
            mode='colorization'
        )
        print("Dataset length:", len(dataset))

        # Test a sample
        sample_input, sample_target = dataset[0]
        print("Input shape (grayscale):", sample_input.shape)
        print("Target shape (color):", sample_target.shape)

        # Test dataloaders
        train_loader, val_loader = get_dataloaders('data/tiny-imagenet-200')
        print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")

    except Exception as e:
        print(f"Error testing dataset: {e}")
        print("Make sure the dataset is downloaded first using kaggle_utils.py")
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

    def __init__(self, root_dir: str, split: str = 'train', transform: callable = None):
        """
        Initialize the Tiny ImageNet dataset.

        Args:
            root_dir (str): Root directory of the tiny-imagenet dataset.
            split (str): 'train' or 'val' to select the data split.
            transform (callable, optional): Optional transform to be applied on an image.
        """
        self.split = split
        self.transform = transform

        # Construct path for the given split
        self.split_dir = os.path.join(root_dir, split)
        self.image_paths = []

        # Check if the dataset directory exists
        if not os.path.exists(self.split_dir):
            raise FileNotFoundError(f"Dataset directory not found: {self.split_dir}. "
                                    "Please download the dataset first.")

        if split == 'train':
            for class_dir in sorted(os.listdir(self.split_dir)):
                class_path = os.path.join(self.split_dir, class_dir, 'images')
                if os.path.isdir(class_path):
                    for file in os.listdir(class_path):
                        if file.lower().endswith(('.jpeg', '.jpg', '.png')):
                            self.image_paths.append(os.path.join(class_path, file))
        elif split == 'val':
            val_images_dir = os.path.join(self.split_dir, 'images')
            for file in os.listdir(val_images_dir):
                if file.lower().endswith(('.jpeg', '.jpg', '.png')):
                    self.image_paths.append(os.path.join(val_images_dir, file))
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

        # Create grayscale input from the color image.
        grayscale = transforms.functional.rgb_to_grayscale(image, num_output_channels=1)
        return {'image': grayscale, 'target': image}


def get_dataloaders(root_dir, batch_size=64, image_size=64):
    """
    Creates and returns training and validation dataloaders.

    Args:
        root_dir (str): Root directory of the tiny-imagenet dataset.
        batch_size (int): Batch size for the dataloader.
        image_size (int): Size to resize images to.

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
    )

    val_dataset = TinyImageNetDataset(
        root_dir=root_dir,
        split='val',
        transform=transform,
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

    # Test colorization mode
    dataset = TinyImageNetDataset(
        root_dir='data/tiny-imagenet-200',
        split='train',
        transform=transform,
    )
    print("Dataset length:", len(dataset))

    # Test a sample
    sample = dataset[0]
    print("Input shape (grayscale):", sample['image'].shape)
    print("Target shape (color):", sample['target'].shape)

    # Test dataloaders
    train_loader, val_loader = get_dataloaders('data/tiny-imagenet-200')
    print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")

    for batch_idx, batch in enumerate(train_loader):
        print(f"Batch {batch_idx} shape: {batch['image'].shape}, {batch['target'].shape}")
        break
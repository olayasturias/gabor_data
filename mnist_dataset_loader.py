import torch
from torch.utils.data import Dataset
from torchvision import transforms
import pandas as pd
from PIL import Image
import os
import pytorch_lightning as pl
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt


class MNISTDataset(Dataset):
    def __init__(self, csv_file, image_dir, transform=None):
        """
        Args:
            csv_file (str): Path to the CSV file with image names and labels.
            image_dir (str): Directory with all images.
            transform (callable, optional): Transformations applied to images.
        """
        self.data = pd.read_csv(csv_file)
        self.image_dir = image_dir
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Load image
        img_name = os.path.join(self.image_dir, self.data.iloc[idx, 0])
        image = Image.open(img_name)
        image = transforms.ToTensor()(image)  # Convert to PyTorch tensor
        image = transforms.Normalize((0.5,), (0.5,))(image)  # Normalize image

        # Load labels (orientations, shifts, and mnist labels)
        labels = self.data.iloc[idx, 1:].astype('float32').values
        labels = torch.tensor(labels, dtype=torch.float32)
        theta_rad = labels[0] * (torch.pi / 180.0)
        cos_sin = torch.tensor([torch.cos(theta_rad), torch.sin(theta_rad)], dtype=torch.float32)
        shifts = labels[1:3].clone().detach().float()
        digit = labels[3].clone().detach().float().unsqueeze(0)
        labels = torch.cat((cos_sin, shifts, digit), dim=0)

        # Apply transformations
        if self.transform:
            image = self.transform(image)

        return image, labels


class MNISTDataModule(pl.LightningDataModule):
    def __init__(self,
                 dataset_path,
                 train_val_set,
                 test_set,
                 train_ratio=0.8,
                 batch_size=32
                 ):
        super().__init__()
        self.dataset_path = dataset_path
        self.train_val_set = train_val_set
        self.test_set = test_set
        self.batch_size = batch_size
        self.train_ratio = train_ratio

    def setup(self, stage=None):
        """ Split dataset into train/val/test. """
        train_val_csv_file = os.path.join(self.dataset_path,
                                          self.train_val_set, "description.csv")
        train_val_img_dir = os.path.join(self.dataset_path, self.train_val_set)
        print(f"Using data from {train_val_csv_file} for training and validation.")
        self.train_dataset = MNISTDataset(train_val_csv_file, train_val_img_dir)
        train_size = self.train_ratio
        val_size = 1 - train_size
        self.train_dataset, self.val_dataset = random_split(self.train_dataset,
                                                            [train_size,
                                                             val_size])
        
        # Load CSV to analyze balance
        df = pd.read_csv(train_val_csv_file)

        # Check distribution of digit labels
        digit_counts = df['label'].value_counts().sort_index()

        # Check distribution of rotation angles
        rotation_counts = df['orientation_0'].value_counts().sort_index()

        # Check distribution of shifts
        shift_x_counts = df['shift_x_0'].value_counts().sort_index()
        shift_y_counts = df['shift_y_0'].value_counts().sort_index()

        # Plot distributions
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))

        axes[0, 0].bar(digit_counts.index, digit_counts.values)
        axes[0, 0].set_title("Digit Label Distribution")
        axes[0, 0].set_xlabel("Digit")
        axes[0, 0].set_ylabel("Count")

        axes[0, 1].bar(rotation_counts.index, rotation_counts.values)
        axes[0, 1].set_title("Rotation Angle Distribution")
        axes[0, 1].set_xlabel("Rotation Angle (Degrees)")
        axes[0, 1].set_ylabel("Count")

        axes[1, 0].bar(shift_x_counts.index, shift_x_counts.values)
        axes[1, 0].set_title("Shift X Distribution")
        axes[1, 0].set_xlabel("Shift X (Pixels)")
        axes[1, 0].set_ylabel("Count")

        axes[1, 1].bar(shift_y_counts.index, shift_y_counts.values)
        axes[1, 1].set_title("Shift Y Distribution")
        axes[1, 1].set_xlabel("Shift Y (Pixels)")
        axes[1, 1].set_ylabel("Count")

        plt.tight_layout()
        plt.show()

        test_csv_file = os.path.join(self.dataset_path,
                                     self.test_set, "description.csv")
        test_img_dir = os.path.join(self.dataset_path, self.test_set)
        print(f"Using data from {test_csv_file} for testing")
        self.test_dataset = MNISTDataset(test_csv_file, test_img_dir)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=1, shuffle=False)


# Initialize DataModule
data_module = MNISTDataModule(
    dataset_path="C:\\Users\\oat\\Datasets\\mnist_data",
    train_val_set="C8_Z2_5",
    test_set="C8_Z2_5",
    train_ratio=0.8,
    batch_size=1
)

# Load data
data_module.setup()

# Get one batch
train_loader = data_module.train_dataloader()
images, labels = next(iter(train_loader))

# Check shapes
print(f"Image batch shape: {images.shape}")  # Expected: (batch_size, 1, 128, 128)
print(f"Label batch shape: {labels.shape}")  # Expected: (batch_size, n_labels)

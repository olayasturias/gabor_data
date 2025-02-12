import torch
from torch.utils.data import Dataset
from torchvision import transforms
import pandas as pd
from PIL import Image
import os
import pytorch_lightning as pl
from torch.utils.data import DataLoader, random_split

class GaborDataset(Dataset):
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

        # Load labels (orientations and shifts)
        labels = self.data.iloc[idx, 1:].astype('float32').values
        labels[0] = torch.tensor(labels[0] * (torch.pi / 180.0), dtype=torch.float32)  # Convert degrees to radians
        labels = torch.tensor(labels, dtype=torch.float32)

        # Apply transformations
        if self.transform:
            image = self.transform(image)

        return image, labels


class GaborDataModule(pl.LightningDataModule):
    def __init__(self, csv_file, image_dir, batch_size=32, train_ratio=0.7):
        super().__init__()
        self.csv_file = csv_file
        self.image_dir = image_dir
        self.batch_size = batch_size
        self.train_ratio = train_ratio

    def setup(self, stage=None):
        """ Split dataset into train/val/test. """
        dataset = GaborDataset(self.csv_file, self.image_dir)
        train_size = self.train_ratio
        test_size = 0.1
        val_size = 1 - train_size - test_size
        self.train_dataset, self.val_dataset, \
            self.test_dataset = random_split(dataset,
                                             [train_size, val_size, test_size])

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=1, shuffle=False)


# Initialize DataModule
data_module = GaborDataModule(
    csv_file="C:\\Users\\oat\\Datasets\\gabor_data\\C8_Z2_5\\description.csv",
    image_dir="C:\\Users\\oat\\Datasets\\gabor_data\\C8_Z2_5",
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

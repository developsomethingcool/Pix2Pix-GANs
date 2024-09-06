import os
import torch
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms

class EdgeToRealDataset(Dataset):
    def __init__(self, edge_dir, real_image_dir, transform=None):
        self.edge_dir = edge_dir
        self.real_image_dir = real_image_dir
        self.transform = transform

        self.edge_images = sorted(os.listdir(self.edge_dir))
        self.real_images = sorted(os.listdir(self.real_image_dir))

        assert len(self.edge_images) == len(self.real_images), "Number of edge images should be the same as number of real images"

    def __len__(self):
        return len(self.real_images)

    def __getitem__(self, idx):
        edge_image_path = os.path.join(self.edge_dir, self.edge_images[idx])
        real_image_path = os.path.join(self.real_image_dir, self.real_images[idx])

        edge_image = Image.open(edge_image_path).convert("RGB")
        real_image = Image.open(real_image_path).convert("RGB")

        if self.transform:
            edge_image = self.transform(edge_image)
            real_image = self.transform(real_image)

        return edge_image, real_image

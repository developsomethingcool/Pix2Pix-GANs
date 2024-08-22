import os
import torch
from torch.utils.data import Dataset
from PIL import Image

class CustomImageDataset(Dataset):
    def __init__(self, image_dir, transform=None):
        self.image_dir = image_dir
        self.image_filenames = [f for f in os.listdir(image_dir) if f.endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif'))]
        self.transform = transform

    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, idx):
        img_name = os.path.join(self.image_dir, self.image_filenames[idx])
        image = Image.open(img_name)

        if self.transform:
            image = self.transform(image)

        return image

# Example usage (if running this file standalone)
if __name__ == "__main__":
    dataset = CustomImageDataset(image_dir='path/to/images')
    print(len(dataset))
    print(dataset[0])

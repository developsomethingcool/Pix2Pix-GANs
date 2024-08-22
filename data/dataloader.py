import torch
from torch.utils.data import DataLoader
from .dataset import CustomImageDataset
from torchvision import transforms

def get_dataloader(image_dir, batch_size=32, shuffle=True, num_workers=4):
    transform = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    dataset = CustomImageDataset(image_dir=image_dir, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)

    return dataloader

# Example usage (if running this file standalone)
if __name__ == "__main__":
    dataloader = get_dataloader(image_dir='path/to/images', batch_size=16)
    for images in dataloader:
        print(images.size())

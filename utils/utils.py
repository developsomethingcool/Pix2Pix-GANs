import torch
import os
from torchvision.utils import save_image

def save_checkpoint(state, filename='checkpoint.pth.tar'):
    print(f"Saving checkpoint to {filename}")
    torch.save(state, filename)

def load_checkpoint(checkpoint_path, model, model_key, optimizer=None, optimizer_key=None,  scheduler=None, scheduler_key=None, device='cpu'):
    print(f"Loading checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint[model_key])
    print(f"Checkpoint for {model_key} loaded successfully")

    # Load optimizer if provided
    if optimizer and optimizer_key:
        optimizer.load_state_dict(checkpoint[optimizer_key])
        print(f"Optimizer state for {optimizer_key} loaded successfully")

    # Load scheduler state if provided
    if scheduler and scheduler_key and scheduler_key in checkpoint:
        scheduler.load_state_dict(checkpoint[scheduler_key])
        print(f"Scheduler state for {scheduler_key} loaded successfully")

def generate_images(generator, dataloader, device, save_path='generated_images', num_images_to_save=64):
    """
    Function to generate and save images using a trained generator.
    """
    generator.eval()
    os.makedirs(save_path, exist_ok=True)  

    with torch.no_grad():
        for i, (edges, _) in enumerate(dataloader):
            edges = edges.to(device)
            fakes = generator(edges)

            if i < num_images_to_save // dataloader.batch_size:
                save_image(fakes, os.path.join(save_path, f'generated_{i}.png'), normalize=True)

    print(f"Images saved to {save_path}")



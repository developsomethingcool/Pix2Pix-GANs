import torch
import torchvision.utils as vutils

def evaluate_model(generator, val_loader, device='cuda', save_path='output'):
    generator.eval()  # Set the generator to evaluation mode

    fixed_noise = torch.randn(64, 3, 256, 256).to(device)

    with torch.no_grad():
        for i, (real_images, _) in enumerate(val_loader):
            fake_images = generator(fixed_noise).detach().cpu()

            # Save generated images for inspection
            vutils.save_image(fake_images, f"{save_path}/fake_images_epoch_{i}.png", normalize=True)

            # Optionally, compute evaluation metrics like FID, IS, etc.

    generator.train()  # Set the generator back to training mode

import torch
import torchvision.utils as vutils
import os

def evaluate_pix2pix(generator, dataloader, device='cuda', save_path='output', epoch=None, num_images_to_save=64):
    generator.eval()
    real_images = []
    generated_images = []

    with torch.no_grad():
        for edges, reals in dataloader:
            edges = edges.to(device)
            reals = reals.to(device)

            fakes = generator(edges)

            real_images.append(reals)
            generated_images.append(fakes)

    real_images = torch.cat(real_images, dim=0)
    generated_images = torch.cat(generated_images, dim=0)

    fid = compute_fid(real_images, generated_images)
    inception_score = compute_is(generated_images)

    print(f"FID: {fid}")
    print(f"Inception Score: {inception_score}")

    return fid, inception_score

    
    # generator.eval()  # Set the generator to evaluation mode

    # # Create save path directory if it doesn't exist
    # os.makedirs(save_path, exist_ok=True)

    # # Generate a fixed noise vector for consistency
    # fixed_noise = torch.randn(num_images_to_save, 3, 256, 256).to(device)

    # with torch.no_grad():
    #     for i, (real_images, _) in enumerate(val_loader):
    #         # Generate fake images
    #         fake_images = generator(fixed_noise).detach().cpu()

    #         # Save generated images for inspection
    #         if epoch is not None:
    #             file_name = f"{save_path}/fake_images_epoch_{epoch}_batch_{i}.png"
    #         else:
    #             file_name = f"{save_path}/fake_images_batch_{i}.png"

    #         vutils.save_image(fake_images, file_name, normalize=True)

    #         # Optionally, compute evaluation metrics like FID, IS, etc.
    #         # If you want to compute metrics, add them here
    #         break  # If you only want to save one batch of images per epoch, keep this break

    # generator.train()  # Set the generator back to training mode

import torch
import torch.optim as optim
from data import get_dataloaders
from models.generator import UNetGenerator
from models.discriminator import PatchGANDiscriminator
from training import train_pix2pix
from training.evaluator import evaluate_pix2pix
from utils.utils import load_checkpoint
import tarfile

#checkpoint_path = "C:/Users/opometun/Desktop/Thesis/pix2pix_checkpoint_epoch_100.pth/pix2pix_checkpoint_epoch_100.pth/data.pkl"


# checkpoint = torch.load('pix2pix_checkpoint_epoch_200.pth.tar')
# print(checkpoint.keys())  # This will show 'generator_state_dict', 'discriminator_state_dict', etc.

# # To inspect the keys of the discriminator's state_dict
# print(checkpoint['discriminator_state_dict'].keys())

# model = PatchGANDiscriminator()
# print(model.state_dict().keys())



def generate_images(generator, dataloader, device, save_path='generated_images', num_images_to_save=64):
    """
    Function to generate and save images using a trained generator.
    """
    generator.eval()
    os.makedirs(save_path, exist_ok=True)  # Create output directory if it doesn't exist

    with torch.no_grad():
        for i, (edges, _) in enumerate(dataloader):
            edges = edges.to(device)
            fakes = generator(edges)

            if i < num_images_to_save // dataloader.batch_size:
                vutils.save_image(fakes, os.path.join(save_path, f'generated_{i}.png'), normalize=True)

    print(f"Images saved to {save_path}")

def main():
    task = 'train'  # Options: 'train', 'eval', 'generate'
    edge_dir = 'edges'
    real_image_dir = 'real_images'
    #checkpoint_path = "pix2pix_checkpoint_epoch_295.pth.tar"
    checkpoint_path = None

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    num_epochs = 100
    batch_size = 16
    lr = 2e-4
    lambda_l1 = 100

    print(f"Task: {task}")
    print(f"Edge images directory: {edge_dir}")
    print(f"Real images directory: {real_image_dir}")
    print(f"Device: {device}")
    print(f"Epochs: {num_epochs}")
    print(f"Batch size: {batch_size}")
    print(f"Learning rate: {lr}")
    print(f"L1 weight: {lambda_l1}")

    # Create DataLoader
    train_loader, val_loader, test_loader = get_dataloaders(
        edge_dir, real_image_dir, batch_size=batch_size, num_workers=4
    )

    # Initialize models
    generator = UNetGenerator().to(device)
    discriminator = PatchGANDiscriminator().to(device) if task == 'train' else None

    # Initialize optimizers
    opt_gen = optim.Adam(generator.parameters(), lr=lr, betas=(0.5, 0.999))
    opt_disc = optim.Adam(discriminator.parameters(), lr=lr, betas=(0.5, 0.999)) if task == 'train' else None

    # Load checkpoint
    start_epoch = 0
    if checkpoint_path:
        print(f"Loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        load_checkpoint(checkpoint_path, generator, 'generator_state_dict', optimizer=opt_gen, optimizer_key='opt_gen_state_dict', device=device)
        if task == 'train':
            load_checkpoint(checkpoint_path, discriminator, 'discriminator_state_dict', optimizer=opt_disc, optimizer_key='opt_disc_state_dict', device=device)
        start_epoch = checkpoint['epoch'] + 1  # Continue from the next epoch

    # Perform task
    if task == 'train':
        print("Starting training...")
        train_pix2pix(generator, discriminator, train_loader, opt_gen, opt_disc, num_epochs=num_epochs, start_epoch=start_epoch, lr=lr, lambda_l1=lambda_l1, device=device)
    elif task == 'eval':
        print("Starting evaluation...")
        evaluate_pix2pix(generator, val_loader, device, save_path='evaluation_results', num_images_to_save=16)
    elif task == 'generate':
        print("Generating images...")
        generate_images(generator, test_loader, device, save_path='generated_images', num_images_to_save=64)

if __name__ == "__main__":
    main()

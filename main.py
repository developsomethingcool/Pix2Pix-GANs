import torch
from data import get_dataloader
from models.generator import Generator
from models.discriminator import Discriminator
from training import train_model
from training.evaluator import evaluate_model

def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Create DataLoader
    train_loader = get_dataloader(image_dir='path/to/train/images', batch_size=32)
    val_loader = get_dataloader(image_dir='path/to/val/images', batch_size=32)

    # Initialize models
    generator = Generator().to(device)
    discriminator = Discriminator().to(device)

    # Optionally load model weights from a specific checkpoint
    checkpoint_epoch = 10  # Example: Load the model from epoch 10
    try:
        generator.load_state_dict(torch.load(f'generator_epoch_{checkpoint_epoch}.pth'))
        discriminator.load_state_dict(torch.load(f'discriminator_epoch_{checkpoint_epoch}.pth'))
        print(f"Loaded models from epoch {checkpoint_epoch}.")
    except FileNotFoundError:
        print(f"No checkpoint found for epoch {checkpoint_epoch}. Training from scratch.")

    # Train the model
    num_epochs = 100
    for epoch in range(checkpoint_epoch, num_epochs):
        train_model(generator, discriminator, train_loader, num_epochs=1, device=device)

        # Evaluate the model after every epoch (or as needed)
        evaluate_model(generator, val_loader, device=device, save_path='output', epoch=epoch)

        # Save model weights after each epoch
        torch.save(generator.state_dict(), f'generator_epoch_{epoch+1}.pth')
        torch.save(discriminator.state_dict(), f'discriminator_epoch_{epoch+1}.pth')

if __name__ == "__main__":
    main()

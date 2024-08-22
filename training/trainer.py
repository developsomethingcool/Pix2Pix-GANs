import torch
import torch.nn as nn
import torch.optim as optim

def train_model(generator, discriminator, train_loader, num_epochs=100, device='cuda'):
    adversarial_loss = nn.BCELoss().to(device)
    optimizer_G = optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
    optimizer_D = optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))

    for epoch in range(num_epochs):
        for i, (real_images, _) in enumerate(train_loader):
            # Move real images to the device
            real_images = real_images.to(device)
            batch_size = real_images.size(0)
            
            # Labels for real and fake images
            valid = torch.ones(batch_size, 1, requires_grad=False).to(device)
            fake = torch.zeros(batch_size, 1, requires_grad=False).to(device)

            # ---------------------
            #  Train Discriminator
            # ---------------------
            optimizer_D.zero_grad()

            # Generate fake images
            noise = torch.randn(batch_size, 3, 256, 256).to(device)
            fake_images = generator(noise)

            # Discriminator loss for real and fake images
            real_loss = adversarial_loss(discriminator(real_images), valid)
            fake_loss = adversarial_loss(discriminator(fake_images.detach()), fake)
            d_loss = (real_loss + fake_loss) / 2

            # Backpropagation and optimization step for the discriminator
            d_loss.backward()
            optimizer_D.step()

            # -----------------
            #  Train Generator
            # -----------------
            optimizer_G.zero_grad()

            # Generator loss: try to fool the discriminator
            g_loss = adversarial_loss(discriminator(fake_images), valid)

            # Backpropagation and optimization step for the generator
            g_loss.backward()
            optimizer_G.step()

            # Print progress (optional, for debugging)
            if i % 100 == 0:
                print(f"[Epoch {epoch}/{num_epochs}] [Batch {i}/{len(train_loader)}] [D loss: {d_loss.item():.4f}] [G loss: {g_loss.item():.4f}]")

        # Optionally save model checkpoints or generated images after each epoch

    # Optionally return the trained models
    return generator, discriminator

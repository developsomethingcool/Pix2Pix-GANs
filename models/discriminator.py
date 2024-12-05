import torch
import torch.nn as nn

# Define the PatchGAN discriminator model
class PatchGANDiscriminator(nn.Module):
    def __init__(self):
        super(PatchGANDiscriminator, self).__init__()

        def conv_block(in_channels, out_channels, kernel_size=4, stride=2, padding=1, normalize=True, dropout=0.0):
            layers = [nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False)]
            if normalize:
                layers.append(nn.BatchNorm2d(out_channels))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            if dropout:
                layers.append(nn.Dropout(dropout))
            return nn.Sequential(*layers)


        # Layers of a discriminator
        self.enc1 = conv_block(6, 64, normalize=False)
        self.enc2 = conv_block(64, 128)
        self.enc3 = conv_block(128, 256, dropout=0.5)
        self.enc4 = conv_block(256, 512, dropout=0.5)
        self.enc5 = nn.Conv2d(512, 1, kernel_size=4, stride=1, padding=1)
        self.activation = nn.Sigmoid()


    # Forward path of PatchGAN Discriminator
    def forward(self, x, y):
        u = torch.cat([x,y], dim=1)
        enc1 = self.enc1(u)
        enc2 = self.enc2(enc1)
        enc3 = self.enc3(enc2)
        enc4 = self.enc4(enc3)
        enc5 = self.enc5(enc4)
        return self.activation(enc5)

if __name__ == "__main__":
    discriminator = PatchGANDiscriminator()
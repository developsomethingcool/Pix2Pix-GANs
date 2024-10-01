import torch
import torch.nn as nn

class PatchGANDiscriminator(nn.Module):
    def __init__(self):
        super(PatchGANDiscriminator, self).__init__()

        def conv_block(in_channels, out_channels, kernel_size=4, stride=2, padding=1, normalize=True):
            layers = [nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False)]
            if normalize:
                layers.append(nn.BatchNorm2d(out_channels))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *conv_block(3, 64, normalize=False), # (512 * 512) -> (256 * 256)  
            *conv_block(64, 128),  # (256 * 256) -> (128 * 128)
            *conv_block(128, 256),  # (128 * 128) -> (64 * 64)
            *conv_block(256, 512),  # (64 * 64) -> (32 * 32)
            *conv_block(512, 512),  # (32 * 32) -> (16 * 16)
            *conv_block(512, 512),  # (16 * 16) -> (8 * 8)
            nn.Conv2d(512, 1, kernel_size=4, stride=1, padding=0),  # (8 * 8) -> (7 * 7)
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)

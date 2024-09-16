import torch
import torch.nn as nn

class UNetGenerator(nn.Module):
    def __init__(self):
        super(UNetGenerator, self).__init__()

        def conv_block(in_channels, out_channels, kernel_size=4, stride=2, padding=1, normalize=True, dropout=0.0):
            layers = [nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False)]
            if normalize:
                layers.append(nn.BatchNorm2d(out_channels))
            if dropout:
                layers.append(nn.Dropout(dropout))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        def deconv_block(in_channels, out_channels, kernel_size=4, stride=2, padding=1, dropout=0.0):
            layers = [nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding, bias=False)]
            layers.append(nn.BatchNorm2d(out_channels))
            if dropout:
                layers.append(nn.Dropout(dropout))
            layers.append(nn.ReLU(inplace=True))
            return layers

        self.model = nn.Sequential(
            *conv_block(3, 64, normalize=False),  
            *conv_block(64, 128), 
            *conv_block(128, 256),  
            *conv_block(256, 512),  
            *conv_block(512, 512),  
            *conv_block(512, 512),  
            *conv_block(512, 512), 
            *conv_block(512, 512), 
            *deconv_block(512, 512),  
            *deconv_block(512, 512), 
            *deconv_block(512, 512),  
            *deconv_block(512, 512),  
            *deconv_block(512, 256),  
            *deconv_block(256, 128),  
            *deconv_block(128, 64),  
            nn.ConvTranspose2d(64, 3, kernel_size=4, stride=2, padding=1),
            nn.Tanh()
        )

    def forward(self, x):
        return self.model(x)

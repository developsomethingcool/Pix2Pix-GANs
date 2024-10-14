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
            return nn.Sequential(*layers)

        def deconv_block(in_channels, out_channels, kernel_size=4, stride=2, padding=1, dropout=0.0):
            layers = [nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding, bias=False)]
            layers.append(nn.BatchNorm2d(out_channels))
            if dropout:
                layers.append(nn.Dropout(dropout))
            layers.append(nn.ReLU(inplace=True))
            return nn.Sequential(*layers)

        # Encoder layers
        self.enc1 = conv_block(3, 64, normalize=False)
        self.enc2 = conv_block(64, 128)
        self.enc3 = conv_block(128, 256)
        self.enc4 = conv_block(256, 512)
        self.enc5 = conv_block(512, 512)
        self.enc6 = conv_block(512, 512)
        self.enc7 = conv_block(512, 512)
        self.enc8 = conv_block(512, 512)

        # Decoder layers
        self.dec1 = deconv_block(512, 512, dropout=0.2)
        self.dec2 = deconv_block(1024, 512, dropout=0.2)
        self.dec3 = deconv_block(1024, 512, dropout=0.2)
        self.dec4 = deconv_block(1024, 512)
        self.dec5 = deconv_block(1024, 256)
        self.dec6 = deconv_block(512, 128)
        self.dec7 = deconv_block(256, 64)

        self.final_layer = nn.Sequential(
            nn.ConvTranspose2d(128, 3, kernel_size=4, stride=2, padding=1),
            nn.Tanh()
        )

    def forward(self, x):
        # Encoder
        enc1 = self.enc1(x)
        enc2 = self.enc2(enc1)
        enc3 = self.enc3(enc2)
        enc4 = self.enc4(enc3)
        enc5 = self.enc5(enc4)
        enc6 = self.enc6(enc5)
        enc7 = self.enc7(enc6)
        enc8 = self.enc8(enc7)

        # Decoder with skip connections
        dec1 = self.dec1(enc8)
        dec2 = self.dec2(torch.cat([dec1, enc7], 1))
        dec3 = self.dec3(torch.cat([dec2, enc6], 1))
        dec4 = self.dec4(torch.cat([dec3, enc5], 1))
        dec5 = self.dec5(torch.cat([dec4, enc4], 1))
        dec6 = self.dec6(torch.cat([dec5, enc3], 1))
        dec7 = self.dec7(torch.cat([dec6, enc2], 1))

        output = self.final_layer(torch.cat([dec7, enc1], 1))
        return output


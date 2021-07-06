from torch import nn
import torch.nn.functional as F
import torch
from lib.models.blocks import RSU, DRSU


def init_weights_he_normal(neg_slope=1e-2):
    def init(m):
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
            m.weight = nn.init.kaiming_normal_(m.weight, a=neg_slope)
            if m.bias is not None:
                m.bias = nn.init.constant_(m.bias, 0)
    return init

class ACUNet(nn.Module):
    def __init__(self, heights, n_channels, n_classes, norm, nonlin, init="he_uniform", dropout=0.):
        super().__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes

        self.first_conv = RSU(heights[0], n_channels[0], n_channels[1] // 2, n_channels[1], norm, nonlin, False, dropout=dropout) 

        self.encoder_blocks = nn.ModuleList([
            RSU(heights[i - 1], n_channels[i], n_channels[i] // 2, n_channels[i + 1], norm, nonlin, True, dropout=dropout) 
            for i in range(1, len(n_channels) - 2)
            ])

        bottom_conv = DRSU(heights[-1], n_channels[-2], n_channels[-2] // 2, n_channels[-1], norm, nonlin,True, dropout=dropout)
        self.encoder_blocks.append(bottom_conv)

        self.decoder_blocks = nn.ModuleList([
            RSU(heights[i - 1], n_channels[i] + n_channels[i - 1], (n_channels[i] + n_channels[i - 1]) // 2, n_channels[i - 1], norm, nonlin, dropout) 
            for i in reversed(range(2, len(n_channels)))
            ])

        self.segment_heads = nn.ModuleList([nn.Conv2d(n_channels[1], n, kernel_size=1, bias=False) for n in n_classes])

        if init == "he_uniform": 
            pass # pytorch conv default initialization
        elif init == "he_normal":
            self.apply(init_weights_he_normal())

    def forward(self, x):
        x = self.first_conv(x)

        encoder_features = []
        for encoder_block in self.encoder_blocks:
            encoder_features.append(x)
            x = encoder_block(x)

        for i, decoder_block in enumerate(self.decoder_blocks):
            x = F.upsample_bilinear(x, scale_factor=2)
            x = torch.cat((x, encoder_features[-(i + 1)]), dim=1)
            x = decoder_block(x)

        output_logits = [classifier(x) for classifier in self.segment_heads]
        return output_logits

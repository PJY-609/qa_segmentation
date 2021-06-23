from torch import nn
import torch.nn.functional as F
from lib.models.blocks import BasicConvBlock, BasicUpBlock


class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, norm, nonlin, init="he_uniform", deep_supervise=False, dropout=0.):
        super().__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes

        self.first_conv = BasicConvBlock(n_channels[0], n_channels[1], norm, nonlin, strided=False, dropout=dropout) 

        self.encoder_blocks = nn.ModuleList([
            BasicConvBlock(n_channels[i], n_channels[i + 1], norm, nonlin, strided=True, dropout=dropout) 
            for i in range(1, len(n_channels) - 1)
            ])

        self.decoder_blocks = nn.ModuleList([
            BasicUpBlock(n_channels[i] + n_channels[i - 1], n_channels[i - 1], norm, nonlin, dropout) 
            for i in reversed(range(2, len(n_channels)))
            ])

        self.segment_heads = nn.ModuleList([nn.Conv2d(i, n_classes, kernel_size=1, bias=False) for i in reversed(n_channels[1:-2])])

    def forward(self, x):
        x = self.first_conv(x)

        encoder_features = []
        for encoder_block in self.encoder_blocks:
            encoder_features.append(x)
            x = encoder_block(x)

        decoder_features = [x]
        for i, decoder_block in enumerate(self.decoder_blocks):
            x = decoder_block(x, encoder_features[-(i + 1)])
            decoder_features.append(x)


        output_logits = []
        for i, (classifier, feature) in enumerate(zip(self.segment_heads, decoder_features[2:])):
            if i == 2:
                feat = feature
            else:
                feat = F.interpolate(feature, scale_factor=2 ** (2 - i), mode='bilinear', align_corners=None)
            logits = classifier(feat)
            output_logits.append(logits)
        
        return output_logits

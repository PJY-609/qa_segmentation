from torch import nn
from lib.models.blocks import BasicConvBlock, BasicUpBlock


def init_weights_he_normal(neg_slope=1e-2):
    def init(m):
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
            m.weight = nn.init.kaiming_normal_(m.weight, a=neg_slope)
            if m.bias is not None:
                m.bias = nn.init.constant_(m.bias, 0)
    return init

class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, norm, nonlin, init="he_uniform", dropout=0.):
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

        self.segment_head = nn.Conv2d(n_channels[1], n_classes, kernel_size=1, bias=False)

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

        decoder_features = [x]
        for i, decoder_block in enumerate(self.decoder_blocks):
            x = decoder_block(x, encoder_features[-(i + 1)])
            decoder_features.append(x)

        output_logits = self.segment_head(x)
        return output_logits, decoder_features

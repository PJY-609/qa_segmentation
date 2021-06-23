from torch import nn
from lib.models.blocks import BasicConvBlock, BasicUpBlock_


def init_weights_he_normal(neg_slope=1e-2):
    def init(m):
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
            m.weight = nn.init.kaiming_normal_(m.weight, a=neg_slope)
            if m.bias is not None:
                m.bias = nn.init.constant_(m.bias, 0)
    return init

class UNetPlusPlus(nn.Module):
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
            nn.ModuleList([
                BasicUpBlock_(in_channels + (i + 1) * out_channels, out_channels, norm, nonlin, dropout) 
                for in_channels, out_channels in zip(n_channels[2:n+1], n_channels[1:n])
                ]) for i, n in enumerate(reversed(range(2, len(n_channels))))
        ])

        self.segment_heads = nn.ModuleList([nn.Conv2d(n_channels[1], n, kernel_size=1, bias=False) for n in n_classes])

        if init == "he_uniform": 
            pass # pytorch conv default initialization
        elif init == "he_normal":
            self.apply(init_weights_he_normal())

    def forward(self, x):
        x = self.first_conv(x)

        encoder_features = [x]
        for encoder_block in self.encoder_blocks:
            x = encoder_block(x)
            encoder_features.append(x)
            
        features = [encoder_features]

        for i, decoder_blocks in enumerate(self.decoder_blocks):
            decoder_features = []
            for j, decoder_block in enumerate(decoder_blocks):
                skip_features = [features[k][j] for k in range(i + 1)]
                x = decoder_block(features[i][j + 1], skip_features)
                decoder_features.append(x)

            features.append(decoder_features)

        output_logits = [classifier(x) for classifier in self.segment_heads]
        return output_logits
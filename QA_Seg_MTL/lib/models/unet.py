from torch import nn
from lib.models.blocks import BasicConvBlock, BasicUpBlock


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

        self.global_decoder_blocks = nn.ModuleList([
            BasicUpBlock(n_channels[i] + n_channels[i - 1], n_channels[i - 1], norm, nonlin, dropout) 
            for i in reversed(range(2, len(n_channels)))
            ])

        self.local_decoder_blocks = nn.ModuleList([
            BasicUpBlock(n_channels[i] + n_channels[i - 1], n_channels[i - 1], norm, nonlin, dropout) 
            for i in reversed(range(2, len(n_channels)))
            ])

        self.segment_head = nn.Conv2d(n_channels[1], n_classes, kernel_size=1, bias=False)
        self.distmap_head = nn.Conv2d(n_channels[1], 1, kernel_size=1, bias=False)

    def forward(self, x):
        x = self.first_conv(x)

        encoder_features = []
        for encoder_block in self.encoder_blocks:
            encoder_features.append(x)
            x = encoder_block(x)

        global_feature = x
        local_feature = x

        for i, decoder_block in enumerate(self.global_decoder_blocks):
            global_feature = decoder_block(global_feature, encoder_features[-(i + 1)])

        for i, decoder_block in enumerate(self.local_decoder_blocks):
            local_feature = decoder_block(local_feature, encoder_features[-(i + 1)])
        
        distmap_logits = self.distmap_head(local_feature)
        segment_logits = self.segment_head(global_feature)
        
        return segment_logits, distmap_logits

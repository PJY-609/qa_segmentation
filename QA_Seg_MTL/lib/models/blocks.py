from torch import nn
import torch
import torch.nn.functional as F


def normalization(norm, n_channels):
	if norm == "batch":
		layer = nn.BatchNorm2d(n_channels)
	elif norm == "instance":
		layer = nn.InstanceNorm2d(n_channels)
	else:
		layer = nn.Identity()
	return layer

def activation(nonlin):
	if nonlin == "relu":
		layer = nn.ReLU(inplace=True)
	elif nonlin == "lrelu":
		layer = nn.LeakyReLU(negative_slope=0.01, inplace=True)
	else:
		layer = nn.Identity()
	return layer

def BasicConvBlock(in_channels, out_channels, norm, nonlin, strided, dropout):
	seq = nn.Sequential(
		nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
		nn.Dropout2d(dropout, inplace=True),
		normalization(norm, out_channels),
		activation(nonlin),

		nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=2 if strided else 1, padding=1, bias=False),
		nn.Dropout2d(dropout, inplace=True),
		normalization(norm, out_channels),
		activation(nonlin)
	)
	return seq


class BasicUpBlock(nn.Module):
	def __init__(self, in_channels, out_channels, norm, nonlin, dropout):
		super().__init__()
		self.up_sample = nn.Upsample(scale_factor=2, mode="bilinear")
		self.conv = BasicConvBlock(in_channels, out_channels, norm, nonlin, False, dropout)

	def forward(self, lr_feature, hr_feature):
		x = self.up_sample(lr_feature)
		x = torch.cat([hr_feature, x], axis=1)
		x = self.conv(x)
		return x


class BasicUpBlock_(nn.Module):
	def __init__(self, in_channels, out_channels, norm, nonlin, dropout):
		super().__init__()
		self.up_sample = nn.Upsample(scale_factor=2, mode="bilinear")
		self.conv = BasicConvBlock(in_channels, out_channels, norm, nonlin, False, dropout)

	def forward(self, lr_feature, features):
		x = self.up_sample(lr_feature)
		features.append(x)
		x = torch.cat(features, axis=1)
		x = self.conv(x)
		return x


class PPM(nn.Module):
    def __init__(self, in_dim, reduction_dim, bins):
        super().__init__()
        self.features = []
        for bin_ in bins:
            self.features.append(nn.Sequential(
                nn.AdaptiveAvgPool2d(bin_),
                nn.Conv2d(in_dim, reduction_dim, kernel_size=1, bias=False),
                nn.BatchNorm2d(reduction_dim),
                nn.ReLU(inplace=True)
            ))
        self.features = nn.ModuleList(self.features)

    def forward(self, x):
        x_size = x.size()
        out = [x]
        for f in self.features:
            out.append(F.interpolate(f(x), x_size[2:], mode='bilinear', align_corners=True))
        return torch.cat(out, 1)




def depthwise_separable_conv(in_channels, out_channels, strided=False, dilate=1):
	seq = nn.Sequential(
			nn.Conv2d(in_channels, in_channels, kernel_size=3, groups=in_channels, stride=2 if strided else 1, padding=1*dilate, dilation=1 * dilate, bias=False),
			nn.Conv2d(in_channels, out_channels, kernel_size=1)
		)
	return seq


def BasicDSConvBlock(in_channels, out_channels, norm, nonlin, strided, dropout):
	seq = nn.Sequential(
		depthwise_separable_conv(in_channels, out_channels),
		nn.Dropout2d(dropout, inplace=True),
		normalization(norm, out_channels),
		activation(nonlin),

		depthwise_separable_conv(out_channels, out_channels, strided=strided),
		nn.Dropout2d(dropout, inplace=True),
		normalization(norm, out_channels),
		activation(nonlin)
	)
	return seq



def dsconv_dropout_norm_nonlin(in_channels, out_channels, norm, nonlin, strided=False, dilate=1, dropout=0.):
	seq = nn.Sequential(
		depthwise_separable_conv(in_channels, out_channels, strided=strided, dilate=dilate),
		nn.Dropout2d(dropout, inplace=True),
		normalization(norm, out_channels),
		activation(nonlin)
		)
	return seq
	

# ReSidual U block
class RSU(nn.Module):
    def __init__(self, height, in_ch, mid_ch, out_ch, norm, nonlin, strided=False, dropout=0.):
        super().__init__()
        self.convin = dsconv_dropout_norm_nonlin(in_ch, out_ch, norm, nonlin, strided, dropout=dropout)

        self.down_conv1 = dsconv_dropout_norm_nonlin(out_ch, mid_ch, norm, nonlin, dropout=dropout)
        self.down_convs = nn.ModuleList([
        		dsconv_dropout_norm_nonlin(mid_ch, mid_ch, norm, nonlin, dropout=dropout)
        		for i in range(2, height)
        		])

        self.up_conv1 = dsconv_dropout_norm_nonlin(mid_ch * 2, out_ch, norm, nonlin, dropout=dropout)
        self.up_convs = nn.ModuleList([
				dsconv_dropout_norm_nonlin(mid_ch * 2, mid_ch, norm, nonlin, dropout=dropout)
				for i in reversed(range(2, height))
        	])
        
        self.bottom_conv = dsconv_dropout_norm_nonlin(mid_ch, mid_ch, norm, nonlin, dilate=2, dropout=dropout)

    def forward(self, x):
        x = self.convin(x)

        feat = self.down_conv1(x)
        
        features = [feat]
        for conv in self.down_convs:
        	feat = conv(F.max_pool2d(feat, kernel_size=2))
        	features.append(feat)

        feat = self.bottom_conv(feat)

        for i, conv in enumerate(self.up_convs):
        	feat = conv(torch.cat((feat, features[-(i + 1)]), dim=1))
        	feat = F.upsample_bilinear(feat, scale_factor=2)

        feat = self.up_conv1(torch.cat((feat, features[0]), dim=1))
        return feat + x

# Dilated ReSidual U block
class DRSU(nn.Module):
	def __init__(self, height, in_ch, mid_ch, out_ch, norm, nonlin, strided, dropout=0.):
		super().__init__()
		self.convin = dsconv_dropout_norm_nonlin(in_ch, out_ch, norm, nonlin, strided=strided, dropout=dropout)

		self.en_conv1 = dsconv_dropout_norm_nonlin(out_ch, mid_ch, norm, nonlin, dilate=1, dropout=dropout)

		self.en_convs = nn.ModuleList([
				dsconv_dropout_norm_nonlin(mid_ch, mid_ch, norm, nonlin, dilate=2 ** (i - 1), dropout=dropout)
				for i in range(2, height)
			])

		self.de_conv1 = dsconv_dropout_norm_nonlin(2 * mid_ch, out_ch, norm, nonlin, dropout=dropout)

		self.de_convs = nn.ModuleList([
				dsconv_dropout_norm_nonlin(2 * mid_ch, mid_ch, norm, nonlin, dilate=2 ** (i - 1), dropout=dropout)
				for i in reversed(range(2, height))
			])

		self.bottom_conv = dsconv_dropout_norm_nonlin(mid_ch, mid_ch, norm, nonlin, dilate=2 ** (height - 1), dropout=dropout)

	def forward(self, x):
		x = self.convin(x)

		feat = self.en_conv1(x)

		features = [feat]

		for conv in self.en_convs:
			feat = conv(feat)
			features.append(feat)

		feat = self.bottom_conv(feat)

		for i, conv in enumerate(self.de_convs):
			feat = conv(torch.cat((feat, features[-(i + 1)]), dim=1))

		feat = self.de_conv1(torch.cat((feat, features[0]), dim=1))
		return feat + x
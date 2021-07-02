from torch import nn
import torch


def normalization(norm, n_channels):
	if norm == "batch":
		layer = nn.BatchNorm2d(n_channels)
	elif norm == "instance":
		layer = nn.InstanceNorm2d(n_channels)
	else:
		layer = nn.Lambda(lambda x: x)
	return layer

def activation(nonlin):
	if nonlin == "relu":
		layer = nn.ReLU(inplace=True)
	elif nonlin == "lrelu":
		layer = nn.LeakyReLU(negative_slope=0.01, inplace=True)
	else:
		layer = nn.Identity()
	return layer


def Dense(in_channels, out_channels, nonlin, dropout):
	seq = nn.Sequential(
		nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=True),
		nn.Dropout(dropout, inplace=True),
		nn.BatchNorm1d(out_channels),
		activation(nonlin)
	)
	return seq

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

def conv_dropout_norm_nonlin(in_channels, out_channels, kernel_size, norm, nonlin, strided, dropout):
	padding = 0 if kernel_size == 1 else 1
	stride = 2 if strided else 1

	seq = nn.Sequential(
		nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, stride=stride, bias=False),
		nn.Dropout2d(dropout, inplace=True),
		normalization(norm, out_channels),
		activation(nonlin)
		)
	return seq


class ResConvBlock(nn.Module):
	def __init__(self, in_channels, out_channels, norm, nonlin, strided, dropout):
		super().__init__()
		self.conv_block = BasicConvBlock(in_channels, out_channels, norm, nonlin, False, dropout)
		self.skip_conv = conv_dropout_norm_nonlin(in_channels, out_channels, 1, norm, nonlin, False, dropout)
		
		self.strided = strided
		if strided:
			seq = conv_dropout_norm_nonlin(out_channels, out_channels, 3, norm, nonlin, strided, dropout)
			self.add_module("strided_conv", seq)

	def forward(self, x):
		x = self.skip_conv(x) + self.conv_block(x)

		if self.strided:
			x = self.strided_conv(x)
		return x

class ResUpBlock(nn.Module):
	def __init__(self, in_channels, out_channels, norm, nonlin, dropout):
		super().__init__()
		self.up_sample = nn.Upsample(scale_factor=2, mode="bilinear")
		self.conv = ResConvBlock(in_channels, out_channels, norm, nonlin, False, dropout)

	def forward(self, lr_feature, hr_feature):
		x = self.up_sample(lr_feature)
		x = torch.cat([hr_feature, x], axis=1)
		x = self.conv(x)
		return x
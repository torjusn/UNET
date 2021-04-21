
import math

#torch
import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvNet(nn.Module):
	'''
	Basic image classifier for comparison
	'''
	def __init__(self, channels_in, conv_filters, n_classes):

		super(ConvNet, self).__init__()

		self.activation = nn.LeakyReLU(negative_slope=0.1)

		self.conv11 = nn.utils.weight_norm(nn.Conv2d(in_channels=channels_in, out_channels=conv_filters[0], kernel_size=3, stride=1, padding=1))
		self.batchnorm11 = nn.BatchNorm2d(num_features=conv_filters[0])
		self.conv12 = nn.utils.weight_norm(nn.Conv2d(in_channels=conv_filters[0], out_channels=conv_filters[0], kernel_size=3, stride=1, padding=1))
		self.batchnorm12 = nn.BatchNorm2d(num_features=conv_filters[0])	
		self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
		self.dropout1  = nn.Dropout(0.5)


		self.conv21 = nn.utils.weight_norm(nn.Conv2d(in_channels=conv_filters[0], out_channels=conv_filters[1], kernel_size=3, stride=1, padding=1))
		self.batchnorm21 = nn.BatchNorm2d(num_features=conv_filters[1])
		self.conv22 = nn.utils.weight_norm(nn.Conv2d(in_channels=conv_filters[1], out_channels=conv_filters[1], kernel_size=3, stride=1, padding=1))
		self.batchnorm22 = nn.BatchNorm2d(num_features=conv_filters[1])	
		self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
		self.dropout2  = nn.Dropout(0.5)

		self.conv31 = nn.utils.weight_norm(nn.Conv2d(in_channels=conv_filters[1], out_channels=conv_filters[1], kernel_size=3, stride=1, padding=0))
		self.batchnorm31 = nn.BatchNorm2d(num_features=conv_filters[1])
		self.conv32 = nn.utils.weight_norm(nn.Conv2d(in_channels=conv_filters[1], out_channels=conv_filters[0], kernel_size=3, stride=1, padding=0))
		self.batchnorm32 = nn.BatchNorm2d(num_features=conv_filters[0])	
		self.avgpool3 = nn.AvgPool2d(kernel_size=6, stride=2, padding=0)

		self.fully_connected = nn.utils.weight_norm(nn.Linear(in_features=conv_filters[0], out_features=n_classes))

	def forward(self, x):
		x = self.activation(self.batchnorm11(self.conv11(x)))
		x = self.activation(self.batchnorm12(self.conv12(x)))
		x = self.maxpool1(x)
		x = self.dropout1(x)

		x = self.activation(self.batchnorm21(self.conv21(x)))
		x = self.activation(self.batchnorm22(self.conv22(x)))
		x = self.maxpool2(x)
		x = self.dropout2(x)

		x = self.activation(self.batchnorm31(self.conv31(x)))
		x = self.activation(self.batchnorm32(self.conv32(x)))
		x = self.avgpool3(x)

		x = self.fully_connected(x)

		return x

class FCN32(nn.Module):
	'''
	Basic segmentation classifier for comparison (not really 32 but without skip connections so similar to 32)

	Output should be (N, C, H, W) to compare with target (N, H, W)
	'''
	def __init__(self, channels_in, num_classes, **kwargs):

		super(FCN32, self).__init__()

		self.activation = nn.LeakyReLU(negative_slope=0.1)

		#layer 1
		self.conv11 = nn.utils.weight_norm(nn.Conv2d(channels_in, 128, kernel_size=3, stride=1, padding=100))
		self.batchnorm11 = nn.BatchNorm2d(128)
		self.conv12 = nn.utils.weight_norm(nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1))
		self.batchnorm12 = nn.BatchNorm2d(128)	
		self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)#1/2
		self.dropout1  = nn.Dropout(0.5)

		#layer 2 
		self.conv21 = nn.utils.weight_norm(nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1))
		self.batchnorm21 = nn.BatchNorm2d(256)
		self.conv22 = nn.utils.weight_norm(nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1))
		self.batchnorm22 = nn.BatchNorm2d(256)	
		self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)#1/4
		self.dropout2  = nn.Dropout(0.5)

		#layer 3
		self.conv31 = nn.utils.weight_norm(nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1))
		self.batchnorm31 = nn.BatchNorm2d(256)
		self.conv32 = nn.utils.weight_norm(nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1))
		self.batchnorm32 = nn.BatchNorm2d(256)	
		self.conv33 = nn.utils.weight_norm(nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1))
		self.batchnorm33 = nn.BatchNorm2d(256)
		self.maxpool3 = nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)#1/8
		self.dropout3  = nn.Dropout(0.5)

		#layer 4
		self.conv41 = nn.utils.weight_norm(nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1))
		self.batchnorm41 = nn.BatchNorm2d(512)
		self.conv42 = nn.utils.weight_norm(nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1))
		self.batchnorm42 = nn.BatchNorm2d(512)	
		self.conv43 = nn.utils.weight_norm(nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1))
		self.batchnorm43 = nn.BatchNorm2d(512)	
		self.maxpool4 = nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)#1/16
		self.dropout4  = nn.Dropout(0.5)

		#layer 5
		self.conv51 = nn.utils.weight_norm(nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1))
		self.batchnorm51 = nn.BatchNorm2d(512)
		self.conv52 = nn.utils.weight_norm(nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1))
		self.batchnorm52 = nn.BatchNorm2d(512)	
		self.conv53 = nn.utils.weight_norm(nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1))
		self.batchnorm53 = nn.BatchNorm2d(512)	
		self.maxpool5 = nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)#1/32
		self.dropout5  = nn.Dropout(0.5)

		#fully connected 6
		self.conv61 = nn.utils.weight_norm(nn.Conv2d(512, 4096, kernel_size=7))
		self.batchnorm61 = nn.BatchNorm2d(4096)

		#fully connected 7
		self.conv71 = nn.utils.weight_norm(nn.Conv2d(4096, 4096, kernel_size=1))
		self.batchnorm71 = nn.BatchNorm2d(4096)

		self.conv72 = nn.utils.weight_norm(nn.Conv2d(4096, num_classes, kernel_size=1))
		self.upsample = nn.ConvTranspose2d(num_classes, num_classes, 64, stride=32, bias=False)

	def forward(self, x):
		input_shape = x.shape

		#layer 1
		x = self.activation(self.batchnorm11(self.conv11(x)))
		x = self.activation(self.batchnorm12(self.conv12(x)))
		x = self.maxpool1(x)
		x = self.dropout1(x)

		#layer 2
		x = self.activation(self.batchnorm21(self.conv21(x)))
		x = self.activation(self.batchnorm22(self.conv22(x)))
		x = self.maxpool2(x)
		x = self.dropout2(x)

		#layer 3
		x = self.activation(self.batchnorm31(self.conv31(x)))
		x = self.activation(self.batchnorm32(self.conv32(x)))
		x = self.activation(self.batchnorm33(self.conv33(x)))
		x = self.maxpool3(x)
		x = self.dropout3(x)

		#layer 4
		x = self.activation(self.batchnorm41(self.conv41(x)))
		x = self.activation(self.batchnorm42(self.conv42(x)))
		x = self.activation(self.batchnorm43(self.conv43(x)))
		x = self.maxpool4(x)
		x = self.dropout4(x)

		#layer 5
		x = self.activation(self.batchnorm51(self.conv51(x)))
		x = self.activation(self.batchnorm52(self.conv52(x)))
		x = self.activation(self.batchnorm53(self.conv53(x)))
		x = self.maxpool5(x)
		x = self.dropout5(x)

		#layer 6
		x = self.activation(self.batchnorm61(self.conv61(x)))

		#layer 7
		x = self.activation(self.batchnorm71(self.conv71(x)))
		x = self.activation(self.conv72(x))

		x = self.upsample(x)

		x = x[:, :, 19:19 + input_shape[2], 19:19 + input_shape[3]].contiguous()

		return x

class OneLayerUNET(nn.Module):
	'''
	Basic segmentation classifier for comparison with gcn.
	Output should be (N, classes, H, W) to compare with target (N, H, W)
	'''
	def __init__(self, channels_in, num_classes, **kwargs):

		super(UNET, self).__init__()

		self.activation = nn.ReLU()

		self.conv11 = nn.utils.weight_norm(nn.Conv2d(channels_in, 128, kernel_size=3, stride=1, padding=1))
		self.batchnorm11 = nn.BatchNorm2d(128)
		self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)#1/2
		self.dropout1  = nn.Dropout(0.5)

		self.conv21 = nn.utils.weight_norm(nn.Conv2d(128, num_classes, kernel_size=3, stride=1, padding=1))
		self.batchnorm21 = nn.BatchNorm2d(num_classes)
		self.upsample2 = nn.ConvTranspose2d(num_classes, num_classes, 2, stride=2, bias=False)

	def forward(self, x):

		x = self.activation(self.batchnorm11(self.conv11(x)))
		x = self.maxpool1(x)
		x = self.dropout1(x)

		x = self.activation(self.batchnorm21(self.conv21(x)))

		x = self.upsample2(x)

		return x

class UNET(nn.Module):
	'''
	Basic segmentation classifier for comparison with gcn.
	Consists of contract and expand blocks with skip connections similar to:
	https://medium.com/analytics-vidhya/creating-a-very-simple-u-net-model-with-pytorch-for-semantic-segmentation-of-satellite-images-223aa216e705

	Output should be (N, classes, H, W) to compare with target (N, H, W)
	'''

	def downsampleBlock(self, channels_in, channels_out, kernel_size, padding):

		block = nn.Sequential(
			nn.Conv2d(channels_in, channels_out, kernel_size=kernel_size, stride=1, padding=1),
			nn.BatchNorm2d(channels_out),
			nn.ReLU(),
			nn.Conv2d(channels_out, channels_out, kernel_size=kernel_size, stride=1, padding=1),
			nn.BatchNorm2d(channels_out),
			nn.ReLU(),
			nn.MaxPool2d(kernel_size=2, stride=2) #one paper adds padding 1?
			)

		return block

	def upsampleBlock(self, channels_in, channels_out, kernel_size, padding):

		block = nn.Sequential(
			nn.Conv2d(channels_in, channels_out, kernel_size, stride=1, padding=padding),
			nn.BatchNorm2d(channels_out),
			nn.ReLU(),
			nn.Conv2d(channels_out, channels_out, kernel_size, stride=1, padding=padding),
			nn.BatchNorm2d(channels_out),
			nn.ReLU(),
			nn.ConvTranspose2d(channels_out, channels_out, kernel_size=3, stride=2, padding=1, output_padding=1)
			)

		return block

	def __init__(self, channels_in, num_classes, **kwargs):

		super(UNET, self).__init__()

		self.activation = nn.ReLU()
		self.dropout = nn.Dropout(0.5)

		self.conv1 = self.downsampleBlock(channels_in, 128, 3, 1)
		self.conv2 = self.downsampleBlock(128, 256, 3, 1)
		self.conv3 = self.downsampleBlock(256, 512, 3, 1)

		self.deconv3 = self.upsampleBlock(512, 256, 3, 1)
		self.deconv2 = self.upsampleBlock(256*2, 128, 3, 1)#times 2 because of concat
		self.deconv1 = self.upsampleBlock(128*2, num_classes, 3, 1)


	def forward(self, x):

		downsample1 = self.dropout(self.conv1(x))
		#downsample1 = self.dropout(downsample1)

		downsample2 = self.conv2(downsample1)
		#downsample2 = self.dropout(downsample2)

		downsample3 = self.conv3(downsample2)
		#downsample3 = self.dropout(downsample3)


		upsample3 = self.deconv3(downsample3)
		#upsample3 = self.dropout(upsample3)

		upsample2 = self.deconv2(torch.cat([upsample3, downsample2], 1))
		#upsample2 = self.dropout(upsample2)

		upsample1 = self.deconv1(torch.cat([upsample2, downsample1], 1))

		return upsample1



#builtin
import argparse
import os

#misc
import numpy as np
import matplotlib.pyplot as plt

#torch
import torch
import torch.nn as nn
import torch.utils.data as data
import torch.optim as optim

#local
from architectures import convnet

parser = argparse.ArgumentParser(description='PyTorch Simple CNN')

#optimizations
parser.add_argument('--epochs', default=64, type=int, metavar='N',
                    help='number of epochs')
parser.add_argument('--eval-epochs', default=5, type=int, metavar='N',
                    help='evaluate model on validation set after eval_epochs (integer) iterations')
parser.add_argument('--batch-size', default=2, type=int, metavar='N',
					help='train batchsize')
parser.add_argument('--learning-rate', default=0.002, type=float,
                    metavar='LR', help='initial learning rate')

parser.add_argument('--checkpoint-directory', default='cnn@trento_stride8',
					help='directory to load checkpoint from, e.g. modelname@120labels')

parser.add_argument('--dataset', default='trento',
					help='Choose dataset from [trento], [trento_stride8], [houston], [s1s2glcm].')
args = parser.parse_args()

if args.dataset == 'trento':
	import dataset.trento as dataset
elif args.dataset == 'houston':
	import dataset.houston as dataset
elif args.dataset == 'trento_stride8':
	import dataset.trento_stride8 as dataset
elif args.dataset == 's1s2glcm':
	import dataset.s1s2glcm as dataset
else:
	print('Please pick dataset from [trento], [houston], [s1s2]')
	exit()

#use_cuda = torch.cuda.is_available()
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

#device_count = torch.cuda.device_count()
#device_name = torch.cuda.get_device_name(0)
#device_index=torch.cuda.current_device()

#make experiments reproducible
np.random.seed(1)
torch.manual_seed(1)
torch.cuda.manual_seed(1)

#optimize graph on first execution, only use if net input/output size always same
#https://github.com/IgorSusmelj/pytorch-styleguide
torch.backends.cudnn.benchmark = True

#Save each run to separate checkpoint directory
###-------------------------------------------------------------------------
#python inference.py --checkpoint-directory cnn@trento
#python inference.py --checkpoint-directory cnn@trento_stride8 --dataset trento_stride8

def main():

	#if not os.path.exists(args.checkpoint_directory):
	#	os.makedirs(args.checkpoint_directory)

	#load dataset
	###-------------------------------------------------------------------------
	root = os.path.abspath('../data-local')

	if args.dataset == 'trento':
		print('==> Preparing trento segmentation dataset')

		import dataset.trento as dataset

		num_channels = 77 #7 + 70
		num_classes = 7 #0-6

		train_dataset, val_dataset = dataset.get_trento(root)
		train_loader = data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0, drop_last=True)
		val_loader = data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0, drop_last=True)

	elif args.dataset == 'trento_stride8':
		print('==> Preparing trento_stride8 segmentation dataset')

		import dataset.trento_stride8 as dataset

		num_channels = 77 #7 + 70 
		num_classes = 7 #0-6

		train_dataset, val_dataset = dataset.get_trento_stride8(root)
		train_loader = data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0, drop_last=True)
		val_loader = data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0, drop_last=True)

	elif args.dataset == 'houston':
		print('==> Preparing houston segmentation dataset')

		import dataset.houston as dataset

		num_channels = 151 #7 + 144 
		num_classes = 16

		train_dataset, val_dataset = dataset.get_houston(root)
		train_loader = data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0, drop_last=True)
		val_loader = data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0, drop_last=True)

	elif args.dataset == 's1s2':
		print('==> Preparing s1s1seg segmentation dataset')

		import dataset.s1s2 as dataset

		num_channels = 15 #13+2
		num_classes = 6

		train_dataset, val_dataset = dataset.get_s1s2(root) #len 48 if 64x64
		train_loader = data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0, drop_last=True)
		val_loader = data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0, drop_last=True)

	#create model+criterion+optimizer
	###-------------------------------------------------------------------------
	params = dict(vars(args))

	print('==> creating UNET model')

	#Pytorch: If you need to move a model to GPU via .cuda(), please do so before constructing optimizers for it.
	model = convnet.UNET(channels_in=num_channels, num_classes=num_classes, **params)
	model.to(device)

	criterion = nn.CrossEntropyLoss()

	#optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
	optimizer = optim.SGD(model.parameters(), lr=args.learning_rate)

	print('==> loading from checkpoint')

	#make sure checkpoint file exists
	load_dir = os.path.abspath(os.path.join('results', args.checkpoint_directory))
	#load_dir = os.path.join(load_dir, '.3')
	print(load_dir)
	load_path = os.path.join(load_dir, 'best_model.pth')
	assert os.path.isfile(load_path), 'Error: no checkpoint directory found!'

	checkpoint = torch.load(load_path)
	model.load_state_dict(checkpoint['model_state_dict'])
	optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
	epoch = checkpoint['epoch']
	loss = checkpoint['loss']


	model.eval()

	with torch.no_grad():

		train_iter = iter(train_loader)

		images, masks = train_iter.next()

		images, masks = images.to(device), masks.to(device)


		logits = model(images)
		_, predicted = torch.max(logits.detach(), dim=1)


		fig=plt.figure()
		plt.imshow(predicted[0].cpu())

		fig=plt.figure()
		plt.imshow(masks[0].cpu())

		plt.show()


def validate(val_loader, model, criterion, epoch, args):

	model.eval()

	with torch.no_grad():

		num_batches = len(val_loader)
		avg_epoch_acc = 0.0
		avg_epoch_loss = 0.0

		for batch_idx, (images, masks) in enumerate(val_loader):

			images, masks = images.to(device), masks.to(device)

			logits = model(images)

			#loss of entire minibatch divided by N batch elements
			loss = criterion(logits, masks)

			avg_batch_acc = pixelAccuracy(logits.detach(), masks.detach())

			avg_epoch_acc += avg_batch_acc

			avg_epoch_loss += loss.detach().item()

			#IOU

			#mIOU

		avg_epoch_acc = avg_epoch_acc / num_batches
		avg_epoch_loss = avg_epoch_loss / num_batches

		#print avg val loss and accuracy for epoch
		print('Epoch: [{}/{}] Avg Epoch Val Loss:[{:.4e}] Avg Epoch Val Acc: [{:.3f}]'.format(epoch, args.epochs, avg_epoch_loss, avg_epoch_acc))
		return avg_epoch_acc, avg_epoch_loss

def pixelAccuracy(logits, masks):
	'''
	Calculates	pixel accuracy average over a minibatch
	Takes in logits.detached() and masks.detached() to leave no risk of changing computational graph.

	Args: 
		logits: (N, classes, H, W)
		masks: (N, H, W)
	'''

	_, predicted = torch.max(logits, dim=1)

	correct_pixels_batch = torch.sum(torch.eq(predicted, masks)).item()
	total_pixels_batch = torch.numel(masks)

	avg_batch_acc = correct_pixels_batch / total_pixels_batch

	return avg_batch_acc

main()
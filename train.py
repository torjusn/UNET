#builtin
import argparse
import os
import shutil

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


#adam 0.01 gave 0.958 on trento_stride8

parser = argparse.ArgumentParser(description='PyTorch Simple CNN')

#optimizations
parser.add_argument('--epochs', default=64, type=int, metavar='N',
					help='number of epochs')

parser.add_argument('--eval-epochs', default=5, type=int, metavar='N',
					help='evaluate model on validation set between every eval_epochs (integer) iterations')

parser.add_argument('--batch-size', default=2, type=int, metavar='N',
					help='train batchsize')

parser.add_argument('--learning-rate', default=0.01, type=float,
					metavar='LR', help='initial learning rate')


parser.add_argument('--checkpoint-directory', default='result',
					help='directory to save checkpoint, e.g. modelname@120labels')

parser.add_argument('--dataset', default='s1s2',
					help='Choose dataset from [trento], [trento_stride8], [houston], [s1s2].')

args = parser.parse_args()

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

#save model state separate from checkpoint file if current avg_epoch_acc or avg_epoch_loss is best until now
best_avg_epoch_acc = 0.0
#best_avg_epoch_loss = 0.0

#Save each run to separate checkpoint directory
###-------------------------------------------------------------------------
#python train.py --checkpoint-directory cnn@trento
#python train.py --checkpoint-directory cnn@trento_stride8 --dataset trento_stride8
#python train.py --checkpoint-directory cnn@s1s2 --dataset s1s2

def main():

	if not os.path.exists(os.path.join('results', args.checkpoint_directory)):
		os.makedirs(os.path.join('results', args.checkpoint_directory))

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
		train_loader = data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0, drop_last=True)
		val_loader = data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0, drop_last=True)

	elif args.dataset == 'houston':
		print('==> Preparing houston segmentation dataset')

		import dataset.houston as dataset

		num_channels = 151 #7 + 144 
		num_classes = 16

		train_dataset, val_dataset = dataset.get_houston(root)
		train_loader = data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0, drop_last=True)
		val_loader = data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0, drop_last=True)

	elif args.dataset == 's1s2':
		print('==> Preparing s1s1seg segmentation dataset')

		import dataset.s1s2 as dataset

		num_channels = 15 #13+2
		num_classes = 6

		train_dataset, val_dataset = dataset.get_s1s2(root) #len 48 if 64x64
		train_loader = data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0, drop_last=True)
		val_loader = data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0, drop_last=True)
	
	else:
		print('Please pick a dataset from [trento], [houston], [s1s2]')
		exit()


	#create model+criterion+optimizer
	###-------------------------------------------------------------------------
	params = dict(vars(args))

	print('==> creating UNET model')

	#Pytorch: If you need to move a model to GPU via .cuda(), please do so before constructing optimizers for it.
	model = convnet.UNET(channels_in=num_channels, num_classes=num_classes, **params)
	model.to(device)

	criterion = nn.CrossEntropyLoss()

	optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
	#optimizer = optim.SGD(model.parameters(), lr=args.learning_rate)

	for epoch in range(args.epochs):

		#train one epoch on training data
		train_loss = train(train_loader, model, criterion, optimizer, epoch, args)

		#run validation on val dataset every eval_epochs

		if epoch % args.eval_epochs == 0:
			print('Evaluating model on validation set')
			val_avg_epoch_acc, val_avg_epoch_loss = validate(val_loader, model, criterion, epoch, args)

		is_best = val_avg_epoch_acc > best_avg_epoch_acc
		#is_best = val_loss < best_loss

		#to resume training or run inference, minimum needed is both model state dict and optimizer state dict
		#save checkpoints has input (dict, checkpoint directory)
		saveCheckpoint({'epoch': epoch,
		'model_state_dict': model.state_dict(),
		'optimizer_state_dict': optimizer.state_dict(),
		'acc': val_avg_epoch_acc,
		'loss': val_avg_epoch_loss,
		}, is_best, args.checkpoint_directory)

def train(train_loader, model, criterion, optimizer, epoch, args):

	#losses = AverageMeter()

	model.train()

	#epoch_loss = 0.0

	#alternatively use train_iter = iter(trainloader) 

	#images, masks = next(train_iter)

	num_batches = len(train_loader)

	for batch_idx, (images, masks) in enumerate(train_loader):

		#losses = AverageMeter('Loss', ':.4e')
		#pixel_acc = AverageMeter('Acc@1', ':6.2f')

		images, masks = images.to(device), masks.to(device)

		#forward->backpropation->loss

		#bsize, classes, h, w
		logits = model(images)

		#loss of entire minibatch divided by N batch elements
		loss = criterion(logits, masks)
		optimizer.zero_grad()
		loss.backward()

		#update model parameters
		optimizer.step()

		batch_loss = loss.detach().item()

		#epoch_loss += loss.detach().item()#*images.shape[0]


		if batch_idx %50 == 0:
			avg_batch_acc = pixelAccuracy(logits.detach(), masks.detach())
			print('Epoch: [{}/{}] Batch: [{}/{}] Loss: [{:.4e}] Acc: [{:.3f}]'.format(epoch, args.epochs, batch_idx, num_batches, batch_loss, avg_batch_acc))

	return batch_loss

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

class AverageMeter(object):
	"""Computes and stores the average and current value"""
	def __init__(self, name, fmt=':f'):
		self.name = name
		self.fmt = fmt
		self.reset()

	def reset(self):
		self.val = 0
		self.avg = 0
		self.sum = 0
		self.count = 0

	def update(self, val, n=1):
		self.val = val
		self.sum += val * n
		self.count += n
		self.avg = self.sum / self.count

	def __str__(self):
		fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
		return fmtstr.format(**self.__dict__)

def saveCheckpoint(state, is_best, checkpoint_dir=args.checkpoint_directory):

	save_dir = os.path.join('results', checkpoint_dir)
	file_name = 'checkpoint.pth'#'checkpoint.{}.ckpt'.format(epoch)
	path = os.path.join(save_dir, file_name)

	torch.save(state, path)

	#if the current model has best validation acc/loss until now
	#then copy checkpoint model to a separate best model
	#(path to current file, path to best copy)
	if is_best:
		shutil.copyfile(path, os.path.join(save_dir, 'best_model.pth'))

main()

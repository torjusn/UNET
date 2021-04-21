#built-in
import os
import os.path
import glob

#misc
import numpy as np
from PIL import Image

#torch
import torchvision
import torch
import torchvision.transforms as transforms

#local
import dataset.custom_datasets as custom

class ToTensor(object):
	"""Transform the image to tensor.
	"""
	def __call__(self, x):
		x = torch.from_numpy(x)
		return x

def get_houston(root):

	norm = np.load(os.path.join(root,'workdir/houston/houston_norm.npy'))

	image_transform = transforms.Compose([
		ToTensor(),
		transforms.Normalize(norm[:, 0], norm[:, 1])])

	mask_transform = transforms.Compose([
		ToTensor()])

	test_image_transform = transforms.Compose([
		ToTensor(),
		transforms.Normalize(norm[:, 0], norm[:, 1])])

	test_mask_transform = transforms.Compose([
		ToTensor()])

	def npy_loader(path):

		#do once either here or in ToTensor
		#sample = torch.from_numpy(np.load(path))

		image = np.load(path)
		return image

	#root is start of datafolder e.g. "../data-local", already abspath
	data_dir = os.path.join(root, 'images/houston/by-image/data')
	mask_dir = os.path.join(root, 'images/houston/by-image/mask')


	#ONLY CREATES ONE COMBINED SET FROM DATA AND MASK DIRS
	###-------------------------------------------------------------------------
	#houston_dataset = SegmentationDataset(root_image=data_dir, root_mask=mask_dir, loader=npy_loader, image_transform=image_transform, mask_transform=mask_transform)


	#TAKES IN LIST OF PATHS AND SPLITS INTO TRAIN + VAL
	###-------------------------------------------------------------------------

	#alternative to find paths inside Dataset from https://discuss.pytorch.org/t/how-make-customised-dataset-for-semantic-segmentation/30881/4
	#glob with wildcard* gets all files with extension
	image_paths = glob.glob(os.path.join(data_dir, '*.npy'))
	mask_paths = glob.glob(os.path.join(mask_dir, '*.npy'))

	train_size = 0.8
	num_paths = len(image_paths)
	num_train = int(num_paths*train_size)

	idxs = np.arange(num_paths) 
	np.random.shuffle(idxs)

	train_idxs = idxs[:num_train]
	test_idxs = idxs[num_train:]

	#split into train + val
	train_image_paths = list(np.array(image_paths)[train_idxs])
	train_mask_paths = list(np.array(mask_paths)[train_idxs])

	test_image_paths = list(np.array(image_paths)[test_idxs])
	test_mask_paths = list(np.array(mask_paths)[test_idxs])

	train_dataset = custom.SegmentationDatasetPath(image_paths=train_image_paths, mask_paths=train_mask_paths, loader=npy_loader, image_transform=train_image_transform, mask_transform=train_mask_transform)
	val_dataset = custom.SegmentationDatasetPath(image_paths=test_image_paths, mask_paths=test_mask_paths, loader=npy_loader, image_transform=test_image_transform, mask_transform=test_mask_transform)

	print('Train: {} Val: {}'.format(len(train_dataset), len(val_dataset)))

	return train_dataset, val_dataset#trainval test

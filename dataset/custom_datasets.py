import os
from typing import Any, Callable, cast, Dict, List, Optional, Tuple

from torch.utils.data import Dataset

'''
CUSTOM PYTORCH SEGMENTATION DATASETS

SegmentationDatasetPath:
Takes as input a list of image and mask paths.
Used e.g. when wanting to split a dataset into smaller sets by 
calling once for Train and once for Test.

SegmentationDataset:
Takes as input directories to image and mask folders and creates only one dataset.
Useful for methods not based on training models, e.g. MBO.
'''
###-------------------------------------------------------------------------

class SegmentationDatasetPath(Dataset):

	'''
	Your custom dataset should inherit Dataset and override the following methods:

		__len__ so that len(dataset) returns the size of the dataset.
		__getitem__ to support the indexing such that dataset[i] can be used to get i'th sample
	'''

	#store just image paths in init

	def __init__(self, image_paths: list, mask_paths: list, loader: Callable[[str], Any], image_transform: Optional[Callable] = None, mask_transform: Optional[Callable] = None, is_valid_file: Optional[Callable[[str], bool]] = None) -> None:
		super(SegmentationDatasetPath, self).__init__()

		self.loader = loader

		self.image_paths = image_paths
		self.mask_paths = mask_paths

		self.image_transform = image_transform
		self.mask_transform = mask_transform

	def __getitem__(self, index: int) -> Tuple[Any, Any]:
		"""
		Args:
			index (int): Index

		Returns:
			tuple: (image, mask)
		"""
		image_path = self.image_paths[index]
		mask_path = self.mask_paths[index]

		image = self.loader(image_path)
		mask = self.loader(mask_path)

		if self.image_transform is not None:
			image = self.image_transform(image)
		if self.mask_transform is not None:
			mask = self.mask_transform(mask)

		return image, mask.long()

	def __len__(self) -> int:
		return len(self.image_paths)


def make_dataset(image_directory: str, mask_directory: str,): #-> List[Tuple[str, int]]:
	image_paths = []
	mask_paths = []
	image_directory = os.path.expanduser(image_directory)
	mask_directory = os.path.expanduser(mask_directory)

	for root, _, fnames in sorted(os.walk(image_directory, followlinks=True)):
		for fname in sorted(fnames):
			path = os.path.join(root, fname)
			#if is_valid_file(path):
			image_paths.append(path)

	for root, _, fnames in sorted(os.walk(mask_directory, followlinks=True)):
		for fname in sorted(fnames):
			path = os.path.join(root, fname)
			#if is_valid_file(path):
			mask_paths.append(path)

	return image_paths, mask_paths


class SegmentationDataset(Dataset):

	'''
	Your custom dataset should inherit Dataset and override the following methods:

		__len__ so that len(dataset) returns the size of the dataset.
		__getitem__ to support the indexing such that dataset[i] can be used to get i'th sample
	'''

	#store just image paths in init

	def __init__(self, root_image: str, root_mask: str, loader: Callable[[str], Any], image_transform: Optional[Callable] = None, mask_transform: Optional[Callable] = None, is_valid_file: Optional[Callable[[str], bool]] = None) -> None:
		super(SegmentationDataset, self).__init__()

		image_paths, mask_paths = make_dataset(root_image, root_mask)

		self.loader = loader

		self.image_paths = image_paths
		self.mask_paths = mask_paths

		self.image_transform = image_transform
		self.mask_transform = mask_transform

	def __getitem__(self, index: int) -> Tuple[Any, Any]:
		"""
		Args:
			index (int): Index

		Returns:
			tuple: (image, mask)
		"""
		image_path = self.image_paths[index]
		mask_path = self.mask_paths[index]

		image = self.loader(image_path)
		mask = self.loader(mask_path)

		if self.image_transform is not None:
			image = self.image_transform(image)
		if self.mask_transform is not None:
			mask = self.mask_transform(mask)

		return image, mask.long()

	def __len__(self) -> int:
		return len(self.image_paths)
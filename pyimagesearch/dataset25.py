# import the necessary packages
from torch.utils.data import Dataset
import cv2
import torch
import os
import numpy as np
class SegmentationDataset(Dataset):
	def __init__(self, imagePaths, maskPaths, transforms):
		# store the image and mask filepaths, and augmentation
		# transforms
		self.imagePaths = imagePaths
		self.maskPaths = maskPaths
		self.transforms = transforms
	def __len__(self):
		# return the number of total samples contained in the dataset
		return len(self.imagePaths)
	def __getitem__(self, idx):
		# grab the image path from the current index
		imagePath = self.imagePaths[idx]
		imagePath_split = imagePath.split('/')
		current_case = imagePath_split[-2]
		current_slice = imagePath_split[-1]
		current_slice = current_slice.split('.')
		current_slice = current_slice[0]
		# load the image from disk, swap its channels from BGR to RGB,
		# and read the associated mask from disk in grayscale mode
		dir_25 = '/nfs/kitbag/data1/jtoledo/tbi_project/image_data_25_PBIb/'
		images_25 = os.listdir(dir_25+current_case)
		images_25.sort()
		stack_images = []
		for i in images_25:
			if current_slice in i:
				image = cv2.imread(dir_25+current_case+'/'+i,cv2.IMREAD_GRAYSCALE)
				
				if self.transforms is not None:
					# apply the transformations to image
					image = self.transforms(image)
				stack_images.append(image)
				
		stack_slices = torch.stack(stack_images,dim=1).squeeze()
				
		


		#image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
		mask = cv2.imread(self.maskPaths[idx], 0)
		# check to see if we are applying any transformations
		if self.transforms is not None:
			# apply the transformations to both image and its mask
			
			mask = self.transforms(mask)
		# return a tuple of the image and its mask
		return (stack_slices, mask)
		
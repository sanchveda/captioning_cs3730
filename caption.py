import torch
from torch.utils.data import Dataset
import h5py
import json
import os
import pdb 

from torch.nn.utils.rnn import pad_sequence
import pandas as pd 
class CapData(Dataset):
	'''
	Dataset class to be used in DataLoader class for creating batches.
	'''

	def __init__(self, input_dir, split, transform=None):
		# get split/section of data
		assert split in {'tr', 'vl', 'ts'}
		self.split = split

		# load file data and attribute
		self.f = h5py.File(os.path.join(input_dir, self.split + '_imgs.hdf5'), 'r')
		self.cpi = self.f.attrs['captions_per_image']
		self.imgs = self.f['images']

		
		# obtain captions and caption lengths
		with open(os.path.join(input_dir, self.split + '_enccaps.json'), 'r') as f:
			self.caps = json.load(f)
		with open(os.path.join(input_dir, self.split + '_lencaps.json'), 'r') as f:
			self.len_caps = json.load(f)

		self.size = len(self.caps)
		self.transform = transform
	
	def __getitem__(self, cap_i):
		"""
		Get the corresponding image and caption with given index
		"""

		#img_i = cap_i // self.cpi  # each image corresponds to cpi of captions
		
		img = torch.FloatTensor(self.imgs[cap_i] / 255.)
		cap = torch.LongTensor(self.caps[cap_i])
		cap_l = self.len_caps[cap_i]

		if self.transform:
			img = self.transform(img)

		if self.split in ['vl', 'tr', 'ts']: #Should be tr 
 			return img, cap, cap_l
	

	def collate_fn(self, data):
		
		if len (data)> 1:
			img= []
			caps = []
			len_caps =[]
			
			for dat in data : 
			
				img.append( dat[0] )
				caps.append(dat[1] )
				len_caps.append (dat[2])
				
				
			img= torch.stack (img)
			caps = pad_sequence(caps,True)
			len_caps = torch.LongTensor(len_caps)
			
			return img, caps, len_caps 
		else:
			
			return data[0][0].unsqueeze(0), data[0][1].unsqueeze(0), torch.tensor(data[0][2],dtype=torch.long).unsqueeze(0)
	def __len__(self):
		return self.size

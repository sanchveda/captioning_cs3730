import json
import os
import h5py
import argparse
import numpy as np
from collections import Counter
from random import choice, sample
import jsonlines 
from imageio import imread
#from scipy.misc import imresize
import cv2 
import pdb 
# I set these default values for my convenience, feel free to change them
ap = argparse.ArgumentParser()
ap.add_argument('-j', '--json', default='/run/user/1435515045/gvfs/smb-share:server=terracotta.psychology.pitt.edu,share=projects/Sanchayan/flickr_localized_narrative/', help="path to json file")
ap.add_argument('-i', '--img_dir', default='/run/user/1435515045/gvfs/smb-share:server=terracotta.psychology.pitt.edu,share=projects/Sanchayan/flickr30k_images/', help="directory to images")
ap.add_argument('-o', '--out_dir', default='/run/user/1435515045/gvfs/smb-share:server=terracotta.psychology.pitt.edu,share=projects/Sanchayan/flickr_localized_narrative/test_input', help='directory to store files')
ap.add_argument('-mf', '--min_freq', default=5, help="minimum word frequency")
ap.add_argument('-l', '--max_len', default=500, help="max length of caption")
ap.add_argument('-cpi', '--captions_per_image', default=1, help="captions per image of dataset")
args = vars(ap.parse_args())

json_path = args['json']
img_dir = args['img_dir']
out_dir = args['out_dir']
min_freq = args['min_freq']
max_len = args['max_len']
cpi = args['captions_per_image']


#-------------This will remove all punctuuations -----------#
def preprocess_text1(x):
	try:
		for punct in '"!&?.,}-/<>#$%\()*+:;=?@[\\]^_`|\~':
			
			x = x.replace(punct, ' ')
			
		x= x.split()

		x = ' '.join(x)
		x = x.lower()

	except:
		raise ValueError
	return x
def read_data (name):


	image_id =[]
	captions =[]

	with open (name) as f:
		for line in f :

			data = json.loads(line)
			caption = preprocess_text1 (data['caption'])
			try:
				print (caption)
			except:
				continue
			image_id.append (data['image_id'])
			captions.append(caption)
	return image_id, captions


train = [ x for x in os.listdir (json_path) if 'train' in x and 'captions' in x][0]
valid = [ x for x in os.listdir (json_path) if 'val' in x and 'captions' in x][0]
test = [ x for x in os.listdir (json_path) if 'test' in x and 'captions' in x][0]



#------------Getting the whole data ---------------------#
train_img = []
train_cap = []
val_img = []
val_cap = []
test_img = []
test_cap = []

train_img, train_cap = read_data (os.path.join(json_path,train))
val_img, val_cap = read_data (os.path.join(json_path,valid))
test_img, test_cap = read_data(os.path.join(json_path,test))


#------------Full path of the image names -----------------#
train_img = [ img_dir + x + '.jpg' for x in train_img]  #Assuming imagegs are in .jpg formatt otherwise chaneg the extension 
val_img = [ img_dir + x + '.jpg' for x in val_img]
test_img = [ img_dir + x + '.jpg' for x in test_img]


#-----------------The following section creads the mapping dictionary with the words appearing at least 5 times. This is to reduce the vocabulary size
freq = dict ()
for cap in train_cap+val_cap+ test_cap:
	for words in cap.split(' '):
		word = words.strip(" ")
		if word not in freq:
			freq[word]= 0
		freq[word] +=1



print(len(train_img)) # 29000
print(len(val_img))   # 1014
print(len(test_img))  # 1000

# create word map
vocab = [w for w in freq.keys() if freq[w] > min_freq] # filter out lesser used words
word_map = {k: v+1 for v, k in enumerate(vocab)}  # word2idx from 1
word_map['<pad>'] = 0    #Zero-Padding
word_map['<start>'] = len(word_map) + 1   #START TOKEN 
word_map['<end>'] = len(word_map) + 1    #END TOKEN
word_map['<unk>'] = len(word_map) + 1  #UNIDENTIFIED TOKEN


# store word map
with open(os.path.join(out_dir, 'wordmap.json'), 'w') as f:
	json.dump(word_map, f)

# store values


#-----This portion is similar to the existing code. It will store the images into hdf5 forrmat so that it is easy while trainin. 
# 'train' will take sometime 
for paths, caps, split in [(train_img, train_cap, 'train'),
						  (val_img, val_cap, 'val'),
						  (test_img, test_cap, 'test')]:

	with h5py.File(os.path.join(out_dir, split + '_imgs.hdf5'), 'a') as f:
		f.attrs['captions_per_image'] = cpi
		images = f.create_dataset('images', (len(paths), 3, 256, 256), dtype='uint8')

		# encoded captions and length of captions
		enc_caps = []
		len_caps = []
		

		for i, path in enumerate(paths):

			'''
			"This section should be uncommmented if we consider individual sentences as separate captions"
			# randomly sample 5 captions
			if len(caps[i]) >= cpi:
				cap_selected = sample(caps[i], k=cpi)
			else:
				cap_selected = caps[i] + [choice(caps[i]) for _ in range(cpi-len(caps[i]))]
			'''
			cap_selected = caps[i] # Only if we consider the entire caption as one 
			
			# process image (take care of B&W) to 3x256x256 then store in h5py to not crash RAM
			img = cv2.imread(paths[i])
			if len(img.shape) == 2:
				img = img[:,:,np.newaxis]
				img = np.concatenate([img, img, img], axis=2)
			img = cv2.resize(img, (256, 256))
			img = img.transpose(2, 0, 1) # Store it in 3 x H X W
			images[i] = img
			

			for j, c in enumerate([cap_selected]):
				#Here if the size of the caption is more than 500, it will take that shape. Such casses are tackled in the data loader part. 
				# concat start, end, pads (encoded caption)
				enc_cap = [word_map['<start>']] + [word_map.get(w, word_map['<unk>']) for w in c.split(' ')] \
						  + [word_map['<end>']] + (max_len - len(c)) * [word_map['<pad>']]
				
				enc_caps.append(enc_cap)
				len_caps.append(len(c.split(' '))+2) # ignore pads

		# sanity check.. This has to be the same.
		assert images.shape[0] * cpi == len(enc_caps) == len(len_caps)

		# store captions and their lengths
		"Saving the captions"
		with open(os.path.join(out_dir, split + '_enccaps.json'), 'w') as f:
			json.dump(enc_caps, f)
		"Saving the length of each captions"
		with open(os.path.join(out_dir, split + '_lencaps.json'), 'w') as f:
			json.dump(len_caps, f)

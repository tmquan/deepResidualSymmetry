#!/usr/bin/env python

"""
We will read through the data and create 2 file
X_train.npy		num x dimy x dimx x 1 (channel)
y_train.npy		num x dimy x dimx x 1 (channel)

Where dimy and dimx are predefined dimy = 184, dimx = 184
if the images do not match that size, we will perform zero padding

6 is the number of classes:
homogeneous 		0
speckled			1
nucleolar			2
centromere			3
nuclear membrane 	4
golgi				5

"""

import pandas as pd # We will use dataframe in pandas to deal with csv
import numpy  as np # For multi-dimensional array
import cv2			# For reading image
import skimage.io  
import natsort 		# For natural sorting
import os
from Utility import *


trainDir =  "data/train/"
testDir  =  "data/test/"
##########################################################################
def processSubdirectory(dataDir):
	# Read the images and concatenate
	images = []
	

	imageDir = dataDir+'images/'
	for dirName, subdirList, fileList in os.walk(imageDir):
		# Sort the tif file numerically
		fileList = natsort.natsorted(fileList) 

		for f in fileList:
			filename = os.path.join(imageDir, f)
			image = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
			# image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
			print filename
			# Append to the images
			images.append(image)

	labels = []
	labelDir = dataDir+'labels/'
	for dirName, subdirList, fileList in os.walk(labelDir):
		# Sort the tif file numerically
		fileList = natsort.natsorted(fileList) 

		for f in fileList:
			filename = os.path.join(labelDir, f)
			label = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
			# label = cv2.cvtColor(label, cv2.COLOR_BGR2GRAY)
			# label = cv2.Canny(label,100,200)
			# print "Canny"
			# print filename
			# Append to the labels
			labels.append(label)

	# Convert images list to numpy array
	X = np.array(images)
	y = np.array(labels)
	y = y/255.0
	print y.max()
	X = np.expand_dims(X, axis=1)
	y = np.expand_dims(y, axis=1)

	# y = np.concatenate((y, 1-y), axis=3) # BG/Forground: 0, 1
	y = np.concatenate((y, 1-y), axis=1) # Forground: 0, 1; background, 2, 0
	# 	# Get the current shape of images
	print X.shape
	print y.shape


	# 	np.save('X_train.npy', X_train)
	# 	np.save('y_train.npy', y_train)
	return X, y


if __name__ == '__main__':
	X_train, y_train = processSubdirectory(trainDir)
	X_test , y_test  = processSubdirectory(testDir)

	np.save('X_train.npy', X_train)
	np.save('y_train.npy', y_train)

	np.save('X_test.npy', X_test)
	np.save('y_test.npy', y_test)
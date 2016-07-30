#!/usr/bin/env python
"""
Created on Sun Feb 21 10:12:52 2016

@author: tmquan
"""

from Model 		import *
from Utility	import *

######################################################################################
def y_predict():
	X = np.load('X_test.npy')
	y = np.load('y_test.npy')
	
	print "X.shape", X.shape
	print "y.shape", y.shape
	
	
	X_test = X
	
	
	print "X_test.shape", X_test.shape
	
	# Load model
	iter = 0
	model 	= mx.model.FeedForward.load('model', iter, ctx=mx.gpu(0))
	
	
	
	# Perform y_prediction
	# batch_size = 1
	print('y_predicting on data...')
	y_pred  = model.predict(X_test, num_batch=None)
	

	print "y_pred.shape", y_pred.shape
	
	y_pred  = np.array(y_pred[:,0,:,:])
	y_pred  = np.squeeze(y_pred)
	
	skimage.io.imsave("y_pred.tif", y_pred)

if __name__ == '__main__':
	y_predict()

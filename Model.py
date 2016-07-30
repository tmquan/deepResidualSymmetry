import mxnet as mx
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 20 21:03:22 2016
@author: tmquan
Original implementation is from https://github.com/juliandewit/kaggle_ndsb2
Thanks to Julian de Wit
"""

from Utility import *



def convolution_module(net, kernel_size, pad_size, filter_count, stride=(1, 1), work_space=2048, batch_norm=True, down_pool=False, up_pool=False, act_type="relu", convolution=True):
	if up_pool:
		net = mx.symbol.Deconvolution(net, kernel=(2, 2), pad=(0, 0), stride=(2, 2), num_filter=filter_count, workspace = work_space)
		net = mx.symbol.BatchNorm(net)
		if act_type != "":
			net = mx.symbol.Activation(net, act_type=act_type)
	
	if convolution:
		net = mx.symbol.Convolution(data=net, kernel=kernel_size, stride=stride, pad=pad_size, num_filter=filter_count, workspace=work_space)
		net = mx.symbol.Convolution(data=net, kernel=kernel_size, stride=stride, pad=pad_size, num_filter=filter_count, workspace=work_space)
		
	
	if batch_norm:
		net = mx.symbol.BatchNorm(net)
	
	if act_type != "":
		net = mx.symbol.Activation(net, act_type=act_type)
	
	if down_pool:
		net = mx.symbol.Pooling(net, pool_type="max", kernel=(2, 2), stride=(2, 2))

	
	return net

def get_unet():
	# Setting hyper parameter
	kernel_size 	= (3, 3)
	pad_size 		= (1, 1) # For the same size of filtering
	filter_count 	= 64	 # Original unet use 64 and 2 layers of conv

	net 	= mx.symbol.Variable("data")
	# net 	= net-128
	# net 	= net/128
	net 	= net/255
	
	net		= convolution_module(net, kernel_size, pad_size, filter_count=filter_count*1, down_pool=True)
	pool1	= net
	net		= mx.symbol.Dropout(net)
	
	net		= convolution_module(net, kernel_size, pad_size, filter_count=filter_count*2, down_pool=True)
	pool2	= net
	net		= mx.symbol.Dropout(net)
	
	net		= convolution_module(net, kernel_size, pad_size, filter_count=filter_count*4, down_pool=True)
	pool3	= net
	net		= mx.symbol.Dropout(net)
	
	# net		= convolution_module(net, kernel_size, pad_size, filter_count=filter_count*8, down_pool=True)
	# pool4	= net
	# net		= mx.symbol.Dropout(net)
	
	# net		= convolution_module(net, kernel_size, pad_size, filter_count=filter_count*16, down_pool=True)
	# pool5	= net
	# net		= mx.symbol.Dropout(net)
	
	net		= convolution_module(net, kernel_size, pad_size, filter_count=filter_count*8)
	
	# net = convolution_module(net, kernel_size, pad_size, filter_count=filter_count * 4, up_pool=True)
	# net = convolution_module(net, kernel_size, pad_size, filter_count=filter_count * 4, up_pool=True)
	
	# net		= mx.symbol.Dropout(net)
	# net		= mx.symbol.Concat(*[pool5, net])
	# net		= convolution_module(net, kernel_size, pad_size, filter_count=filter_count*16)
	# net		= convolution_module(net, kernel_size, pad_size, filter_count=filter_count*16, up_pool=True)
	
	# net		= mx.symbol.Dropout(net)	
	# net		= mx.symbol.Concat(*[pool4, net])
	# net		= convolution_module(net, kernel_size, pad_size, filter_count=filter_count*8)
	# net		= convolution_module(net, kernel_size, pad_size, filter_count=filter_count*8, up_pool=True)

	net		= mx.symbol.Dropout(net)
	net		= mx.symbol.Concat(*[pool3, net])
	net		= convolution_module(net, kernel_size, pad_size, filter_count=filter_count*4)
	net		= convolution_module(net, kernel_size, pad_size, filter_count=filter_count*4, up_pool=True)
	
	net		= mx.symbol.Dropout(net)	
	net		= mx.symbol.Concat(*[pool2, net])
	net		= convolution_module(net, kernel_size, pad_size, filter_count=filter_count*2)
	net		= convolution_module(net, kernel_size, pad_size, filter_count=filter_count*2, up_pool=True)
	
	net		= mx.symbol.Dropout(net)
	net		= mx.symbol.Concat(*[pool1, net])
	net		= convolution_module(net, kernel_size, pad_size, filter_count=filter_count*1)
	net		= convolution_module(net, kernel_size, pad_size, filter_count=filter_count*1, up_pool=True)
	
	net		= mx.symbol.Dropout(net)	
	net		= convolution_module(net, kernel_size, pad_size, filter_count=2, batch_norm=False, act_type="")
	
	# net = mx.symbolbol.Flatten(net)
	net 	= mx.symbol.LogisticRegressionOutput(data=net, name='softmax')
	return net


	
if __name__ == '__main__':
	# Draw the net
	data 	= mx.symbol.Variable('data')
	network = get_unet()
	dot = mx.viz.plot_network(network,
		None,
		shape={"data" : (30, 1, 512, 512)}
		) 
	dot.graph_attr['rankdir'] = 'RL'
	
	
	
	

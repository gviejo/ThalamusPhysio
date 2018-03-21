#!/usr/bin/env python

'''
	File name: main_make_map.py
	Author: Guillaume Viejo
	Date created: 28/09/2017    
	Python Version: 3.5.2

To make shank mapping

'''

import numpy as np
import pandas as pd
# from matplotlib.pyplot import plot,show,draw
import scipy.io
from functions import *
from pylab import *
from sklearn.decomposition import PCA
import _pickle as cPickle
import sys
from scipy.ndimage import gaussian_filter	
import os

###############################################################################################################
# PARAMETERS
###############################################################################################################

mouses 					= ['Mouse12', 'Mouse17', 'Mouse20', 'Mouse32']

nbins 					= 200
binsize					= 5
times 					= np.arange(0, binsize*(nbins+1), binsize) - (nbins*binsize)/2
times2 					= times
space 					= 0.01
interval_to_cut 		= {	'Mouse12':[88,120],
							'Mouse17':[84,123]}
							# 'Mouse20':[92,131],
							# 'Mouse32':[80,125]}

figure(figsize = (10, 100))
index = (times>-60)*(times<100)
timesindex = np.where(index)[0]
index = np.arange(0, len(times), 2)
indexindex = np.arange(0, len(times), 2)
timestoplot = times[index]

###############################################################################################################
# LOADING DATA
###############################################################################################################
for m in mouses:
	data 		= cPickle.load(open("../data/maps/"+m+".pickle", 'rb'))
	headdir 	= data['headdir']
	x 			= data['x']
	y 			= data['y']
	total 		= data['total']
	swr 		= data['movies']['swr']
	theta		= data['movies']['theta']

###############################################################################################################
# INTERPOLATE DATA
###############################################################################################################
	# total neuron	
	total = total / total.max()
	xnew, ynew, xytotal = interpolate(total.copy(), x, y, space)
	filtotal = gaussian_filter(xytotal, (10, 10))
	newtotal = softmax(filtotal, 10.0, 0.0)

	# head direction
	xnew, ynew, newheaddir = interpolate(headdir.copy(), x, y, space)
	newheaddir[newheaddir < np.percentile(newheaddir, 80)] = 0.0
	
	# swr
	newswr = []
	for t in range(len(times)):	
		xnew, ynew, frame = interpolate(swr[:,:,t].copy(), x, y, space)
		# frame = gaussian_filter(frame, (10, 10))
		newswr.append(frame)
	newswr = np.array(newswr)
	newswr = gaussian_filter(newswr, (10,10,10))
	newswr = newswr - newswr.min()
	newswr = newswr / newswr.max()	

	thl_lines = None
	# thalamus lines and shanks position to creat the mapping session nucleus
	if m+"_thalamus_lines.png" in os.listdir("../figures/mapping_to_align"):
		thl_lines = scipy.ndimage.imread("../figures/mapping_to_align/"+m+"_thalamus_lines.png").sum(2)
		xlines, ylines, thl_lines = interpolate(thl_lines, 	np.linspace(x.min(), x.max(), thl_lines.shape[1]),
															np.linspace(y.min(), y.max(), thl_lines.shape[0]), space*0.1)
		thl_lines -= thl_lines.min()
		thl_lines /= thl_lines.max()
		thl_lines[thl_lines<0.6] = np.NaN
	


	for i, (idx, t) in enumerate(zip(timesindex, timestoplot)):
		print(m, 4*i+mouses.index(m)+1)
		subplot(len(timestoplot),4,4*i+mouses.index(m)+1)
		frame = newswr[idx]
		rgbframe = get_rgb(frame.copy(), np.ones_like(newtotal), newtotal.copy(), 0.65)		
		# imshow(rgbframe, aspect = 'equal', vmin = newswr.min(), vmax = newswr.max(), extent = (xnew[0], xnew[-1], ynew[-1], ynew[0]))
		# imshow(frame, aspect = 'equal', vmin = newswr.min(), vmax = newswr.max(), extent = (xnew[0], xnew[-1], ynew[-1], ynew[0]), cmap = 'jet')
		imshow(rgbframe, aspect = 'equal', extent = (xnew[0], xnew[-1], ynew[-1], ynew[0]))
		contour(newheaddir, extent = (xnew[0], xnew[-1], ynew[0], ynew[-1]), cmap = 'winter')
		gca().set_aspect('equal')
		gca().set_xticks(np.arange(xnew[0], xnew[-1], 0.2))
		gca().set_yticks(np.arange(ynew[0], ynew[-1], 0.2))
		title("T = "+str(int(t))+" ms")		
		if thl_lines is not None:			
			contour(thl_lines, extent = (xnew[0], xnew[-1], ynew[0], ynew[-1]), cmap = 'winter')


subplots_adjust(hspace =  0.0, top = 1.0, bottom = 0.0)
savefig("../figures/mapping_to_align/all_swr_frames.pdf")
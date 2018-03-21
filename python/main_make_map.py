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


###############################################################################################################
# LOADING DATA
###############################################################################################################
for m in mouses:
# for m in ['Mouse20']:
	data 		= cPickle.load(open("../data/maps/"+m+".pickle", 'rb'))
	headdir 	= data['headdir']
	x 			= data['x']
	y 			= data['y']
	total 		= data['total']
	swr 		= data['movies']['swr']
	theta		= data['movies']['theta']
	theta_dens 	= data['theta_dens']
###############################################################################################################
# INTERPOLATE DATA
###############################################################################################################
	# total neuron	
	total = total / total.max()
	xnew, ynew, xytotal = interpolate(total.copy(), x, y, space)
	filtotal = gaussian_filter(xytotal, (10, 10))
	newtotal = softmax(filtotal, 10.0, 0.1)

	# head direction
	xnew, ynew, newheaddir = interpolate(headdir.copy(), x, y, space)
	newheaddir[newheaddir < np.percentile(newheaddir, 80)] = 0.0
	
	# theta dens
	xnew, ynew, newthetadens = interpolate(theta_dens.copy(), x, y, space)	

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

	# theta
	phase = np.linspace(0, 2*np.pi, theta.shape[-1])
	newtheta = []
	for i in range(len(phase)):
		xnew, ynew, frame = interpolate(theta[:,:,i].copy(), x, y, space)
		newtheta.append(frame)
	newtheta = np.array(newtheta)
	newtheta = gaussian_filter(newtheta, (0, 0.2, 0.2))
	newtheta = newtheta - newtheta.min()
	newtheta = newtheta / newtheta.max()

	thl_lines = None
	# # thalamus lines and shanks position to creat the mapping session nucleus
	if m+"_thalamus_lines.png" in os.listdir("../figures/mapping_to_align"):
		thl_lines = scipy.ndimage.imread("../figures/mapping_to_align/"+m+"_thalamus_lines.png").sum(2)
		xlines, ylines, thl_lines = interpolate(thl_lines, 	np.linspace(x.min(), x.max(), thl_lines.shape[1]),
	 														np.linspace(y.min(), y.max(), thl_lines.shape[0]), space*0.1)

		thl_lines -= thl_lines.min()
		thl_lines /= thl_lines.max()
		thl_lines[thl_lines<0.6] = np.NaN	
		figure()	
		imshow(thl_lines, extent = (xlines[0], xlines[-1], ylines[-1], ylines[0]))
		xx, yy = np.meshgrid(x, y)
		scatter(xx.flatten(), yy.flatten())
		xticks(x, np.arange(len(x))[::-1])
		yticks(y, np.arange(len(y)))
		savefig("../figures/mapping_to_align/"+m+"_shanks_postion.pdf")
	
	mse = np.abs(np.median(newswr, 0) - newswr).sum(1).sum(1)		
	idx_frames = np.arange(0, len(newswr), 10)#[np.argmax(mse)-15:np.argmax(mse)+15]

	figure(figsize=(20,100))
	col = 4
	row = int(np.ceil(len(idx_frames)/col))
	for i, fr in enumerate(idx_frames):
		subplot(row, col, i+1)
		frame = newswr[fr]
		rgbframe = get_rgb(frame.copy(), np.ones_like(newtotal), newtotal.copy(), 0.65)		
		imshow(rgbframe, aspect = 'equal', extent = (xnew[0], xnew[-1], ynew[-1], ynew[0]))
		# imshow(frame, aspect = 'equal', vmin = newswr.min(), vmax = newswr.max(), extent = (xnew[0], xnew[-1], ynew[-1], ynew[0]), cmap = 'jet')
		if thl_lines is not None:
			imshow(thl_lines, aspect = 'equal', extent = (xnew[0], xnew[-1], ynew[-1], ynew[0]))
		contour(newheaddir, extent = (xnew[0], xnew[-1], ynew[0], ynew[-1]), cmap = 'winter')
		gca().set_aspect('equal')
		gca().set_xticks(np.arange(xnew[0], xnew[-1], 0.2))
		gca().set_yticks(np.arange(ynew[0], ynew[-1], 0.2))
		title("T = "+str(int(times[fr]))+" ms")

	savefig("../figures/mapping_to_align/"+m+"_swr_frames.pdf")
		
	
	idx_frames = np.arange(len(newtheta))
	figure(figsize = (20,100))
	col = 4
	row = int(np.ceil(len(idx_frames)/col))
	for i, fr in enumerate(idx_frames):
		subplot(row, col, i+1)
		frame = newtheta[fr]
		rgbframe = get_rgb(frame.copy(), np.ones_like(newtotal), newtotal.copy(), 0.65)		
		imshow(rgbframe, aspect = 'equal', extent = (xnew[0], xnew[-1], ynew[-1], ynew[0]))
		contour(newheaddir, extent = (xnew[0], xnew[-1], ynew[0], ynew[-1]), cmap = 'winter')
		if thl_lines is not None:
			imshow(thl_lines, aspect = 'equal', extent = (xnew[0], xnew[-1], ynew[-1], ynew[0]))
		gca().set_aspect('equal')
		gca().set_xticks(np.arange(xnew[0], xnew[-1], 0.2))
		gca().set_yticks(np.arange(ynew[0], ynew[-1], 0.2))			
		
		title("phi = "+str(int(phase[fr])))

	savefig("../figures/mapping_to_align/"+m+"_theta_frames.pdf")

	figure()
	imshow(newthetadens, extent = (xnew[0], xnew[-1], ynew[-1], ynew[0]), cmap = 'jet')
	contour(newheaddir, extent = (xnew[0], xnew[-1], ynew[-1], ynew[0]), origin = 'upper', cmap = 'winter')
	if thl_lines is not None:
		imshow(thl_lines, aspect = 'equal', extent = (xnew[0], xnew[-1], ynew[-1], ynew[0]))	
	gca().set_aspect('equal')
	savefig("../figures/mapping_to_align/"+m+"_hd_density.pdf")
	

	from matplotlib import animation, rc
	from IPython.display import HTML, Image

	rc('animation', html='html5')
	fig, axes = plt.subplots(1,1)
	
	start = 0

	frame = newswr[start]
	rgbframe = get_rgb(frame.copy(), np.ones_like(newtotal), newtotal.copy(), 0.65)		
	images = [axes.imshow(get_rgb(newswr[0].copy(), np.ones_like(newtotal), newtotal.copy(), 0.65), vmin = newswr.min(), vmax = newswr.max(), aspect = 'equal', origin = 'upper', extent = (xnew[0], xnew[-1], ynew[-1], ynew[0]))]
	if thl_lines is not None:	
		axes.imshow(thl_lines, aspect = 'equal', origin = 'upper', extent = (xlines[0], xlines[-1], ylines[-1], ylines[0]))

	def init():
		images[0].set_data(get_rgb(frame.copy(), np.ones_like(newtotal), newtotal.copy(), 0.65))
		return images
			
	def animate(t):
		frame = newswr[t]
		rgbframe = get_rgb(frame.copy(), np.ones_like(newtotal), newtotal.copy(), 0.65)
		images[0].set_data(rgbframe)	
		images[0].axes.set_title("time = "+str(times[t]))
		return images
		
	anim = animation.FuncAnimation(fig, animate, init_func=init,
							   frames=range(start,len(newswr)), interval=10, blit=False, repeat_delay = 1000)

	anim.save('../figures/mapping_to_align/swr_mod_'+m+'.gif', writer='imagemagick', fps=15)
	


		


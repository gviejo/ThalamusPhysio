#!/usr/bin/env python

import sys
sys.path.append("../")
import numpy as np
import pandas as pd

import scipy.io
from functions import *

from sklearn.decomposition import PCA
import _pickle as cPickle
import neuroseries as nts
import sys
import scipy.ndimage.filters as filters
from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans
from functools import reduce
from multiprocessing import Pool
import h5py as hd
from scipy.stats import zscore
from sklearn.manifold import TSNE, SpectralEmbedding
from skimage import filters
import os
from scipy.misc import imread
from skimage.filters import gaussian


space = pd.read_hdf("../../figures/figures_articles/figure1/space.hdf5")

burst = pd.HDFStore("/mnt/DataGuillaume/MergedData/BURSTINESS.h5")['w']
burst = burst.loc[space.index]


# autocorr = pd.read_hdf("../../figures/figures_articles/figure1/autocorr.hdf5")
store_autocorr = pd.HDFStore("/mnt/DataGuillaume/MergedData/AUTOCORR_ALL.h5")

carte38_mouse17 = imread('../../figures/mapping_to_align/paxino/paxino_38_mouse17.png')
carte38_mouse17_2 = imread('../../figures/mapping_to_align/paxino/paxino_38_mouse17_2.png')
bound_map_38 = (-2336/1044, 2480/1044, 0, 2663/1044)
cut_bound_map = (-86/1044, 2480/1044, 0, 2663/1044)

tmp = cPickle.load(open("../../figures/figures_articles/figure1/shifts.pickle", 'rb'))
angles = tmp['angles']
shifts = tmp['shifts']


hd_index = space.index.values[space['hd'] == 1]

neurontoplot = [np.intersect1d(hd_index, space.index.values[space['cluster'] == 1])[0],
				burst.loc[space.index.values[space['cluster'] == 0]].sort_values('sws').index[3],
				burst.sort_values('sws').index.values[-20]]

# specific to mouse 17
subspace = pd.read_hdf("../../figures/figures_articles/figure1/subspace_Mouse17.hdf5")

data = cPickle.load(open("../../figures/figures_articles/figure1/rotated_images_Mouse17.pickle", 'rb'))
rotated_images = data['rotated_images']
new_xy_shank = data['new_xy_shank']
bound = data['bound']

data 		= cPickle.load(open("../../data/maps/Mouse17.pickle", 'rb'))
x 			= data['x']
y 			= data['y']*-1.0+np.max(data['y'])
headdir 	= data['headdir']

xy_pos = new_xy_shank.reshape(len(y), len(x), 2)


###############################################################################################################
###############################################################################################################
# PLOT
###############################################################################################################
def figsize(scale):
	fig_width_pt = 483.69687                         # Get this from LaTeX using \the\textwidth
	inches_per_pt = 1.0/72.27                       # Convert pt to inch
	golden_mean = (np.sqrt(5.0)-1.0)/2.0            # Aesthetic ratio (you could change this)
	fig_width = fig_width_pt*inches_per_pt*scale    # width in inches
	fig_height = fig_width*golden_mean*1.5          # height in inches
	fig_size = [fig_width,fig_height]
	return fig_size

def simpleaxis(ax):
	ax.spines['top'].set_visible(False)
	ax.spines['right'].set_visible(False)
	ax.get_xaxis().tick_bottom()
	ax.get_yaxis().tick_left()
	# ax.xaxis.set_tick_params(size=6)
	# ax.yaxis.set_tick_params(size=6)

def noaxis(ax):
	ax.spines['top'].set_visible(False)
	ax.spines['right'].set_visible(False)
	ax.spines['left'].set_visible(False)
	ax.spines['bottom'].set_visible(False)
	ax.get_xaxis().tick_bottom()
	ax.get_yaxis().tick_left()
	ax.set_xticks([])
	ax.set_yticks([])
	# ax.xaxis.set_tick_params(size=6)
	# ax.yaxis.set_tick_params(size=6)


import matplotlib as mpl
from mpl_toolkits.axes_grid1 import make_axes_locatable

# mpl.use("pdf")
pdf_with_latex = {                      # setup matplotlib to use latex for output
	"pgf.texsystem": "pdflatex",        # change this if using xetex or lautex
	# "text.usetex": True,                # use LaTeX to write all text
	# "font.family": "serif",
	"font.serif": [],                   # blank entries should cause plots to inherit fonts from the document
	"font.sans-serif": [],
	"font.monospace": [],
	"axes.labelsize": 6,               # LaTeX default is 10pt font.
	"font.size": 7,
	"legend.fontsize": 6,               # Make the legend/label fonts a little smaller
	"xtick.labelsize": 6,
	"ytick.labelsize": 6,
	"pgf.preamble": [
		r"\usepackage[utf8x]{inputenc}",    # use utf8 fonts becasue your computer can handle it :)
		r"\usepackage[T1]{fontenc}",        # plots will be generated using this preamble
		],
	"lines.markeredgewidth" : 0.2,
	"axes.linewidth"        : 0.5,
	"ytick.major.size"      : 1.0,
	"xtick.major.size"      : 1.0
	}    
mpl.rcParams.update(pdf_with_latex)
import matplotlib.gridspec as gridspec
from matplotlib.pyplot import *
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib.patches import Rectangle
from matplotlib.lines import Line2D

colors = ['red', 'green', 'blue', 'orange']
cmaps = ['Reds', 'Greens', 'Blues', 'Purples', 'Oranges']
markers = ['o', '^', '*', 's']

fig = figure(figsize = figsize(1.0), tight_layout=True)

outergs = gridspec.GridSpec(5,4, figure = fig)

#############################################
# A. TOTAL MATRIX NEURON COUNT MOUSE 17
#############################################
count_cmap = 'jet'
axA = fig.add_subplot(outergs[0,0])
simpleaxis(axA)
axA.text(-1.0, 0.95, "A", transform = axA.transAxes, fontsize = 9)
im = imshow(data['total'], aspect = 'equal', cmap = count_cmap)
ylabel("Session number")
xlabel("Shank number")

# arrow
cax = inset_axes(axA, "100%", "50%",
                   bbox_to_anchor=(1.4, 0.35, 1, 1),
                   bbox_transform=axA.transAxes, 
                   loc = 'lower left')
cax.arrow(0,0.2,0.8,0, linewidth = 4)
noaxis(cax)

cax.set_xticks([])
cax.set_yticks([])
# gaussian
cax = inset_axes(axA, "50%", "20%",
                   bbox_to_anchor=(1.6, 0.1, 1, 1),
                   bbox_transform=axA.transAxes, 
                   loc = 'lower left')
window = scipy.signal.gaussian(51, std=7)
simpleaxis(cax)
cax.plot(window)
cax.set_xticks([])
cax.set_yticks([])
cax.set_xlabel("2. Smoothing")
cax.set_title("1. Interpolation", fontsize = 6, pad = 15.0)



#############################################
# B. COUT NEURONS MOUSE 17 SMOOTHED
#############################################
axB = fig.add_subplot(outergs[0,1])
simpleaxis(axB)
axB.text(-0.9, 0.95, "B", transform = axB.transAxes, fontsize = 9)
xnew, ynew, tmp = interpolate(data['total'], data['x'], data['y'], 0.010)
total2 = gaussian(tmp, sigma = 15.0, mode = 'reflect')
imshow(total2, aspect = 'equal', extent = (x[0], x[-1], y[0], y[-1]), cmap = count_cmap)
xlabel("Shank position (mm)")
ylabel("Session position (mm)")
xl = ['' for _ in range(len(x))]
for i in np.arange(0 ,len(x), 2): xl[i] = str(np.round(data['x'][i], 2))
yl = ['' for _ in range(len(y))]
for i in np.arange(0 ,len(y), 2): yl[i] = str(np.round(data['y'][i], 3))
xticks(data['x'], xl)
yticks(data['y'], yl)

#############################################
# C. RECORDINGS SITE ALL MOUSE SQUARE
#############################################
axC = fig.add_subplot(outergs[0,2])
noaxis(axC)
axC.text(-0.2, 0.95, "C", transform = axC.transAxes, fontsize = 9)
sc = scatter(new_xy_shank[:,0], new_xy_shank[:,1], c = data['total'].flatten(), s = data['total'].flatten()*0.6, cmap = count_cmap)
imshow(carte38_mouse17_2[:,2250:], extent = cut_bound_map, interpolation = 'bilinear', aspect = 'equal')

#colorbar	
cax = inset_axes(axC, "50%", "5%",
                   bbox_to_anchor=(0.85, 0, 1, 1),
                   bbox_transform=axC.transAxes, 
                   loc = 'lower left')
cb = colorbar(sc, cax = cax, orientation = 'horizontal')
cb.set_label('Neuron count', labelpad = 0)
cb.ax.xaxis.set_tick_params(pad = 2)
cax.set_title("Mouse 1", fontsize = 9, pad = 2.5)


#############################################
# D. DENSITY NEURONS MOUSE 17 SMOOTHED 
#############################################
axD = fig.add_subplot(outergs[0,3])
noaxis(axD)
axD.text(-0.2, 0.95, "D", transform = axD.transAxes, fontsize = 9)
h, w = total2.shape
total3 = np.zeros((h*3, w*3))*np.nan
total3[h:h*2,w:w*2] = total2.copy() + 1.0
total3 = rotateImage(total3, -angles[1])
total3[total3 == 0.0] = np.nan
total3 -= 1.0
tocrop = np.where(~np.isnan(total3))
total3 = total3[tocrop[0].min()-1:tocrop[0].max()+1,tocrop[1].min()-1:tocrop[1].max()+1]
xlength, ylength = getXYshapeofRotatedMatrix(data['x'].max(), data['y'].max(), angles[1])
bound = (shifts[1][0],xlength+shifts[1][0],shifts[1][1],ylength+shifts[1][1])
imshow(total3, extent = bound, alpha = 0.8, aspect = 'equal', cmap = count_cmap)
imshow(carte38_mouse17_2[:,2250:], extent = cut_bound_map, interpolation = 'bilinear', aspect = 'equal')



#############################################
# E. SQUARE ALL NEURONS
#############################################
axE = fig.add_subplot(outergs[1,0:2])
noaxis(axE)
axE.text(-0.2, 0.95, "E", transform = axE.transAxes, fontsize = 9)
imshow(carte38_mouse17_2, extent = bound_map_38, interpolation = 'bilinear', aspect = 'equal')
leghandles = []
xbins = np.arange(-1, 2.1, 0.2)
ybins = np.arange(0.2, 3.0, 0.2)[::-1]
all_count = np.zeros((len(ybins), len(xbins)))
for i, m, l in zip([1,0,2,3], ['Mouse17', 'Mouse12', 'Mouse20', 'Mouse32'], [1,2,3,4]):
	data = cPickle.load(open("../../figures/figures_articles/figure1/rotated_images_"+m+".pickle", 'rb'))
	new_xy_shank = data['new_xy_shank']
	xidx = np.digitize(new_xy_shank[:,0], xbins)
	yidx = np.digitize(new_xy_shank[:,1], ybins)	
	data = cPickle.load(open("../../data/maps/"+m+".pickle", 'rb'))
	xx, yy = np.meshgrid(np.arange(len(data['x'])), np.arange(len(data['y'])))
	for j, x, y in zip(np.arange(len(xidx)), xidx, yidx):
		all_count[y,x] += data['total'][yy.flatten()[j],xx.flatten()[j]]
	xx = new_xy_shank[:,0].reshape(len(data['y']), len(data['x']))
	yy = new_xy_shank[:,1].reshape(len(data['y']), len(data['x']))
	lower_left = (xx[-1,0], yy[-1,0])
	rect = Rectangle(lower_left, data['x'].max(), data['y'].max(), -angles[i], fill = False, edgecolor = colors[i])
	axE.add_patch(rect)
	leghandles.append(Line2D([], [], color = colors[i], marker = '', label = 'Mouse '+str(l)))

legend(handles = leghandles, loc = 'lower left', bbox_to_anchor=(-0.2, -0.1))
ylim(0, 2.7)

#############################################
# F. DENSITY NEURONS ALL MOUSE HISTOFRAM
#############################################
axF = fig.add_subplot(outergs[1,2:])
noaxis(axF)
axF.text(-0.2, 0.95, "F", transform = axF.transAxes, fontsize = 9)
# all_count[all_count <= 0.0] = np.nan
im = imshow(all_count, extent = (xbins[0], xbins[-1], ybins[-1], ybins[0]), aspect = 'equal', alpha = 0.8, cmap = count_cmap, interpolation = 'bilinear')
imshow(carte38_mouse17_2, extent = bound_map_38, interpolation = 'bilinear', aspect = 'equal')
xlim(-2.2, 2.2)
ylim(0, 2.7)
#colorbar	
cax = inset_axes(axF, "50%", "5%",
                   bbox_to_anchor=(-0.5, 0.1, 1, 1),
                   bbox_transform=axF.transAxes, 
                   loc = 'lower left')
cb = colorbar(im, cax = cax, orientation = 'horizontal')
cb.set_label('Neuron count', labelpad = 0)
cb.ax.xaxis.set_tick_params(pad = 2)
cax.set_title("All Mice", fontsize = 9, pad = 2.5)


#############################################
# G. H. I MOUSE 12 20 32
#############################################
lb = ['G', 'H', 'I']
mn = ['Mouse 2', 'Mouse 3', 'Mouse 4']
idn = [0,2,3]
for i, m in enumerate(['Mouse12', 'Mouse20', 'Mouse32']):
	for j in range(4):
		ax = fig.add_subplot(outergs[2+i,j])
		noaxis(ax)
		if j == 0: ax.text(-0.2, 0.95, lb[i], transform = ax.transAxes, fontsize = 9)		
		data1 = cPickle.load(open("../../figures/figures_articles/figure1/rotated_images_"+m+".pickle", 'rb'))
		data2 = cPickle.load(open("../../data/maps/"+m+".pickle", 'rb'))
		imshow(carte38_mouse17_2[:,2250:], extent = cut_bound_map, interpolation = 'bilinear', aspect = 'equal')
		xlim(-0.5,3.0)
		ylim(0, 2.8)

		if j == 0: # POSITION HEAD DIR
			new_xy_shank = data1['new_xy_shank']		
			tmp2 = data2['headdir']
			tmp2[tmp2<0.05] = 0.0
			scatter(new_xy_shank[:,0], new_xy_shank[:,1], s = tmp2*5., label = 'HD',
					color = 'red', marker = 'o', alpha = 1.0)
			scatter(new_xy_shank[:,0], new_xy_shank[:,1], s = 1, color = 'black', marker = '.', 
					alpha = 1.0, linewidths = 0.5, label = 'shanks')
			title(mn[i], loc = 'right', fontsize = 12)
			if i == 0: leg = legend(loc = 'lower left', fontsize = 6, frameon = False, bbox_to_anchor=(0.8, 0))
		elif j == 1: # NEURON COUNT
			xnew, ynew, tmp = interpolate(data2['total'], data2['x'], data2['y'], 0.010)
			total2 = gaussian(tmp, sigma = 15.0, mode = 'reflect')
			h, w = total2.shape
			total3 = np.zeros((h*3, w*3))*np.nan
			total3[h:h*2,w:w*2] = total2.copy() + 1.0
			total3 = rotateImage(total3, -angles[idn[i]])
			total3[total3 == 0.0] = np.nan
			total3 -= 1.0
			tocrop = np.where(~np.isnan(total3))
			total3 = total3[tocrop[0].min()-1:tocrop[0].max()+1,tocrop[1].min()-1:tocrop[1].max()+1]
			xlength, ylength = getXYshapeofRotatedMatrix(data2['x'].max(), data2['y'].max(), angles[idn[i]])
			bound = (shifts[idn[i]][0],xlength+shifts[idn[i]][0],shifts[idn[i]][1],ylength+shifts[idn[i]][1])
			im = imshow(total3, extent = bound, alpha = 0.8, aspect = 'equal', cmap = count_cmap)
			#colorbar	
			cax = inset_axes(ax, "50%", "5%",
			                   bbox_to_anchor=(0.7, 0.1, 1, 1),
			                   bbox_transform=ax.transAxes, 
			                   loc = 'lower left')
			cb = colorbar(im, cax = cax, orientation = 'horizontal')
			cb.set_label('Neuron count', labelpad = 0)
			cb.ax.xaxis.set_tick_params(pad = 2)			
		elif j == 2:
			# cluster 2 burstiness									
			tmp = data1['rotated_images'][-1]
			bound = data1['bound']
			imshow(tmp, extent = bound, alpha = 0.9, aspect = 'equal', cmap = 'viridis')
			if i == 0: 
				title("Cluster 1 (Burstiness)")			
				#colorbar	
				cax = inset_axes(ax, "30%", "8%",
				               bbox_to_anchor=(0.7, -0.1, 1, 1),
				               bbox_transform=ax.transAxes, 
				               loc = 'lower left')

				cb = matplotlib.colorbar.ColorbarBase(cax, cmap='viridis', ticks = [0, 1], orientation = 'horizontal')			
				cb.set_label('Burstiness', labelpad = -20)
				cb.ax.xaxis.set_tick_params(pad = 1)								
		elif j == 3: # Cluster 2
			# cluster 1 HD									
			tmp = data1['rotated_images'][1]
			tmp[tmp<0.0] = 0.0
			bound = data1['bound']
			imshow(tmp, extent = bound, alpha = 0.9, aspect = 'equal', cmap = 'Reds')			
			if i == 0:
				title('Cluster 2 (HD)')
				#colorbar	
				cax = inset_axes(ax, "30%", "8%",
				                   bbox_to_anchor=(0.7, -0.1, 1, 1),
				                   bbox_transform=ax.transAxes, 
				                   loc = 'lower left')
				cb = matplotlib.colorbar.ColorbarBase(cax, cmap='Reds', ticks = [0, 1], orientation = 'horizontal')			
				cb.set_label('Density', labelpad = -20)
				cb.ax.xaxis.set_tick_params(pad = 1)		



# subplots_adjust(left = 0.01, right = 0.98, top = 0.98, bottom = 0.1, wspace = 0.2, hspace = 0.9)
subplots_adjust(wspace = 0.0)

savefig("../../figures/figures_articles/figart_supp_1.pdf", dpi = 900, facecolor = 'white')
os.system("evince ../../figures/figures_articles/figart_supp_1.pdf &")


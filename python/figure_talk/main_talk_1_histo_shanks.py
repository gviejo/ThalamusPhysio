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


space = pd.read_hdf("../../figures/figures_articles_v2/figure1/space.hdf5")

burst = pd.HDFStore("/mnt/DataGuillaume/MergedData/BURSTINESS.h5")['w']
burst = burst.loc[space.index]


# autocorr = pd.read_hdf("../../figures/figures_articles_v2/figure1/autocorr.hdf5")
store_autocorr = pd.HDFStore("/mnt/DataGuillaume/MergedData/AUTOCORR_ALL.h5")

# carte38_mouse17 = imread('../../figures/mapping_to_align/paxino/paxino_38_mouse17.png')
# carte38_mouse17_2 = imread('../../figures/mapping_to_align/paxino/paxino_38_mouse17_2.png')
# bound_map_38 = (-2336/1044, 2480/1044, 0, 2663/1044)
# cut_bound_map = (-86/1044, 2480/1044, 0, 2663/1044)

carte_adrien = imread('/home/guillaume/Dropbox (Peyrache Lab)/Peyrache Lab Team Folder/Projects/HPC-Thal/Figures/ATAnatomy_ALL-01.png')
carte_adrien2 = imread('/home/guillaume/Dropbox (Peyrache Lab)/Peyrache Lab Team Folder/Projects/HPC-Thal/Figures/ATAnatomy_Contour-01.png')
bound_adrien = (-398/1254, 3319/1254, -(239/1254 - 20/1044), 3278/1254)

# carte_adrien2[:,:,-1][carte_adrien2[:,:,-1]<150] = 0.0

tmp = cPickle.load(open("../../figures/figures_articles_v2/figure1/shifts.pickle", 'rb'))
angles = tmp['angles']
shifts = tmp['shifts']


hd_index = space.index.values[space['hd'] == 1]

neurontoplot = [np.intersect1d(hd_index, space.index.values[space['cluster'] == 1])[0],
				burst.loc[space.index.values[space['cluster'] == 0]].sort_values('sws').index[3],
				burst.sort_values('sws').index.values[-20]]

# specific to mouse 17
subspace = pd.read_hdf("../../figures/figures_articles_v2/figure1/subspace_Mouse17.hdf5")

data = cPickle.load(open("../../figures/figures_articles_v2/figure1/rotated_images_Mouse17.pickle", 'rb'))
rotated_images = data['rotated_images']
new_xy_shank = data['new_xy_shank']
bound = data['bound']

data 		= cPickle.load(open("../../data/maps/Mouse17.pickle", 'rb'))
x 			= data['x']
y 			= data['y']*-1.0+np.max(data['y'])
headdir 	= data['headdir']

xy_pos = new_xy_shank.reshape(len(y), len(x), 2)


# XGB score
mean_score = pd.read_hdf('/mnt/DataGuillaume/MergedData/'+'SCORE_XGB.h5')




###############################################################################################################
###############################################################################################################
# PLOT
###############################################################################################################
def figsize(scale):
	fig_width_pt = 483.69687                         # Get this from LaTeX using \the\textwidth
	inches_per_pt = 1.0/72.27                       # Convert pt to inch
	golden_mean = (np.sqrt(5.0)-1.0)/2.0            # Aesthetic ratio (you could change this)
	fig_width = fig_width_pt*inches_per_pt*scale    # width in inches
	fig_height = fig_width*golden_mean         # height in inches
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
	"axes.labelsize": 8,               # LaTeX default is 10pt font.
	"font.size": 7,
	"legend.fontsize": 7,               # Make the legend/label fonts a little smaller
	"xtick.labelsize": 7,
	"ytick.labelsize": 7,
	"pgf.preamble": [
		r"\usepackage[utf8x]{inputenc}",    # use utf8 fonts becasue your computer can handle it :)
		r"\usepackage[T1]{fontenc}",        # plots will be generated using this preamble
		],
	"lines.markeredgewidth" : 0.2,
	"axes.linewidth"        : 0.8,
	"ytick.major.size"      : 1.5,
	"xtick.major.size"      : 1.5
	}    
mpl.rcParams.update(pdf_with_latex)
import matplotlib.gridspec as gridspec
from matplotlib.pyplot import *
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

colors = ['red', 'green', 'blue', 'purple', 'orange']
cmaps = ['Reds', 'Greens', 'Blues', 'Purples', 'Oranges']
markers = ['o', '^', '*', 's']

fig = figure(figsize = figsize(1.0))#, tight_layout=True)

outergs = gridspec.GridSpec(1,2, figure = fig, hspace = 0.3)

#############################################
# A. HISTOLOGY
#############################################
gs = gridspec.GridSpecFromSubplotSpec(1,2,subplot_spec = outergs[0,:], width_ratios = [1, 1.2], wspace = 0.5)
axA = fig.add_subplot(gs[0,0])
# noaxis(axA)
histo = imread("../../data/histology/Mouse17/Mouse17_2_Slice7_Thalamus_Dapi_2.png")
imshow(histo, interpolation = 'bilinear',aspect= 'equal')
text(2500.0, 600.0, "Shanks", rotation = -10, color = 'white', fontsize = 18)
text(1600.0, 1900.0, "AD", color = 'red', fontsize = 18)
xticks([], [])
yticks([], [])
#axA.text(-0.07, 1.04, "a", transform = axA.transAxes, fontsize = 10, fontweight='bold')

#############################################
# B. MAP AD + HD
#############################################
def show_labels(ax):
	ax.text(0.68,	1.09,	"AM", 	fontsize = 10,  bbox=dict(facecolor='#C9C9C9', edgecolor = 'none', boxstyle='square,pad=-0.1'))
	ax.text(1.26,	1.26,	"VA",	fontsize = 10,  bbox=dict(facecolor='#C9C9C9', edgecolor = 'none', boxstyle='square,pad=-0.1'))
	ax.text(0.92,	2.05,	"AVd",	fontsize = 10,  bbox=dict(facecolor='#C9C9C9', edgecolor = 'none', boxstyle='square,pad=-0.1'), rotation = 50)
	ax.text(1.14,	1.72,	"AVv",	fontsize = 10,  bbox=dict(facecolor='#C9C9C9', edgecolor = 'none', boxstyle='square,pad=-0.1'))
	ax.text(1.28,	2.25,	"LD",	fontsize = 10,  bbox=dict(facecolor='#C9C9C9', edgecolor = 'none', boxstyle='square,pad=-0.1'))
	ax.text(0.42,	2.17,	"sm",	fontsize = 10,  bbox=dict(facecolor='#C9C9C9', edgecolor = 'none', boxstyle='square,pad=-0.1'))
	ax.text(0.20,	1.89,	"MD",	fontsize = 10,  bbox=dict(facecolor='#C9C9C9', edgecolor = 'none', boxstyle='square,pad=-0.1'))
	ax.text(-0.06,	1.58,	"PV",	fontsize = 10,  bbox=dict(facecolor='#C9C9C9', edgecolor = 'none', boxstyle='square,pad=-0.1'))
	ax.text(0.4,	1.5,	"IAD",	fontsize = 10,  bbox=dict(facecolor='#C9C9C9', edgecolor = 'none', boxstyle='square,pad=-0.1'), rotation = 52)
	return


suB = fig.add_subplot(gs[0,1])
# imshow(carte38_mouse17, extent = bound_map_38, interpolation = 'bilinear', aspect = 'equal')
# imshow(carte38_mouse17[:,2250:], extent = cut_bound_map, interpolation = 'bilinear', aspect = 'equal')
imshow(carte_adrien, extent = bound_adrien, interpolation = 'bessel', aspect = 'equal')
i = 1
m = 'Mouse17'


tmp2 = headdir
tmp2[tmp2<0.05] = 0.0
scatter(new_xy_shank[:,0], new_xy_shank[:,1], s = 8, color = 'black', marker = '.', 
	alpha = 1.0, linewidths = 0.5, label = 'shank positions')
scatter(new_xy_shank[:,0], new_xy_shank[:,1], s = tmp2*10., label = 'HD positions',
	color = 'red', marker = 'o', alpha = 0.6)


plot([2.2,2.2],[0,1], '-', linewidth = 1.3, color = 'black')
suB.text(2.25, 0.5, "1 mm", rotation = -90)

show_labels(suB)

leg = legend(loc = 'lower left', fontsize = 16, framealpha=1.0, bbox_to_anchor=(-0.1, -0.25)) #, title = 'HD recording sites', )

noaxis(suB)
leg.get_title().set_fontsize(7)
leg.get_frame().set_facecolor('white')

annotate('Antero-dorsal (AD)', xy=(0.9,2.4), xytext=(0.9,2.7), xycoords='data', textcoords='data',
arrowprops=dict(facecolor='black',
	shrink=0.05,
	headwidth=3,
	headlength=2,
	width=0.3),
fontsize = 16, ha = 'center', va = 'bottom')

#suB.text(-0.04, 1.03, "b", transform = suB.transAxes, fontsize = 10, fontweight = 'bold')



subplots_adjust(bottom = 0.05, top = 0.97, right = 0.98, left = 0.01, hspace = 0.1)

savefig(r"../../../Dropbox (Peyrache Lab)/Talks/fig_talk_4.png", dpi = 300, facecolor = 'white')
#savefig(r"../../../Dropbox (Peyrache Lab)/Talks/fig_talk_4.pdf", dpi = 900, facecolor = 'white')

#os.system("evince ../../figures/figures_articles_v3/figart_1.pdf &")

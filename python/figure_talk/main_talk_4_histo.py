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
	fig_height = fig_width*golden_mean        # height in inches
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
	"axes.labelsize": 16,               # LaTeX default is 10pt font.
	"font.size": 16,
	"legend.fontsize": 16,               # Make the legend/label fonts a little smaller
	"xtick.labelsize": 16,
	"ytick.labelsize": 16,
	"pgf.preamble": [
		r"\usepackage[utf8x]{inputenc}",    # use utf8 fonts becasue your computer can handle it :)
		r"\usepackage[T1]{fontenc}",        # plots will be generated using this preamble
		],
	"lines.markeredgewidth" : 1,
	"axes.linewidth"        : 2,
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

fig = figure(figsize = figsize(1.6))#, tight_layout=True)

outergs = gridspec.GridSpec(2,1, figure = fig)




# #############################################
# # F. GRadient
# #############################################
# mean_burst = pd.DataFrame(columns = ['12', '17','20', '32'])
# count_nucl = pd.DataFrame(columns = ['12', '17','20', '32'])

# for m in ['12', '17','20', '32']:
# 	subspace = pd.read_hdf("../../figures/figures_articles_v2/figure1/subspace_Mouse"+m+".hdf5")	
# 	nucleus = np.unique(subspace['nucleus'])
# 	mean_burstiness = [burst.loc[subspace.index, 'sws'][subspace['nucleus'] == nu].mean() for nu in nucleus]
# 	mean_burst[m] = pd.Series(index = nucleus, data = mean_burstiness)	
# 	total = [np.sum(subspace['nucleus'] == n) for n in nucleus]
# 	count_nucl[m] = pd.Series(index = nucleus, data = total)
# # nucleus = ['AD', 'LDvl', 'AVd', 'MD', 'AVv', 'IAD', 'CM', 'AM', 'VA', 'Re']
# nucleus = list(count_nucl.dropna().index.values)

# # mean all
# meanall = pd.DataFrame(index = nucleus, columns = ['mean','sem'])
# for n in nucleus:
# 	tmp = burst[space['nucleus'] == n]
# 	# if len(tmp)>20:
# 	meanall.loc[n,'mean'] = tmp.mean(0)['sws']
# 	meanall.loc[n,'sem'] = tmp.sem(0)['sws']

# meanall = meanall.sort_values('mean')

# # axF = fig.add_subplot(3,4,11)
# # axG = Subplot(fig, gs[1:,3])
# axG = Subplot(fig, outergs[0,:])
# fig.add_subplot(axG)
# simpleaxis(axG)
# # mean_burst = mean_burst.loc[nucleus]
# # mean_burst[0] = np.arange(len(nucleus))
# # for i, m in enumerate(['17', '12','20', '32']):	
# # 	tmp = mean_burst[[m,0]].dropna()
# # 	plot(tmp[m], tmp[0], 'o', label = str(i+1), markersize = 2, linewidth = 1)
# # plot(mean_burst.mean(1).values, mean_burst[0].values, 'o-', label = 'Mean', markersize = 2, linewidth = 1, color = 'black')
# x, s = (meanall['mean'].values.astype('float32'), meanall['sem'].values.astype('float32'))
# # plot(np.arange(len(nucleus)), x, 'o-', markersize = 2, linewidth = 1.5, color = 'black')
# # fill_between(np.arange(len(nucleus)), x-s, x+s, color = 'grey', alpha = 0.5)

# bar(np.arange(len(nucleus)), x, yerr = s, 
# 	linewidth = 3, color = 'none', edgecolor = 'black')

# leg = legend(frameon=False, bbox_to_anchor = (0.9, 1.15))
# # leg.set_title("Mouse", prop={'size':5})
# xticks(np.arange(len(nucleus)), nucleus)
# ylabel("NREM \n Burst index",  multialignment='center', rotation = 0, labelpad = 70)
# xlabel("Nuclei", labelpad = 0.1)
# # annotate(s='', xy = (4,25), xytext=(6,25), arrowprops=dict(arrowstyle='<->'))
# # text(2.5, 24.5, 'Dorsal')
# # text(6, 24.5, 'Ventral')
# # axG.invert_yaxis()
# # axG.text(-0.35, 1.05, "f", transform = axG.transAxes, fontsize = 16, fontweight='bold')


###########################################################################
# G SCORES XGB HD/NOHD
###########################################################################
# count = pd.read_hdf("../../figures/figures_articles_v2/figure1/count_time.h5")
# score = pd.read_hdf("../../figures/figures_articles_v2/figure1/score_logreg.h5")
store = pd.HDFStore("../../figures/figures_articles_v2/figure1/score_XGB_HDNOHD.h5", 'r')
score = store['score']
shuff = store['shuff']
store.close()
a = (score.mean(1)-shuff.mean(1))/(1.0 - shuff.mean(1))


# axE = Subplot(fig, gs[1,5])
axE = Subplot(fig, outergs[0,:])
fig.add_subplot(axE)
simpleaxis(axE)
# semilogx(score.loc[0:3000], '+-', markersize = 2, color = 'firebrick', linewidth = 1.5)
semilogx(a.loc[0:1000], 'o-', markersize = 3, color = 'firebrick', linewidth = 3)
axhline(0.5, linewidth = 1.0, color = 'black', alpha = 0.8, linestyle = '--')
# ylim(-0.1,1)
# title("HD classification")
xlabel("Time (ms)", labelpad = 0.1)
ylabel("Classification \n score", multialignment='center', rotation = 0, labelpad = 70)
axE.text(0.5, 0.3, 'HD/no-HD', transform = axE.transAxes, fontsize = 16)
# axE.text(-0.35, 1.05, 'g', transform = axE.transAxes, fontsize = 16, fontweight='bold')
axE.annotate('6 ms', xy=(6, a.loc[6]), 
				xytext=(2, a.loc[6]+0.3), 
				fontsize = 16,
				arrowprops=dict(facecolor='black',								
								arrowstyle="->",
								connectionstyle="arc3")
				)

# axE.semilogx(count.mean(1).loc[0:3000], 'o-', markersize = 2, color = 'black', linewidth = 1.5)
# axE2 = axE.twinx()
# axE2.spines['top'].set_visible(False)
# axE2.set_ylabel("Logit score for HD classification", rotation = -90, labelpad = 10)
# axE2.yaxis.label.set_color('firebrick')
# axE2.tick_params(axis='y', labelcolor='firebrick')

# ylabel("Clustering specificity of HD cells")

###########################################################################
# H SCORE NUCLEUS
###########################################################################
# axH = Subplot(fig, gs[2,5])
axH = Subplot(fig, outergs[1,:])
fig.add_subplot(axH)
simpleaxis(axH)
# axH.text(-0.35, 1.05, 'h', transform = axH.transAxes, fontsize = 16, fontweight='bold')
xlabel("Nuclei", labelpad = 0.1)
ylabel("Classification\n score", rotation = 0, labelpad = 70)
# title("Classification", pad = 1.0)
tmp = mean_score[('score', 'auto', 'mean')]
tmp2 = mean_score[('shuffle', 'auto', 'mean')]
tmp3 = (tmp-tmp2)/(1-tmp2)
tmp3 = tmp3.sort_values(ascending=False)
order = tmp3.index.values
# tmp2 = mean_score[('shuffle', 'swr', 'mean')].sort_values()
# bar(np.arange(len(tmp)), tmp2.values, linewidth = 1, color = 'none', edgecolor = 'black', linestyle = '--')
bar(np.arange(len(tmp3)), tmp3.values, yerr = mean_score.loc[order,('score','swr','sem')], 
	linewidth = 3, color = 'none', edgecolor = 'black')
xticks(np.arange(mean_score.shape[0]), order)#, rotation = 45)
# axhline(1/8, linestyle = '--', color = 'black', linewidth = 0.5)
# yticks([0, 0.2,0.4], [0, 20,40])



subplots_adjust(right = 0.98, left = 0.25, hspace = 0.6)


#savefig("../../figures/figures_articles_v3/figart_1.pdf", dpi = 900, facecolor = 'white')
savefig(r"../../../Dropbox (Peyrache Lab)/Talks/fig_talk_10.png", dpi = 300, facecolor = 'white')

#os.system("evince ../../figures/figures_articles_v3/figart_1.pdf &")


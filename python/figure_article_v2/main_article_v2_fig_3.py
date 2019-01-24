import numpy as np
import pandas as pd
# from matplotlib.pyplot import plot,show,draw
import scipy.io
import sys
sys.path.append("../")
from functions import *
from pylab import *
from sklearn.decomposition import PCA
import _pickle as cPickle
import matplotlib.cm as cm
import os


###############################################################################################################
# TO LOAD
###############################################################################################################
store = pd.HDFStore("../../figures/figures_articles_v2/figure6/determinant_corr.h5", 'r')
det_all = store['det_all']
shufflings = store['shufflings']
shuffl_shank = store['shuffling_shank']
store.close()


data_directory 	= '/mnt/DataGuillaume/MergedData/'
datasets 		= np.loadtxt(data_directory+'datasets_ThalHpc.list', delimiter = '\n', dtype = str, comments = '#')

# WHICH NEURONS
space = pd.read_hdf("../../figures/figures_articles_v2/figure1/space.hdf5")
burst = pd.HDFStore("/mnt/DataGuillaume/MergedData/BURSTINESS.h5")['w']
burst = burst.loc[space.index]

hd_index = space.index.values[space['hd'] == 1]

neurontoplot = [np.intersect1d(hd_index, space.index.values[space['cluster'] == 1])[0],
				burst.loc[space.index.values[space['cluster'] == 0]].sort_values('sws').index[3],
				burst.sort_values('sws').index.values[-20]]

firing_rate = pd.read_hdf("/mnt/DataGuillaume/MergedData/FIRING_RATE_ALL.h5")
fr_index = firing_rate.index.values[((firing_rate >= 1.0).sum(1) == 3).values]

# SWR MODULATION
swr_mod, swr_ses 		= loadSWRMod('/mnt/DataGuillaume/MergedData/SWR_THAL_corr.pickle', datasets, return_index=True)
nbins 					= 400
binsize					= 5
times 					= np.arange(0, binsize*(nbins+1), binsize) - (nbins*binsize)/2
swr 					= pd.DataFrame(	columns = swr_ses, 
										index = times,
										data = gaussFilt(swr_mod, (5,)).transpose())
swr = swr.loc[-500:500]

# AUTOCORR FAST
store_autocorr = pd.HDFStore("/mnt/DataGuillaume/MergedData/AUTOCORR_ALL.h5")
autocorr_wak = store_autocorr['wake'].loc[0.5:]
autocorr_rem = 	store_autocorr['rem'].loc[0.5:]
autocorr_sws = 	store_autocorr['sws'].loc[0.5:]
autocorr_wak = autocorr_wak.rolling(window = 20, win_type = 'gaussian', center = True, min_periods = 1).mean(std = 3.0)
autocorr_rem = autocorr_rem.rolling(window = 20, win_type = 'gaussian', center = True, min_periods = 1).mean(std = 3.0)
autocorr_sws = autocorr_sws.rolling(window = 20, win_type = 'gaussian', center = True, min_periods = 1).mean(std = 3.0)
autocorr_wak = autocorr_wak[2:20]
autocorr_rem = autocorr_rem[2:20]
autocorr_sws = autocorr_sws[2:20]


neurons = np.intersect1d(swr.dropna(1).columns.values, autocorr_sws.dropna(1).columns.values)
neurons = np.intersect1d(neurons, fr_index)

X = np.copy(swr[neurons].values.T)
Y = np.copy(np.vstack((autocorr_wak[neurons].values,autocorr_rem[neurons].values, autocorr_sws[neurons].values))).T
Y = Y - Y.mean(1)[:,np.newaxis]
Y = Y / Y.std(1)[:,np.newaxis]	
pca_swr = PCA(n_components=10).fit(X)
pca_aut = PCA(n_components=10).fit(Y)
pc_swr = pca_swr.transform(X)
pc_aut = pca_aut.transform(Y)

All = np.hstack((pc_swr, pc_aut))
corr = np.corrcoef(All.T)


#shuffle
Xs = np.copy(X)
Ys = np.copy(Y)
np.random.shuffle(Xs)
np.random.shuffle(Ys)
pc_swr_sh = PCA(n_components=10).fit_transform(Xs)
pc_aut_sh = PCA(n_components=10).fit_transform(Ys)
Alls = np.hstack((pc_swr_sh, pc_aut_sh))
corrsh = np.corrcoef(Alls.T)


# HEllinger distance
# store = pd.HDFStore("../../figures/figures_articles_v2/figure6/score_hellinger.h5", 'r')
store = pd.HDFStore("/mnt/DataGuillaume/MergedData/score_hellinger.h5", 'r')
HL = store['HL']
HLS = store['HLS']
store.close()

# XGB score
mean_score = pd.read_hdf(data_directory+'SCORE_XGB.h5')


###############################################################################################################
# PLOT
###############################################################################################################
def figsize(scale):
	fig_width_pt = 483.69687                         # Get this from LaTeX using \the\textwidth
	inches_per_pt = 1.0/72.27                       # Convert pt to inch
	golden_mean = (np.sqrt(5.0)-1.0)/2.0            # Aesthetic ratio (you could change this)
	fig_width = fig_width_pt*inches_per_pt*scale    # width in inches
	fig_height = fig_width*golden_mean*0.9          # height in inches
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
import matplotlib.cm as cmx
import matplotlib.colors as colors
# colors = ['#444b6e', '#708b75', '#9ab87a']

fig = figure(figsize = figsize(1.0))
gs = gridspec.GridSpec(2,3, wspace = 0.4, hspace = 0.5, width_ratios = [1,0.8,1])

#########################################################################
# A. Exemple 
#########################################################################
labels = ['HD', 'Non-bursty', 'Bursty']
titles = ['Wake', 'REM', 'SWS']
viridis = get_cmap('viridis')
cNorm = colors.Normalize(vmin=burst['sws'].min(), vmax = burst['sws'].max())
scalarMap = cmx.ScalarMappable(norm=cNorm, cmap = viridis)
color_ex = ['red', scalarMap.to_rgba(burst.loc[neurontoplot[1], 'sws']), scalarMap.to_rgba(burst.loc[neurontoplot[2], 'sws'])] 

# gsA = gridspec.GridSpecFromSubplotSpec(3,1,subplot_spec=gs[:,0], height_ratios = [1,0.2,0.2], hspace = 0.4, wspace = 0.1)

# SWR EXAMPLES
subplot(gs[0,0])
simpleaxis(gca())
for i,n in enumerate(neurontoplot):
	plot(swr[n], color = color_ex[i], linewidth = 1.5, label = labels[i])
xlabel("Time from SWRs (ms)")
ylabel("SWR modulation (z)")
locator_params(axis='y', nbins=4)
legend(edgecolor = None, facecolor = None, frameon = False, loc = 'lower left', bbox_to_anchor = (0.0, 0.9), ncol = 2)	
gca().text(-0.3, 1.10, "A", transform = gca().transAxes, fontsize = 9)

# ########################################################################
# # B. SCORE CLASSIFICATION
# ########################################################################
# subplot(gs[1,0])
# simpleaxis(gca())
# gca().text(-0.15, 1.05, "B", transform = gca().transAxes, fontsize = 9)
# xlabel("Nuclei")
# ylabel("Classification score")
# title("SWR classification")
tmp = mean_score[('score', 'swr', 'mean')]
tmp2 = mean_score[('shuffle', 'swr', 'mean')]
tmp3 = (tmp-tmp2)/(1-tmp2)
tmp3 = tmp3.sort_values(ascending = False)
order = tmp3.index.values

# # bar(np.arange(len(tmp)), tmp2.values, linewidth = 1, color = 'none', edgecolor = 'black', linestyle = '--')
# bar(np.arange(len(tmp3)), tmp3.values, yerr = mean_score.loc[order,('score','swr','sem')], 
# 	linewidth = 1, color = 'none', edgecolor = 'black')
# xticks(np.arange(mean_score.shape[0]), order)
# # axhline(1/8, linestyle = '--', color = 'black', linewidth = 0.5)
# # yticks([0, 0.2,0.4], [0, 20,40])


#########################################################################
# B. SCORE for the three neurons
#########################################################################
gsB = gridspec.GridSpecFromSubplotSpec(2,1,subplot_spec=gs[0,1], hspace = 0.2, wspace = 0.1)

store = pd.HDFStore("../../figures/figures_articles_v2/figure6/example_proba.h5", 'r')
proba_aut = store["proba_aut"]
proba_swr = store["proba_swr"]
store.close()

order2 = ['CM']+list(order)
# score SWR
subplot(gsB[0,0])
simpleaxis(gca())
ylabel("p/SWR")
# title("p(Nucleus/SWR)")
title("Classifier")
gca().text(-0.3, 1.22, "B", transform = gca().transAxes, fontsize = 9)
ct = 0
for i,n in enumerate(neurontoplot):
	bar(np.arange(len(proba_swr.index))+ct, proba_swr.loc[order2,n].values, width = 0.2, color = color_ex[i])
	ct += 0.21

xticks(np.arange(len(order2)),[])

# score AUTO
subplot(gsB[1,0])
simpleaxis(gca())
xlabel("Nuclei")
ylabel("p/Autocorr.")
# title("p(Nucleus/Autocorr.)")
ct = 0
for i,n in enumerate(neurontoplot):
	bar(np.arange(len(proba_aut.index))+ct, proba_aut.loc[order2,n].values, width = 0.2, color = color_ex[i])
	ct += 0.21

xticks(np.arange(len(order2)), order2, rotation = 90)


#########################################################################
# C. Hellingger distance
#########################################################################
subplot(gs[0,2])
simpleaxis(gca())
gca().text(-0.3, 1.10, "C", transform = gca().transAxes, fontsize = 9)
# title()
xlabel("Hellinger Distance (a.u.)")
hist(HLS.mean(), 100, color = 'black', weights = np.ones(HLS.shape[1])/float(HLS.shape[1]), histtype='stepfilled')
axvline(HL.mean(), color = 'red')
ylabel("Probability (%)")
yticks([0, 0.01, 0.02, 0.03], ['0', '1', '2', '3'])
gca().text(HL.mean()+0.01, gca().get_ylim()[1], "p<0.001",fontsize = 7, ha = 'center', color = 'red')
# cax = inset_axes(axbig, "20%", "20%",
#                bbox_to_anchor=(0, 2.5, 1, 1),
#                bbox_transform=axbig.transData, 
#                loc = 'lower left')



########################################################################
# D. PCA
########################################################################
gsA = gridspec.GridSpecFromSubplotSpec(2,1,subplot_spec=gs[1,0])#, hspace = 0.1, wspace = 0.5)#, height_ratios = [1,1,0.2,1])

# EXEMPLE PCA SWR
subplot(gsA[0,:])
simpleaxis(gca())
gca().spines['bottom'].set_visible(False)
gca().set_xticks([])
axhline(0, linewidth = 0.5, color = 'black')
for i, n in enumerate(neurontoplot):
	idx = np.where(n == neurons)[0][0]
	scatter(np.arange(pc_swr.shape[1])+i*0.2, pc_swr[idx], 2, color = color_ex[i])
	for j in np.arange(pc_swr.shape[1]):
		plot([j+i*0.2, j+i*0.2],[0, pc_swr[idx][j]], linewidth = 1.2, color = color_ex[i])

ylabel("PCA weights")
gca().yaxis.set_label_coords(-0.2,0.1)
# title("PCA")
gca().text(-0.30, 1.10, "D", transform = gca().transAxes, fontsize = 9)
gca().text(0.15, 1.05, "SWR", transform = gca().transAxes, fontsize = 8)

# EXEMPLE PCA AUTOCORR
# gsAA = gridspec.GridSpecFromSubplotSpec(1,1,subplot_spec=gsA[1,:])#, hspace = 0.1, height_ratios = [1,0.4])
ax1 = subplot(gsA[1,:])
# ax2 = subplot(gsAA[1,:], sharex = ax1)
simpleaxis(ax1)
# simpleaxis(ax2)
# ax1.spines['bottom'].set_visible(False)
# ax2.spines['bottom'].set_visible(False)
# ax1.set_ylabel("PC")
ax1.set_xticks(np.arange(10))
ax1.set_xticklabels(np.arange(10)+1)
# ax2.set_xticks([])
ax1.axhline(0, linewidth = 0.5, color = 'black')
for i, n in enumerate(neurontoplot):
	idx = np.where(n == neurons)[0][0]
	ax1.scatter(np.arange(pc_aut.shape[1])+i*0.2, pc_aut[idx], 2, color = color_ex[i])
	# ax2.scatter(np.arange(pc_aut.shape[1])+i*0.2, pc_aut[idx], 2, color = color_ex[i])
	for j in np.arange(pc_aut.shape[1]):
		ax1.plot([j+i*0.2, j+i*0.2], [0, pc_aut[idx][j]], linewidth = 1.2, color = color_ex[i])
		# ax2.plot([j+i*0.2, j+i*0.2], [0, pc_aut[idx][j]], linewidth = 1.2, color = color_ex[i])
idx = [np.where(n == neurons)[0][0] for n in neurontoplot]
xlabel("Components")
# ax1.set_ylim(pc_aut[idx,1:].min()-8.0, pc_aut[idx,1:].max()+8.0)
# ax2.set_ylim(pc_aut[idx,0].min()-2.0, pc_aut[idx,0].max()+2.0)
# ax1.set_yticks([0])
# ax2.set_yticks([-260])
# d = .005  # how big to make the diagonal lines in axes coordinates
# arguments to pass to plot, just so we don't keep repeating them
# kwargs = dict(transform=ax1.transAxes, color='k', clip_on=False, linewidth = 1)
# ax1.plot((-d, +d), (-d, +d), **kwargs)        # top-left diagonal

# kwargs.update(transform=ax2.transAxes)  # switch to the bottom axes
# ax2.plot((-d, +d), (1 - d, 1 + d), **kwargs)  # bottom-left diagonal

ax1.text(0.15, 0.95, "Autocorr.", transform = ax1.transAxes, fontsize = 8)

# title("PCA")

###########################################################################
# E MATRIX CORRELATION
###########################################################################
gsA = gridspec.GridSpecFromSubplotSpec(1,2,subplot_spec=gs[1,1])#, hspace = 0.1, wspace = 0.5)#, height_ratios = [1,1,0.2,1])
subplot(gsA[0,0])
noaxis(gca())
vmin = np.minimum(corr[0:10,10:].min(), corrsh[0:10,10:].min())
vmax = np.maximum(corr[0:10,10:].max(), corrsh[0:10,10:].max())
imshow(corr[0:10,10:], vmin = vmin, vmax = vmax)
ylabel("SWR")
xlabel("Autocorr.")
gca().text(0.25, 1.3, "Cell-by-cell correlation", transform = gca().transAxes, fontsize = 7)
gca().text(-0.35, 1.6, "E", transform = gca().transAxes, fontsize = 9)
gca().text(0.02, -0.5, r"$\rho^{2} = $"+str(np.round(1-np.linalg.det(corr),2)), transform = gca().transAxes, fontsize = 7)

# MATRIX SHUFFLED
subplot(gsA[0,1])
noaxis(gca())
imshow(corrsh[0:10,10:], vmin = vmin, vmax = vmax)
title("Shuffle", fontsize = 8, pad = 0.3)
# ylabel("SWR") 
# xlabel("Autocorr.")
gca().text(0.02, -0.5, r"$\rho^{2} = $"+str(np.round(1-np.linalg.det(corrsh),2)), transform = gca().transAxes, fontsize = 7)



#########################################################################
# F. SHUFFLING + CORR
#########################################################################
subplot(gs[1,2])
simpleaxis(gca())
axvline(1-det_all['all'], color = 'red')
hist(1-shufflings['all'], 100, color = 'black', weights = np.ones(len(shufflings['all']))/len(shufflings['all']), label = 'All', histtype='stepfilled')
hist(1-shuffl_shank, 100, color = 'grey', alpha = 0.7, weights = np.ones(len(shuffl_shank))/len(shuffl_shank), label = 'Nearby', histtype='stepfilled')
xlabel(r"Total correlation $\rho^{2}$")
ylabel("Probability (%)")
yticks([0,0.02,0.04], ['0','2','4'])
gca().text(-0.3, 1.05, "F", transform = gca().transAxes, fontsize = 9)
gca().text(1-det_all['all']-0.05, gca().get_ylim()[1], "p<0.001",fontsize = 7, ha = 'center', color = 'red')
legend(edgecolor = None, facecolor = None, frameon = False, loc = 'lower left', bbox_to_anchor = (0.35, 0.6))	

subplots_adjust(top = 0.93, bottom = 0.1, right = 0.96, left = 0.08)

savefig("../../figures/figures_articles_v2/figart_3.pdf", dpi = 900, facecolor = 'white')
os.system("evince ../../figures/figures_articles_v2/figart_3.pdf &")

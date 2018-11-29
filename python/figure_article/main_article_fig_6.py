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
store = pd.HDFStore("../../figures/figures_articles/figure6/determinant_corr.h5", 'r')
det_all = store['det_all']
shufflings = store['shufflings']
store.close()


data_directory 	= '/mnt/DataGuillaume/MergedData/'
datasets 		= np.loadtxt(data_directory+'datasets_ThalHpc.list', delimiter = '\n', dtype = str, comments = '#')

# WHICH NEURONS
space = pd.read_hdf("../../figures/figures_articles/figure1/space.hdf5")
burst = pd.HDFStore("/mnt/DataGuillaume/MergedData/BURSTINESS.h5")['w']
burst = burst.loc[space.index]

hd_index = space.index.values[space['hd'] == 1]

neurontoplot = [np.intersect1d(hd_index, space.index.values[space['cluster'] == 1])[0],
				burst.loc[space.index.values[space['cluster'] == 0]].sort_values('sws').index[3],
				burst.sort_values('sws').index.values[-20]]


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
autocorr_wak = autocorr_wak[2:150]
autocorr_rem = autocorr_rem[2:150]
autocorr_sws = autocorr_sws[2:150]


neurons = np.intersect1d(swr.dropna(1).columns.values, autocorr_sws.dropna(1).columns.values)
X = np.copy(swr[neurons].values.T)
Y = np.copy(np.vstack((autocorr_wak[neurons].values,autocorr_rem[neurons].values, autocorr_sws[neurons].values))).T
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



###############################################################################################################
# PLOT
###############################################################################################################
def figsize(scale):
	fig_width_pt = 483.69687                         # Get this from LaTeX using \the\textwidth
	inches_per_pt = 1.0/72.27                       # Convert pt to inch
	golden_mean = (np.sqrt(5.0)-1.0)/2.0            # Aesthetic ratio (you could change this)
	fig_width = fig_width_pt*inches_per_pt*scale    # width in inches
	fig_height = fig_width*golden_mean*0.8          # height in inches
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
	"font.size": 6,
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
import matplotlib.cm as cmx
import matplotlib.colors as colors
# colors = ['#444b6e', '#708b75', '#9ab87a']

fig = figure(figsize = figsize(1.0))
gs = gridspec.GridSpec(2,3, wspace = 0.4, hspace = 0.5, width_ratios = [1,0.5,1])

#########################################################################
# A. Exemple 
#########################################################################
labels = ['1\nHD', '2\nNon-bursty', '3\nBursty']
titles = ['Wake', 'REM', 'SWS']
viridis = get_cmap('viridis')
cNorm = colors.Normalize(vmin=burst['sws'].min(), vmax = burst['sws'].max())
scalarMap = cmx.ScalarMappable(norm=cNorm, cmap = viridis)
color_ex = ['red', scalarMap.to_rgba(burst.loc[neurontoplot[1], 'sws']), scalarMap.to_rgba(burst.loc[neurontoplot[2], 'sws'])] 

gsA = gridspec.GridSpecFromSubplotSpec(2,2,subplot_spec=gs[:,0], width_ratios = [1, 0.5], hspace = 0.4, wspace = 0.1)

# SWR EXAMPLES
subplot(gsA[0,0])
simpleaxis(gca())
for i,n in enumerate(neurontoplot):
	plot(swr[n], color = color_ex[i], linewidth = 0.8)
xlabel("Times (ms)")
ylabel("SWR modulation (z)")
locator_params(axis='y', nbins=4)
gca().text(-0.15, 1.05, "A", transform = gca().transAxes, fontsize = 9)
# AUTOCORR EXAMPLES
# subplot(gsA[1,0])
titles = ['Wake', 'REM', "SWS"]
tickss = [[0,4], [5], [36]]
gsAA = gridspec.GridSpecFromSubplotSpec(1,3,subplot_spec=gsA[1,0], wspace = 0.1)
for j,auto in zip(range(3),[autocorr_wak,autocorr_rem,autocorr_sws]):
	subplot(gsAA[0,j])
	simpleaxis(gca())
	for i, n in enumerate(neurontoplot):
		plot(auto[n], color = color_ex[i], linewidth = 0.8)
	if j == 1:
		xlabel("Times (ms)")
	if j == 0:
		ylabel("Autocorrelogram")
	ylim(0)
	title(titles[j])
	yticks(tickss[j])

# EXEMPLE PCA SWR
subplot(gsA[0,1])
simpleaxis(gca())
gca().spines['left'].set_visible(False)
gca().set_yticks([])
axvline(0, linewidth = 0.5, color = 'black')
for i, n in enumerate(neurontoplot):
	idx = np.where(n == neurons)[0][0]
	scatter(pc_swr[idx], np.arange(pc_swr.shape[1])+i*0.2, 3, color = color_ex[i])
	for j in np.arange(pc_swr.shape[1]):
		plot([0, pc_swr[idx][j]], [j+i*0.2, j+i*0.2], linewidth = 0.7, color = color_ex[i])

title("PCA")

# EXEMPLE PCA AUTOCORR
subplot(gsA[1,1])
simpleaxis(gca())
gca().spines['left'].set_visible(False)
gca().set_yticks([])
axvline(0, linewidth = 0.5, color = 'black')
for i, n in enumerate(neurontoplot):
	idx = np.where(n == neurons)[0][0]
	scatter(pc_aut[idx], np.arange(pc_aut.shape[1])+i*0.2, 3, color = color_ex[i])
	for j in np.arange(pc_aut.shape[1]):
		plot([0, pc_aut[idx][j]], [j+i*0.2, j+i*0.2], linewidth = 0.7, color = color_ex[i])

title("PCA")


#########################################################################
# B. Matrixes
#########################################################################
subplot(gs[0,1])
noaxis(gca())
vmin = np.minimum(corr[0:10,10:].min(), corrsh[0:10,10:].min())
vmax = np.maximum(corr[0:10,10:].max(), corrsh[0:10,10:].max())
imshow(corr[0:10,10:], vmin = vmin, vmax = vmax)
xlabel("PCA(SWR)")
ylabel("PCA(AUTOCORR)")
title("Correlation", fontsize = 8)
gca().text(-0.25, 1.15, "B", transform = gca().transAxes, fontsize = 9)
gca().text(0.08, -0.3, "|C| = "+str(np.round(np.linalg.det(corr),2)), transform = gca().transAxes, fontsize = 9)

subplot(gs[1,1])
noaxis(gca())
imshow(corrsh[0:10,10:], vmin = vmin, vmax = vmax)
title("Shuffle", fontsize = 8)
xlabel("PCA(SWR)") 
ylabel("PCA(AUTOCORR)")
gca().text(0.08, -0.3, "|C| = "+str(np.round(np.linalg.det(corrsh),2)), transform = gca().transAxes, fontsize = 9)



#########################################################################
# C. SHUFFLING + CORR
#########################################################################
subplot(gs[0,2])
simpleaxis(gca())
axvline(1-det_all['all'], color = 'black')
hist(1-shufflings['all'], 100, color = 'black', density = True, stacked = True)
xlabel("1 - |C|")
ylabel("Density (%)")
gca().text(-0.15, 1.05, "C", transform = gca().transAxes, fontsize = 9)

#########################################################################
# D. Kullbacj Leibler
#########################################################################
subplot(gs[1,2])
simpleaxis(gca())
gca().text(-0.15, 1.05, "D", transform = gca().transAxes, fontsize = 9)


# cax = inset_axes(axbig, "20%", "20%",
#                bbox_to_anchor=(0, 2.5, 1, 1),
#                bbox_transform=axbig.transData, 
#                loc = 'lower left')



subplots_adjust(top = 0.93, bottom = 0.1, right = 0.96, left = 0.08)

savefig("../../figures/figures_articles/figart_6.pdf", dpi = 900, facecolor = 'white')
os.system("evince ../../figures/figures_articles/figart_6.pdf &")

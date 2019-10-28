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

# neurontoplot = [np.intersect1d(hd_index, space.index.values[space['cluster'] == 1])[0],
# 				burst.loc[space.index.values[space['cluster'] == 0]].sort_values('sws').index[3],
# 				burst.sort_values('sws').index.values[-20]]

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




###############################################################################################################
# PLOT
###############################################################################################################
def figsize(scale):
	fig_width_pt = 483.69687                         # Get this from LaTeX using \the\textwidth
	inches_per_pt = 1.0/72.27                       # Convert pt to inch
	golden_mean = (np.sqrt(5.0)-1.0)/2.0            # Aesthetic ratio (you could change this)
	fig_width = fig_width_pt*inches_per_pt*scale    # width in inches
	fig_height = fig_width*golden_mean          # height in inches
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
	"lines.markeredgewidth" : 0.2,
	"axes.linewidth"        : 2,
	"ytick.major.size"      : 3,
	"xtick.major.size"      : 3
	}     
mpl.rcParams.update(pdf_with_latex)
import matplotlib.gridspec as gridspec
from matplotlib.pyplot import *
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import matplotlib.cm as cmx
import matplotlib.colors as colors
# colors = ['#444b6e', '#708b75', '#9ab87a']

fig = figure(figsize = figsize(2.0))
gs = gridspec.GridSpec(2,3, wspace = 0.3, hspace = 0.4, width_ratios = [1,0.8,1], height_ratios = [1,0.9])

#########################################################################
# A. Examples shank
#########################################################################
# see main_search_examples_fig3.py
# neurons_to_plot = ['Mouse17-130207_39', 'Mouse17-130207_43', 'Mouse17-130207_37']
neurons_to_plot = ['Mouse17-130207_42', 'Mouse17-130207_37']
neuron_seed = 'Mouse17-130207_43'

titles = ['Wake', 'REM', 'NREM']
# colors = ['#384d48', '#7a9b76', '#6e7271']
# cNorm = colors.Normalize(vmin=0, vmax = 1)
# scalarMap = cmx.ScalarMappable(norm=cNorm, cmap = viridis)
# color1 = scalarMap.to_rgba(1)
# color2 = 'crimson'
# color1 = 'steelblue'
# color3 = 'darkgrey'
# color1 = '#003049'
# color2 = '#d62828'
# color3 = '#fcbf49'
# color1 = 'blue'
# color2 = 'darkgrey'
# color3 = 'red'
cmap = get_cmap('tab10')
color1 = cmap(0)
color2 = cmap(1)
color3 = cmap(2)

colors = [color1, color3]
color_ex = [color1, color2, color3]
# axG = subplot(gs[2,:])
axA = subplot(gs[0,:])

noaxis(axA)
gsA = gridspec.GridSpecFromSubplotSpec(1,3,subplot_spec=gs[0,:],width_ratios=[0.6,0.6,0.6], hspace = 0.2, wspace = 0.2)#, height_ratios = [1,1,0.2,1])

new_path = data_directory+neuron_seed.split('-')[0]+'/'+neuron_seed.split("_")[0]
meanWaveF = scipy.io.loadmat(new_path+'/Analysis/SpikeWaveF.mat')['meanWaveF'][0]
lw = 3
# WAWEFORMS
gswave = gridspec.GridSpecFromSubplotSpec(1,3,subplot_spec = gsA[0,1])#, wspace = 0.3, hspace = 0.6)
axmiddle = subplot(gswave[:,1])
noaxis(gca())
for c in range(8):
	plot(meanWaveF[int(neuron_seed.split('_')[1])][c]+c*200, color = color2, linewidth = lw)
title("Mean waveforms (a.u.)", fontsize = 16)
idx = [0,2]
for i, n in enumerate(neurons_to_plot):
	axchan = subplot(gswave[:,idx[i]])
	noaxis(axchan)
	for c in range(8):
		plot(meanWaveF[int(n.split('_')[1])][c]+c*200, color = colors[i], linewidth = lw)
	# # ylabel("Channels")
	# if i == 0:
	# 	gca().text(-0.4, 1.06, "b", transform = gca().transAxes, fontsize = 16, fontweight='bold')

cax = inset_axes(axmiddle, "100%", "5%",
                   bbox_to_anchor=(-1.2, -0.1, 3.3, 1),
                   bbox_transform=axmiddle.transAxes, 
                   loc = 'lower left')
noaxis(cax)
plot([0,1],[0,0], color = 'black'		,linewidth = 1.0)
plot([0,0],[0,0.01], color = 'black'	,linewidth = 1.0)
plot([1,1],[0,0.01], color = 'black'	,linewidth = 1.0)
plot([0.5,0.5],[0,0.01],color='black'	,linewidth = 1.0)
xlabel("Shank 3")




idx = [0,2]
for i, n in enumerate(neurons_to_plot):
	gsneur = gridspec.GridSpecFromSubplotSpec(2,3,subplot_spec = gsA[0,idx[i]], wspace = 0.6, hspace = 0.6)

	pairs = [neuron_seed, n]
	# CORRELATION AUTO
	for j, ep in enumerate(['wake', 'rem', 'sws']):
		subplot(gsneur[0,j])
		simpleaxis(gca())
		title(titles[j], fontsize = 16)
		
		tmp = store_autocorr[ep][pairs]		
		tmp.loc[0] = 0.0
		tmp1 = tmp.loc[:0].rolling(window=20,win_type='gaussian',center=True,min_periods=1).mean(std=3.0)
		tmp2 = tmp.loc[0:].rolling(window=20,win_type='gaussian',center=True,min_periods=1).mean(std=3.0)
		tmp = pd.concat([tmp1.loc[:-0.5],tmp2])
		tmp.loc[0] = 0.0
		plot(tmp.loc[-50:50,neuron_seed], color = color2, linewidth = lw)
		plot(tmp.loc[-50:50,n], color = colors[i], linewidth = lw)
		if j == 1:
			xlabel("Time (ms)")
		# if i == 0 and j == 0:
		# 	gca().text(-0.9, 1.15, "a", transform = gca().transAxes, fontsize = 16, fontweight='bold')
		# if i == 1 and j == 0:
		# 	gca().text(-0.5, 1.15, "c", transform = gca().transAxes, fontsize = 16, fontweight='bold')			

	# CORRELATION SWR
	subplot(gsneur[1,:])
	simpleaxis(gca())
	plot(swr[neuron_seed], color = color2, linewidth = lw)
	plot(swr[n], color =colors[i], linewidth = lw)
	xlabel("Time from SWRs (ms)")#, labelpad = -0.1)
	if i == 0:
		ylabel("Modulation")



########################################################################
# B. PCA
########################################################################

neurontoplot = [neuron_seed]+neurons_to_plot


gsB = gridspec.GridSpecFromSubplotSpec(2,1,subplot_spec=gs[1,0])#, width_ratios = [0.05, 0.95])#, hspace = 0.1, wspace = 0.5)#, height_ratios = [1,1,0.2,1])

# EXEMPLE PCA SWR
subplot(gsB[0,:])
simpleaxis(gca())
gca().spines['bottom'].set_visible(False)
gca().set_xticks([])
axhline(0, linewidth = 1.5, color = 'black')
for i, n in enumerate(neurontoplot):
	idx = np.where(n == neurons)[0][0]
	scatter(np.arange(pc_aut.shape[1])+i*0.2, pc_aut[idx], 2, color = color_ex[i])	
	for j in np.arange(pc_swr.shape[1]):
		plot([j+i*0.2, j+i*0.2], [0, pc_aut[idx][j]], linewidth = 3, color = color_ex[i])

yticks([-4,0])
ylabel("PCA weights")
gca().yaxis.set_label_coords(-0.15,0.1)
# title("PCA")
# gca().text(-0.2, 1.15, "d", transform = gca().transAxes, fontsize = 16, fontweight='bold')
gca().text(0.15, 0.95, "Autocorr.", transform = gca().transAxes, fontsize = 16)


# EXEMPLE PCA AUTOCORR
gsAA = gridspec.GridSpecFromSubplotSpec(2,1,subplot_spec=gsB[1,:], height_ratios = [0.4,1], hspace = 0.1)#, hspace = 0.1, height_ratios = [1,0.4])
ax1 = subplot(gsAA[0,:])
ax2 = subplot(gsAA[1,:], sharex = ax1)
simpleaxis(ax1)
simpleaxis(ax2)
ax1.spines['bottom'].set_visible(False)
# ax2.spines['bottom'].set_visible(False)
ax1.set_xticks([])
ax1.xaxis.set_tick_params(size=0)

ax2.set_xticks(np.arange(10))
ax2.set_xticklabels(np.arange(10)+1)
ax2.axhline(0, linewidth = 1.5, color = 'black')
for i, n in enumerate(neurontoplot):
	idx = np.where(n == neurons)[0][0]	
	ax1.scatter(np.arange(pc_swr.shape[1])+i*0.2, pc_swr[idx], 2, color = color_ex[i])
	ax2.scatter(np.arange(pc_swr.shape[1])+i*0.2, pc_swr[idx], 2, color = color_ex[i])
	for j in np.arange(pc_aut.shape[1]):
		ax1.plot([j+i*0.2, j+i*0.2],[0, pc_swr[idx][j]], linewidth = 3, color = color_ex[i])
		ax2.plot([j+i*0.2, j+i*0.2],[0, pc_swr[idx][j]], linewidth = 3, color = color_ex[i])
		
ax2.set_xlabel("Components")
idx = [np.where(n == neurons)[0][0] for n in neurontoplot]
ax2.set_ylim(pc_swr[idx,0].min()-1, pc_swr[idx,1:].max()+0.6)
ax1.set_ylim(pc_swr[idx,0].max()-1, pc_swr[idx,0].max()+0.6)
ax1.set_yticks([13])
d = .005  # how big to make the diagonal lines in axes coordinates
kwargs = dict(transform=ax1.transAxes, color='k', clip_on=False, linewidth = 1)
ax1.plot((-d, +d), (-d, +d), **kwargs)        # top-left diagonal
kwargs.update(transform=ax2.transAxes)  # switch to the bottom axes
ax2.plot((-d, +d), (1 - d, 1 + d), **kwargs)  # bottom-left diagonal

ax1.text(0.2, 1.15, "SWR", transform = ax1.transAxes, fontsize = 16)


# title("PCA")

###########################################################################
# E MATRIX CORRELATION
###########################################################################
gsC = gridspec.GridSpecFromSubplotSpec(1,2,subplot_spec=gs[1,1])#, hspace = 0.1, wspace = 0.5)#, height_ratios = [1,1,0.2,1])
subplot(gsC[0,0])
noaxis(gca())
vmin = np.minimum(corr[0:10,10:].min(), corrsh[0:10,10:].min())
vmax = np.maximum(corr[0:10,10:].max(), corrsh[0:10,10:].max())
imshow(corr[0:10,10:], vmin = vmin, vmax = vmax)
ylabel("SWR")
xlabel("Autocorr.")
gca().text(0.25, 1.3, "Cell-by-cell correlation", transform = gca().transAxes, fontsize = 16)
# gca().text(-0.35, 1.62, "e", transform = gca().transAxes, fontsize = 9, fontweight='bold')
gca().text(0.02, -0.5, r"$\rho^{2} = $"+str(np.round(1-np.linalg.det(corr),2)), transform = gca().transAxes, fontsize = 16)

# MATRIX SHUFFLED
subplot(gsC[0,1])
noaxis(gca())
imshow(corrsh[0:10,10:], vmin = vmin, vmax = vmax)
title("Shuffle", fontsize = 16, pad = 0.6)
# ylabel("SWR") 
# xlabel("Autocorr.")
gca().text(0.02, -0.5, r"$\rho^{2} = $"+str(np.round(1-np.linalg.det(corrsh),2)), transform = gca().transAxes, fontsize = 16)



#########################################################################
# F. SHUFFLING + CORR
#########################################################################
subplot(gs[1,2])
simpleaxis(gca())
axvline(1-det_all['all'], color = 'red')
hist(1-shufflings['all'], 100, color = 'black', weights = np.ones(len(shufflings['all']))/len(shufflings['all']), label = 'All', histtype='stepfilled')
# hist(1-shuffl_shank, 100, color = 'grey', alpha = 0.7, weights = np.ones(len(shuffl_shank))/len(shuffl_shank), label = 'Nearby', histtype='stepfilled')
xlabel(r"Total correlation $\rho^{2}$")
ylabel("Probability (%)")
yticks([0,0.02,0.04], ['0','2','4'])
# gca().text(-0.3, 1.08, "f", transform = gca().transAxes, fontsize = 16, fontweight='bold')
gca().text(1-det_all['all']-0.05, gca().get_ylim()[1], "p<0.001",fontsize = 16, ha = 'center', color = 'red')
legend(edgecolor = None, facecolor = None, frameon = False, loc = 'lower left', bbox_to_anchor = (0.35, 0.6))	





subplots_adjust(top = 0.95, bottom = 0.1, right = 0.99, left = 0.06)

savefig(r"../../../Dropbox (Peyrache Lab)/Talks/fig_talk_13.png", dpi = 300, facecolor = 'white')

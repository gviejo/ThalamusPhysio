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
	fig_height = fig_width*golden_mean*1.5        # height in inches
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
gs = gridspec.GridSpec(3,3, wspace = 0.3, hspace = 0.3, width_ratios = [1,1,1], height_ratios = [1.8,1.8,1])

#########################################################################
# A. EXEMPLES HD
#########################################################################
gsA = gridspec.GridSpecFromSubplotSpec(2,1, subplot_spec=gs[0,0], hspace = 0.6, wspace = 0.4) #, width_ratios=[0.6,0.6,0.6], hspace = 0.2, wspace = 0.2)#, height_ratios = [1,1,0.2,1])
gs1 = gridspec.GridSpecFromSubplotSpec(1,2, subplot_spec=gsA[0,0], width_ratios = [0.7, 0.3])

ex_hd  = ['Mouse17-130129_14', 'Mouse17-130129_18']
titles = ['Wake', 'REM', 'NREM']

# SWR HD
subplot(gs1[0,0])
simpleaxis(gca())
plot(swr[ex_hd[0]], '-', color = 'red', linewidth = 1)
plot(swr[ex_hd[1]], '--', color = 'red', linewidth = 1)
xlabel("Time from SWRs (ms)", labelpad = -0.0)
ylabel("Modulation")

gca().text(-0.3, 1.02, "a", transform = gca().transAxes, fontsize = 10, fontweight='bold')


# TUNING CURVES HD
tcurves = cPickle.load(open('../../figures/figures_articles_v4/figure1/good_100ms_pickle/Mouse17-130129.pickle', 'rb'))['swr'][0]['tcurves']
subplot(gs1[0,1], projection = 'polar')

for n, l in zip(ex_hd, ['-', '--']):
	tmp = tcurves[int(n.split("_")[1])]
	plot(tmp/tmp.max(), l, color = 'red', linewidth = 1)
	
gca().get_xaxis().tick_bottom()
gca().get_yaxis().tick_left()
xticks(np.arange(0, 2*np.pi, np.pi/4), ['0', '', '$\pi/2$', '', '$\pi$', '', '$3\pi/2$',''])
yticks([])	
grid(linestyle = '--')
gca().tick_params(axis='x', pad = -3)
# title("Wake", pad = 7)


gs2 = gridspec.GridSpecFromSubplotSpec(1, 3, subplot_spec=gsA[1,0], wspace = 0.5)

# AUTOCORR
for i, ep in zip(range(3),['wake', 'rem', 'sws']):
	subplot(gs2[0,i])
	simpleaxis(gca())	

	tmp = store_autocorr[ep][ex_hd]		
	tmp.loc[0] = 0.0
	tmp1 = tmp.loc[:0].rolling(window=20,win_type='gaussian',center=True,min_periods=1).mean(std=3.0)
	tmp2 = tmp.loc[0:].rolling(window=20,win_type='gaussian',center=True,min_periods=1).mean(std=3.0)
	tmp = pd.concat([tmp1.loc[:-0.5],tmp2])
	tmp.loc[0] = 0.0
	plot(tmp.loc[-50:50,ex_hd[0]], '-', color = 'red', linewidth = 1)
	plot(tmp.loc[-50:50,ex_hd[1]], '--', color = 'red', linewidth = 1)
	title(titles[i], fontsize = 8, pad = 3)
	yticks([0, 5])
	if i == 0:
		ylabel("Autocorr.")
	if i == 1:
		xlabel("Time (ms)", labelpad = -0.0)

#########################################################################
# B. Examples shank
#########################################################################
# see main_search_examples_fig3.py
# neurons_to_plot = ['Mouse17-130207_39', 'Mouse17-130207_43', 'Mouse17-130207_37']
neurons_to_plot = ['Mouse17-130207_42', 'Mouse17-130207_37']
neuron_seed = 'Mouse17-130207_43'

titles = ['Wake', 'REM', 'NREM']
cmap = get_cmap('tab10')
color1 = cmap(0)
color2 = cmap(1)
color3 = cmap(2)

colors = [color1, color3]
colors = ['#4D85BD', '#7CAA2D']

color2 = 'rosybrown'

color_ex = [colors[0], color2, colors[1]]

lbs = ['b', 'c']

new_path = data_directory+neuron_seed.split('-')[0]+'/'+neuron_seed.split("_")[0]
meanWaveF = scipy.io.loadmat(new_path+'/Analysis/SpikeWaveF.mat')['meanWaveF'][0]
lw = 1.25

idx = [0,2]
for i, n in enumerate(neurons_to_plot):	
	gsB = gridspec.GridSpecFromSubplotSpec(2,1, subplot_spec=gs[0,i+1], hspace = 0.6, wspace = 0.4) #, width_ratios=[0.6,0.6,0.6], hspace = 0.2, wspace = 0.2)#, height_ratios = [1,1,0.2,1])	
	pairs = [neuron_seed, n]

	# CORRELATION SWR
	gs2 = gridspec.GridSpecFromSubplotSpec(1,2, subplot_spec=gsB[0,0], width_ratios = [0.7, 0.3])
	subplot(gs2[0,0])
	simpleaxis(gca())
	plot(swr[neuron_seed], color = color2, linewidth = lw)
	plot(swr[n], color =colors[i], linewidth = lw)
	xlabel("Time from SWRs (ms)", labelpad = -0.01)	
	
	gca().text(-0.3, 1.0, lbs[i], transform = gca().transAxes, fontsize = 10, fontweight='bold')

	# WAVEFORMS
	gswave = gridspec.GridSpecFromSubplotSpec(1,2,subplot_spec = gs2[0,1])#, wspace = 0.3, hspace = 0.6)
	subplot(gswave[:,0])
	noaxis(gca())
	for c in range(8):
		plot(meanWaveF[int(neuron_seed.split('_')[1])][c]+c*200, color = color2, linewidth = lw)

	# title("Mean waveforms (a.u.)", fontsize = 8)
	subplot(gswave[:,1])
	noaxis(gca())
	for c in range(8):
		plot(meanWaveF[int(n.split('_')[1])][c]+c*200, color = colors[i], linewidth = lw)
	if i == 0:
		xlabel("Waveforms", fontsize = 7)

	# CORRELATION AUTO
	gs3 = gridspec.GridSpecFromSubplotSpec(1, 3, subplot_spec=gsB[1,0], wspace = 0.5)
	for j, ep in enumerate(['wake', 'rem', 'sws']):
		subplot(gs3[0,j])
		simpleaxis(gca())
		title(titles[j], fontsize = 8, pad = 3)
		
		tmp = store_autocorr[ep][pairs]		
		tmp.loc[0] = 0.0
		tmp1 = tmp.loc[:0].rolling(window=20,win_type='gaussian',center=True,min_periods=1).mean(std=3.0)
		tmp2 = tmp.loc[0:].rolling(window=20,win_type='gaussian',center=True,min_periods=1).mean(std=3.0)
		tmp = pd.concat([tmp1.loc[:-0.5],tmp2])
		tmp.loc[0] = 0.0
		plot(tmp.loc[-50:50,neuron_seed], color = color2, linewidth = lw)
		plot(tmp.loc[-50:50,n], color = colors[i], linewidth = lw)
		if j == 1:
			xlabel("Time (ms)", labelpad = -0.0)

		# if i == 1 and j == 0:
		# 	gca().text(-0.5, 1.15, "c", transform = gca().transAxes, fontsize = 10, fontweight='bold')			


########################################################################
# D. MAPS FAR-AWAY EXEMPLES
########################################################################
gsD = gridspec.GridSpecFromSubplotSpec(1,3, subplot_spec=gs[1,0:2], hspace = 0.6, wspace = 0.35, width_ratios = [0.7, 0.7, 0.05]) #, width_ratios=[0.6,0.6,0.6], hspace = 0.2, wspace = 0.2)#, height_ratios = [1,1,0.2,1])	


carte_adrien = imread('/home/guillaume/Dropbox (Peyrache Lab)/Peyrache Lab Team Folder/Projects/HPC-Thal/Figures/ATAnatomy_ALL-01.png')
bound_adrien = (-398/1254, 3319/1254, -(239/1254 - 20/1044), 3278/1254)
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

def show_labels(ax):
	ax.text(0.68,	1.09,	"AM", 	fontsize = 6.5,  bbox=dict(facecolor='#C9C9C9', edgecolor = 'none', boxstyle='square,pad=-0.1'))
	ax.text(1.26,	1.26,	"VA",	fontsize = 6.5,  bbox=dict(facecolor='#C9C9C9', edgecolor = 'none', boxstyle='square,pad=-0.1'))
	ax.text(0.92,	2.05,	"AVd",	fontsize = 6.5,  bbox=dict(facecolor='#C9C9C9', edgecolor = 'none', boxstyle='square,pad=-0.1'), rotation = 50)
	ax.text(1.14,	1.72,	"AVv",	fontsize = 6.5,  bbox=dict(facecolor='#C9C9C9', edgecolor = 'none', boxstyle='square,pad=-0.1'))
	ax.text(1.28,	2.25,	"LD",	fontsize = 6.5,  bbox=dict(facecolor='#C9C9C9', edgecolor = 'none', boxstyle='square,pad=-0.1'))
	ax.text(0.42,	2.17,	"sm",	fontsize = 6.5,  bbox=dict(facecolor='#C9C9C9', edgecolor = 'none', boxstyle='square,pad=-0.1'))
	ax.text(0.20,	1.89,	"MD",	fontsize = 6.5,  bbox=dict(facecolor='#C9C9C9', edgecolor = 'none', boxstyle='square,pad=-0.1'))
	ax.text(-0.06,	1.58,	"PV",	fontsize = 6.5,  bbox=dict(facecolor='#C9C9C9', edgecolor = 'none', boxstyle='square,pad=-0.1'))
	ax.text(0.4,	1.5,	"IAD",	fontsize = 6.5,  bbox=dict(facecolor='#C9C9C9', edgecolor = 'none', boxstyle='square,pad=-0.1'), rotation = 52)
	return


suB = subplot(gsD[0,0])
imshow(carte_adrien, extent = bound_adrien, interpolation = 'bessel', aspect = 'equal')
i = 1
m = 'Mouse17'


tmp2 = headdir
tmp2[tmp2<0.05] = 0.0
scatter(new_xy_shank[:,0], new_xy_shank[:,1], s = 2, color = 'black', marker = '.', 
	alpha = 1.0, linewidths = 0.5, label = 'shank position')
scatter(new_xy_shank[:,0], new_xy_shank[:,1], s = tmp2*7., label = 'HD cell position',
	color = 'red', marker = 'o', alpha = 0.6)


plot([2.2,2.2],[0,1], '-', linewidth = 1.3, color = 'black')
suB.text(2.25, 0.5, "1 mm", rotation = -90)

# show_labels(suB)

leg = legend(loc = 'lower left', fontsize = 7, framealpha=1.0, bbox_to_anchor=(0.0, -0.09)) #, title = 'HD recording sites', )

noaxis(suB)
leg.get_title().set_fontsize(7)
leg.get_frame().set_facecolor('white')

annotate('Anterodorsal (AD)', xy=(0.9,2.4), xytext=(0.9,2.7), xycoords='data', textcoords='data',
arrowprops=dict(facecolor='black',
	shrink=0.05,
	headwidth=3,
	headlength=2,
	width=0.3),
fontsize = 7, ha = 'center', va = 'bottom')

suB.text(-0.20, 1.24, "d", transform = suB.transAxes, fontsize = 10, fontweight = 'bold')

# pair = ['Mouse17-130207_29', 'Mouse17-130211_20', 'Mouse17-130212_13']
# pair = ['Mouse17-130201_31', 'Mouse17-130218_8', 'Mouse17-130218_8']
# pair = ['Mouse17-130203_13', 'Mouse17-130211_27', 'Mouse17-130205_27']
pair = ['Mouse17-130129_3', 'Mouse17-130206_29', 'Mouse17-130212_23']

tricolor = ['#3e3e3e', '#7d7d7d', '#9e9e9e']

xy_pos = new_xy_shank.reshape(len(y), len(x), 2)

pos = space.loc[list(pair), ['session', 'shank']]
x = xy_pos[pos['session'],pos['shank'],0]
y = xy_pos[pos['session'],pos['shank'],1]

plot(x, y, linewidth = 1, color = 'black', zorder = 1)
scatter(x, y, edgecolors = 'white', c = tricolor, zorder = 2)



########################################################################
# D. EXEMPLES FAR AWAY
########################################################################
gsDD = gridspec.GridSpecFromSubplotSpec(2,1, subplot_spec=gsD[0,1], hspace = 0.6, wspace = 0.4) #, width_ratios=[0.6,0.6,0.6], hspace = 0.2, wspace = 0.2)#, height_ratios = [1,1,0.2,1])	

# CORRELATION SWR
subplot(gsDD[0,0])
simpleaxis(gca())
plot(swr[pair[0]], color = tricolor[0], linewidth = lw)
plot(swr[pair[1]], color = tricolor[1], linewidth = lw)
plot(swr[pair[2]], color = tricolor[2], linewidth = lw)
xlabel("Time from SWRs (ms)", labelpad = -0.01)	
ylabel("Modulation")

# CORRELATION AUTO
gs3 = gridspec.GridSpecFromSubplotSpec(1, 3, subplot_spec=gsDD[1,0], wspace = 0.5)
for j, ep in enumerate(['wake', 'rem', 'sws']):
	subplot(gs3[0,j])
	simpleaxis(gca())
	title(titles[j], fontsize = 8, pad = 3)
	
	tmp = store_autocorr[ep][pair]		
	tmp.loc[0] = 0.0
	tmp1 = tmp.loc[:0].rolling(window=20,win_type='gaussian',center=True,min_periods=1).mean(std=3.0)
	tmp2 = tmp.loc[0:].rolling(window=20,win_type='gaussian',center=True,min_periods=1).mean(std=3.0)
	tmp = pd.concat([tmp1.loc[:-0.5],tmp2])
	tmp.loc[0] = 0.0
	plot(tmp.loc[-50:50,pair[0]], color = tricolor[0], linewidth = lw)
	plot(tmp.loc[-50:50,pair[1]], color = tricolor[1], linewidth = lw)
	plot(tmp.loc[-50:50,pair[2]], color = tricolor[2], linewidth = lw)
	if j == 1:
		xlabel("Time lag (ms)", labelpad = -0.0)
	if j == 0:
		ylabel("Autocorr.")







########################################################################
# E. PCA
########################################################################

neurontoplot = [neuron_seed]+neurons_to_plot


gsB = gridspec.GridSpecFromSubplotSpec(2,1,subplot_spec=gs[1,-1])#, width_ratios = [0.05, 0.95])#, hspace = 0.1, wspace = 0.5)#, height_ratios = [1,1,0.2,1])

# EXEMPLE PCA SWR
subplot(gsB[0,:])
simpleaxis(gca())
gca().spines['bottom'].set_visible(False)
gca().set_xticks([])
axhline(0, linewidth = 0.5, color = 'black')
for i, n in enumerate(neurontoplot):
	idx = np.where(n == neurons)[0][0]
	scatter(np.arange(pc_aut.shape[1])+i*0.2, pc_aut[idx], 2, color = color_ex[i])	
	for j in np.arange(pc_swr.shape[1]):
		plot([j+i*0.2, j+i*0.2], [0, pc_aut[idx][j]], linewidth = 1.2, color = color_ex[i])

yticks([-4,0])
ylabel("PCA weights")
gca().yaxis.set_label_coords(-0.15,0.1)
# title("PCA")
gca().text(-0.2, 1.10, "e", transform = gca().transAxes, fontsize = 10, fontweight='bold')
gca().text(0.15, 0.95, "Autocorr.", transform = gca().transAxes, fontsize = 8)


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
ax2.axhline(0, linewidth = 0.5, color = 'black')
for i, n in enumerate(neurontoplot):
	idx = np.where(n == neurons)[0][0]	
	ax1.scatter(np.arange(pc_swr.shape[1])+i*0.2, pc_swr[idx], 2, color = color_ex[i])
	ax2.scatter(np.arange(pc_swr.shape[1])+i*0.2, pc_swr[idx], 2, color = color_ex[i])
	for j in np.arange(pc_aut.shape[1]):
		ax1.plot([j+i*0.2, j+i*0.2],[0, pc_swr[idx][j]], linewidth = 1.2, color = color_ex[i])
		ax2.plot([j+i*0.2, j+i*0.2],[0, pc_swr[idx][j]], linewidth = 1.2, color = color_ex[i])
		
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

ax1.text(0.2, 1.15, "SWRs", transform = ax1.transAxes, fontsize = 8)


# title("PCA")

###########################################################################
# F MATRIX CORRELATION
###########################################################################
gsC = gridspec.GridSpecFromSubplotSpec(1,2,subplot_spec=gs[2,0], wspace = 0.04)#, hspace = 0.1, wspace = 0.5)#, height_ratios = [1,1,0.2,1])
subplot(gsC[0,0])
noaxis(gca())
vmin = np.minimum(corr[0:10,10:].min(), corrsh[0:10,10:].min())
vmax = np.maximum(corr[0:10,10:].max(), corrsh[0:10,10:].max())
imshow(corr[0:10,10:], vmin = vmin, vmax = vmax)
ylabel("SWR")
xlabel("Autocorr.")
gca().text(0.25, 1.25, "Cell-by-cell correlation", transform = gca().transAxes, fontsize = 7)
gca().text(-0.35, 1.23, "f", transform = gca().transAxes, fontsize = 10, fontweight='bold')
gca().text(0.1, -0.45, r"$\rho^{2} = $"+str(np.round(1-np.linalg.det(corr),2)), transform = gca().transAxes, fontsize = 7)
title("Actual", fontsize = 8, pad = 2)

# MATRIX SHUFFLED
subplot(gsC[0,1])
noaxis(gca())
imshow(corrsh[0:10,10:], vmin = vmin, vmax = vmax)
title("Shuffle", fontsize = 8, pad = 2)
# ylabel("SWR") 
# xlabel("Autocorr.")
gca().text(0.15, -0.45, r"$\rho^{2} = $"+str(np.round(1-np.linalg.det(corrsh),2)), transform = gca().transAxes, fontsize = 7)



#########################################################################
# G. SHUFFLING + CORR
#########################################################################
subplot(gs[2,1])
simpleaxis(gca())
axvline(1-det_all['all'], color = 'red')
hist(1-shufflings['all'], 100, color = 'black', weights = np.ones(len(shufflings['all']))/len(shufflings['all']), label = 'All', histtype='stepfilled')
hist(1-shuffl_shank, 100, color = 'grey', alpha = 0.7, weights = np.ones(len(shuffl_shank))/len(shuffl_shank), label = 'Nearby', histtype='stepfilled')
xlabel(r"Total correlation $\rho^{2}$")
ylabel("Probability (%)")
yticks([0,0.02,0.04], ['0','2','4'])
gca().text(-0.23, 1.08, "g", transform = gca().transAxes, fontsize = 10, fontweight='bold')
gca().text(1-det_all['all']-0.05, gca().get_ylim()[1], "p<0.001",fontsize = 7, ha = 'center', color = 'red')
legend(edgecolor = None, facecolor = None, frameon = False, loc = 'lower left', bbox_to_anchor = (0.35, 0.6))	


#########################################################################
# H. CONTROL 1 
#########################################################################
# subplot(gs[2,2])
# simpleaxis(gca())

gsG = gridspec.GridSpecFromSubplotSpec(2,1,subplot_spec=gs[2,2], hspace = 0.4)#, hspace = 0.1, wspace = 0.5)#, height_ratios = [1,1,0.2,1])

store = pd.HDFStore("../../figures/figures_articles_v2/figure6/determinant_corr_noSWS.h5", 'r')
det_all = store['det_all']
shufflings = store['shufflings']
store.close()

subplot(gsG[0,0])
simpleaxis(gca())
gca().text(-0.22, 1.2, "h", transform = gca().transAxes, fontsize = 9, fontweight='bold')

# colors = ['blue', 'red', 'green']
colors = ["#CA3242","#849FAD",  "#27647B", "#57575F"]
labels = ['WAKE', 'REM', 'NREM']
offset = [0.0265, 0.0265, 0.011]
for i, k in enumerate(['wak', 'rem', 'sws']):
	shuf, x = np.histogram(1-shufflings[k], bins = 100, weights = np.ones(len(shufflings[k]))/len(shufflings[k]))
	axvline(1-det_all[k], color = colors[i])
	# plot([1-det_all[k], 1-det_all[k]], [0, 0.032], color = colors[i], label = labels[i])
	plot(x[0:-1]+np.diff(x), shuf, color = colors[i], alpha = 0.7)
	# hist(, label = k, histtype='stepfilled', facecolor = 'None', edgecolor = colors[i])
	# gca().text(1-det_all[k], gca().get_ylim()[1], "p<0.001",fontsize = 7, ha = 'center', color = 'red')
	gca().text(1-det_all[k], offset[i], labels[i], ha = 'center', fontsize = 7, bbox = dict(facecolor='white', edgecolor=colors[i],boxstyle='square,pad=0.2'))
axvline(0.33, color = 'black', linestyle = '--')
gca().text(0.33, 0.025, 'ALL', ha = 'center', fontsize = 7, bbox = dict(facecolor='white', edgecolor='black',boxstyle='square,pad=0.2'))
ylim(0, 0.035)
# xlabel(r"Total correlation $\rho^{2}$")
ylabel("P (%)")
yticks([0,0.01,0.02,0.03], ['0','1','2','3'])
# gca().text(-0.15, 1.0, "A", transform = gca().transAxes, fontsize = 9)
legend(edgecolor = None, facecolor = None, frameon = False, loc = 'lower left', bbox_to_anchor = (0.35, 0.6))	

# #########################################################################
# # I CONTROL 2
# #########################################################################
store = pd.HDFStore("../../figures/figures_articles_v2/figure6/determinant_corr_noSWS_shank_shuffled.h5", 'r')
det_all = store['det_all']
shufflings = store['shufflings']
store.close()

subplot(gsG[1,0])
simpleaxis(gca())
gca().text(-0.22, 1.2, "i", transform = gca().transAxes, fontsize = 9, fontweight='bold')

# colors = ['blue', 'red', 'green']
labels = ['WAKE', 'REM', 'NREM']
offset = [0.125, 0.125, 0.06]
for i, k in enumerate(['wak', 'rem', 'sws']):
	shuf, x = np.histogram(1-shufflings[k], bins = 20, weights = np.ones(len(shufflings[k]))/len(shufflings[k]))
	axvline(1-det_all[k], color = colors[i])
	# plot([1-det_all[k], 1-det_all[k]], [0, 0.032], color = colors[i], label = labels[i])
	plot(x[0:-1]+np.diff(x), shuf, color = colors[i], alpha = 1)
	# hist(, label = k, histtype='stepfilled', facecolor = 'None', edgecolor = colors[i])
	# gca().text(1-det_all[k], gca().get_ylim()[1], "p<0.001",fontsize = 7, ha = 'center', color = 'red')
	gca().text(1-det_all[k], offset[i], labels[i], ha = 'center', fontsize = 7, bbox = dict(facecolor='white', edgecolor=colors[i],boxstyle='square,pad=0.2'))
axvline(0.33, color = 'black', linestyle = '--')
gca().text(0.33, 0.12, 'ALL', ha = 'center', fontsize = 7, bbox = dict(facecolor='white', edgecolor='black',boxstyle='square,pad=0.2'))
ylim(0, 0.16)
xlabel(r"Total correlation $\rho^{2}$")
ylabel("P (%)")
yticks([0,0.05,0.10,0.15], ['0','5','10','15'])
# gca().text(-0.15, 1.0, "A", transform = gca().transAxes, fontsize = 9)
legend(edgecolor = None, facecolor = None, frameon = False, loc = 'lower left', bbox_to_anchor = (0.35, 0.6))	

store = pd.HDFStore("../../figures/figures_articles_v2/figure6/determinant_corr.h5", 'r')
det_all = store['det_all']
shufflings = store['shufflings']
shuffl_shank = store['shuffling_shank']
store.close()

shuf, x = np.histogram(1-shuffl_shank, bins = 20, weights = np.ones(len(shuffl_shank))/len(shuffl_shank))
plot(x[0:-1]+np.diff(x), shuf, '--', color = colors[-1], alpha = 0.7)










subplots_adjust(top = 0.98, bottom = 0.06, right = 0.99, left = 0.06)

savefig("../../figures/figures_articles_v4/figart_4.pdf", dpi = 900, facecolor = 'white')
os.system("evince ../../figures/figures_articles_v4/figart_4.pdf &")


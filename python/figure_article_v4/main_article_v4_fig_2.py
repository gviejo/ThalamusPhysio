#!/usr/bin/env python


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
import neuroseries as nts
import os
from scipy.ndimage import gaussian_filter	


data_directory 	= '/mnt/DataGuillaume/MergedData/'
datasets 		= np.loadtxt(data_directory+'datasets_ThalHpc.list', delimiter = '\n', dtype = str, comments = '#')
session 		= 'Mouse12/Mouse12-120810'
neurons 		= [session.split("/")[1]+"_"+str(u) for u in [23, 19, 40]]

path_snippet 	= "../../figures/figures_articles_v4/figure1/"

spike_in_swr 	= pd.HDFStore(path_snippet+'spikes_in_swr_'+session.split("/")[1]+'.h5')		

store 			= pd.HDFStore(path_snippet+'snippet_'+session.split("/")[1]+'.h5')
H0 = store['H0']
Hm = store['Hm']
Hstd = store['Hstd']
store.close()
###################################################################################################################
# LFP EXEMPLE
###################################################################################################################
# session_ex = 'Mouse12/Mouse12-120807'
# generalinfo 	= scipy.io.loadmat(data_directory+session_ex+'/Analysis/GeneralInfo.mat')
# shankStructure 	= loadShankStructure(generalinfo)	
# if len(generalinfo['channelStructure'][0][0][1][0]) == 2:
# 	hpc_channel 	= generalinfo['channelStructure'][0][0][1][0][1][0][0] - 1
# else:
# 	hpc_channel 	= generalinfo['channelStructure'][0][0][1][0][0][0][0] - 1	
# spikes,shank	= loadSpikeData(data_directory+session_ex+'/Analysis/SpikeData.mat', shankStructure['thalamus'])		
# n_channel,fs, shank_to_channel = loadXML(data_directory+session_ex+"/"+session_ex.split("/")[1]+'.xml')	
# hd_info 			= scipy.io.loadmat(data_directory+session_ex+'/Analysis/HDCells.mat')['hdCellStats'][:,-1]
# hd_phi 				= scipy.io.loadmat(data_directory+session_ex+'/Analysis/HDCells.mat')['hdCellStats'][:,0]
# hd_info_neuron		= np.array([hd_info[n] for n in spikes.keys()])
# hd_phi_neuron 		= np.array([hd_phi[n] for n in spikes.keys()])
# lfp_hpc 		= loadLFP(data_directory+session_ex+"/"+session_ex.split("/")[1]+'.eeg', n_channel, hpc_channel, float(fs), 'int16')
# sws_ex = nts.IntervalSet(start = 3920, end = 3923, time_units = 's')
# rem_ex = nts.IntervalSet(start = 3191, end = 3195, time_units = 's')
# lfp_sws_ex 		= lfp_hpc.restrict(sws_ex)
# lfp_rem_ex 		= lfp_hpc.restrict(rem_ex)
# spikes_sws_ex 	= pd.concat({n:spikes[n].restrict(sws_ex).isnull()*n for n in spikes}, axis = 1)
# spikes_rem_ex 	= pd.concat({n:spikes[n].restrict(rem_ex).isnull()*n for n in spikes}, axis = 1)
# rip_ep,rip_tsd 	= loadRipples(data_directory+session_ex)

# store_ex = pd.HDFStore('../../figures/figures_articles_v2/figure3/lfp_exemple.h5', 'w')
# store_ex.put('lfp_sws_ex', lfp_sws_ex.as_series())	
# store_ex.put('lfp_rem_ex', lfp_rem_ex.as_series())
# store_ex.put('spikes_sws_ex', spikes_sws_ex)
# store_ex.put('spikes_rem_ex', spikes_rem_ex)
# store_ex.put('hd_info_neuron', pd.Series(hd_info_neuron))
# store_ex.put('rip_ep', pd.DataFrame(rip_ep.intersect(sws_ex)))
# store_ex.put('rip_tsd', rip_tsd.restrict(sws_ex).as_series())
# store_ex.put('hd_phi_neuron', pd.Series(hd_phi_neuron))
# store_ex.close()


store_ex = pd.HDFStore('../../figures/figures_articles_v2/figure3/lfp_exemple.h5', 'r')
lfp_sws_ex  = store_ex['lfp_sws_ex']
lfp_rem_ex  = store_ex['lfp_rem_ex'] 	
spikes_sws_ex  = store_ex['spikes_sws_ex']
spikes_rem_ex  = store_ex['spikes_rem_ex']
hd_info_neuron = store_ex['hd_info_neuron']
rip_ep = store_ex['rip_ep']
rip_tsd = store_ex['rip_tsd']
hd_phi_neuron = store_ex['hd_phi_neuron']
store_ex.close()


# generalinfo 	= scipy.io.loadmat(data_directory+session+'/Analysis/GeneralInfo.mat')
# shankStructure 	= loadShankStructure(generalinfo)
# if len(generalinfo['channelStructure'][0][0][1][0]) == 2:
# 	hpc_channel 	= generalinfo['channelStructure'][0][0][1][0][1][0][0] - 1
# else:
# 	hpc_channel 	= generalinfo['channelStructure'][0][0][1][0][0][0][0] - 1	
# spikes,shank	= loadSpikeData(data_directory+session+'/Analysis/SpikeData.mat', shankStructure['thalamus'])		
# allneurons 		= [session.split("/")[1]+"_"+str(list(spikes.keys())[i]) for i in spikes.keys()]

# theta_mod, theta_ses 	= loadThetaMod('/mnt/DataGuillaume/MergedData/THETA_THAL_mod.pickle', datasets, return_index=True)
# theta_mod_rem 	= pd.DataFrame(index = theta_ses['rem'], columns = ['phase', 'pvalue', 'kappa'], data = theta_mod['rem'])
# theta_mod_rem 	= theta_mod_rem.loc[allneurons]
# theta_mod_rem['phase'] += 2*np.pi
# theta_mod_rem['phase'] %= 2*np.pi
# theta_mod_rem 	= theta_mod_rem.sort_values('phase')
# allneurons_sorted = theta_mod_rem.index.values


# spikes_swr_ex = {
# 'Mouse12-120810_37':store['spike_swrMouse12-120810_37'],
# 'Mouse12-120810_38':store['spike_swrMouse12-120810_38'],
# 'Mouse12-120810_40':store['spike_swrMouse12-120810_40']	
# }
# spikes_theta_ex = {
# 'Mouse12-120810_37':store['spike_thetaMouse12-120810_37'],
# 'Mouse12-120810_38':store['spike_thetaMouse12-120810_38'],
# 'Mouse12-120810_40':store['spike_thetaMouse12-120810_40']	
# }
# swr_ep = store['swr_ep']

# generalinfo 	= scipy.io.loadmat(data_directory+session+'/Analysis/GeneralInfo.mat')
# shankStructure 	= loadShankStructure(generalinfo)
# if len(generalinfo['channelStructure'][0][0][1][0]) == 2:
# 	hpc_channel 	= generalinfo['channelStructure'][0][0][1][0][1][0][0] - 1
# else:
# 	hpc_channel 	= generalinfo['channelStructure'][0][0][1][0][0][0][0] - 1	
# spikes,shank	= loadSpikeData(data_directory+session+'/Analysis/SpikeData.mat', shankStructure['thalamus'])		
# allneurons 		= [session.split("/")[1]+"_"+str(list(spikes.keys())[i]) for i in spikes.keys()]

# theta_mod, theta_ses 	= loadThetaMod('/mnt/DataGuillaume/MergedData/THETA_THAL_mod.pickle', datasets, return_index=True)
# theta_mod_rem 	= pd.DataFrame(index = theta_ses['rem'], columns = ['phase', 'pvalue', 'kappa'], data = theta_mod['rem'])
# theta_mod_rem 	= theta_mod_rem.loc[allneurons]
# theta_mod_rem['phase'] += 2*np.pi
# theta_mod_rem['phase'] %= 2*np.pi
# theta_mod_rem 	= theta_mod_rem.sort_values('phase')
# allneurons_sorted = theta_mod_rem.index.values

# carte38_mouse17 = imread('../../figures/mapping_to_align/paxino/paxino_38_mouse17_2.png')
# bound_map_38 = (-2336/1044, 2480/1044, 0, 2663/1044)
# cut_bound_map = (-86/1044, 2480/1044, 0, 2663/1044)

carte_adrien = imread('/home/guillaume/Dropbox (Peyrache Lab)/Peyrache Lab Team Folder/Projects/HPC-Thal/Figures/ATAnatomy_ALL-01.png')
carte_adrien2 = imread('/home/guillaume/Dropbox (Peyrache Lab)/Peyrache Lab Team Folder/Projects/HPC-Thal/Figures/ATAnatomy_Contour-01.png')
bound_adrien = (-398/1254, 3319/1254, -(239/1254 - 20/1044), 3278/1254)


mappings = pd.read_hdf("/mnt/DataGuillaume/MergedData/MAPPING_NUCLEUS.h5")

# from figure 5
nucleus = ['AD', 'AVd', 'IAD', 'PV', 'AM', 'AVv', 'MD', 'sm']

p_40 = "-0.03"
p_60 = "0.06"

# score from figure 6
# XGB score
mean_score = pd.read_hdf(data_directory+'SCORE_XGB.h5')

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
# colors = ['#444b6e', '#708b75', '#9ab87a']
colors = ['red', '#231f20', '#707174']#, '#abacad']

fig = figure(figsize = figsize(1.0))

outergs = gridspec.GridSpec(3,2, figure = fig, height_ratios = [1.0,1.5,0.8], wspace = 0.3, hspace = 0.3, width_ratios = [0.6, 0.4])

#############################################
# A. HISTOLOGY
#############################################
gs = gridspec.GridSpecFromSubplotSpec(1,4,subplot_spec = outergs[0,:], width_ratios=[0.6,0.7,0.1,1.2],wspace = 0.1)
axA = fig.add_subplot(gs[0,0])
# noaxis(axA)
histo = imread("../../data/histology/Mouse17/Mouse17_2_Slice7_Thalamus_Dapi_2.png")
imshow(histo, interpolation = 'bilinear',aspect= 'equal')
text(2500.0, 600.0, "Shanks", rotation = -10, color = 'white', fontsize = 9)
text(1600.0, 1900.0, "AD", color = 'red', fontsize = 8)
xticks([], [])
yticks([], [])
axA.text(-0.15, 1.04, "a", transform = axA.transAxes, fontsize = 10, fontweight='bold')

#############################################
# B. MAP AD + HD
#############################################
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


suB = fig.add_subplot(gs[0,1])
imshow(carte_adrien, extent = bound_adrien, interpolation = 'bessel', aspect = 'equal')
i = 1
m = 'Mouse17'


tmp2 = headdir
tmp2[tmp2<0.05] = 0.0
scatter(new_xy_shank[:,0], new_xy_shank[:,1], s = 1.5, color = 'black', marker = '.', 
	alpha = 1.0, linewidths = 0.5, label = 'shank positions')
scatter(new_xy_shank[:,0], new_xy_shank[:,1], s = tmp2*7., label = 'HD positions',
	color = 'red', marker = 'o', alpha = 0.6)


plot([2.2,2.2],[0,1], '-', linewidth = 1.3, color = 'black')
suB.text(2.25, 0.5, "1 mm", rotation = -90)

show_labels(suB)

leg = legend(loc = 'lower left', fontsize = 7, framealpha=1.0, bbox_to_anchor=(0.0, -0.09)) #, title = 'HD recording sites', )

noaxis(suB)
leg.get_title().set_fontsize(7)
leg.get_frame().set_facecolor('white')

annotate('Antero-dorsal (AD)', xy=(0.9,2.4), xytext=(0.9,2.7), xycoords='data', textcoords='data',
arrowprops=dict(facecolor='black',
	shrink=0.05,
	headwidth=3,
	headlength=2,
	width=0.3),
fontsize = 7, ha = 'center', va = 'bottom')

suB.text(-0.04, 1.03, "b", transform = suB.transAxes, fontsize = 10, fontweight = 'bold')


#############################################################################
# C. LFP EXEMPLE
#############################################################################
gsEx = gridspec.GridSpecFromSubplotSpec(2,1, subplot_spec = gs[0,3], hspace = 0, wspace = 0, height_ratios = [0.4,1])

lbs = ['c']
titles = ['NREM']
# 1 lfps
for i, lfp in zip(range(1),[lfp_sws_ex]):
	# ax = Subplot(fig, )
	ax = fig.add_subplot(gsEx[0,i])
	noaxis(ax)
	plot(lfp, color = 'black', linewidth = 0.4)
	plot(rip_tsd.index.values, [1300]*2, '*', color = 'blue', markersize = 5)
	title(titles[i], pad = -2)
	text(-0.15, 1.1, lbs[i], transform=ax.transAxes, fontsize = 10, fontweight='bold')
	ylabel("CA1")

		
# 2 spikes
for i, spikes in zip(range(1), [spikes_sws_ex]):
	# ax = Subplot(fig, )
	ax = fig.add_subplot(gsEx[1,i])
	noaxis(ax)
	# no hd
	id = 0
	for n in np.where(hd_info_neuron == 0)[0]:
		plot(spikes[n].dropna().replace(n, id), '|', markersize = 2, mew = 0.8, color = 'black')
		id += 1
	# hd 
	hd_order = np.where(hd_info_neuron==1)[0][np.argsort(hd_phi_neuron.values[hd_info_neuron==1])]
	for n in hd_order:
		plot(spikes[n].dropna().replace(n, id), '|', markersize = 2, mew = 0.8, color = 'red')
		id += 1

	if i == 0:
		ylabel("Thalamus")
	start = spikes.index.min()
	plot([start, start+5e5], [-2, -2], color = 'black', linewidth = 2)
	text(0.05, -0.1,'500 ms', transform=ax.transAxes, fontsize = 8)

	ylim(-2, id)

#############################################################################
# D. RIPPLES MODULATION
#############################################################################
gsC = gridspec.GridSpecFromSubplotSpec(5,4,subplot_spec = outergs[1,0], wspace = 0.6, width_ratios = [0.07, 1, 1, 1])
for i, n in enumerate(neurons):	
	# spikes	
	ax = Subplot(fig, gsC[0:2,i+1])
	fig.add_subplot(ax)
	simpleaxis(ax)
	if neurons.index(n) == 1:
		text(0.5, 1.23,'SWR modulation', horizontalalignment='center', verticalalignment='center', transform=ax.transAxes, fontsize = 8)
	if neurons.index(n) == 0:
		ylabel("SWR\nevents",  multialignment='center',  labelpad = 1)
		ax.text(-0.75, 1.05, "d", transform = ax.transAxes, fontsize = 10, fontweight='bold')
	if neurons.index(n) > 0:
		ax.spines['left'].set_visible(False)	
		yticks([], [])
	ax.spines['bottom'].set_visible(False)	
	xticks([], [])
	sp = spike_in_swr[n][np.arange(0, 500)]
	plot(sp.iloc[:,0:100], '|', markersize = 1, color = colors[neurons.index(n)], mew = 0.5)	

	# firing rate	
	ax = Subplot(fig, gsC[2,i+1])
	fig.add_subplot(ax)
	simpleaxis(ax)
	ax.spines['bottom'].set_visible(False)	
	H0[n] = gaussFilt(H0[n].values, (2,))
	plot(H0[n].loc[-500:500], color = colors[neurons.index(n)], label = '', linewidth = 0.7)
	plot(Hm[n].loc[-500:500], ':', color = colors[neurons.index(n)], label = 'Jitter', linewidth = 0.7)

	axvline(0, color = 'grey', linewidth = 0.5)
	xticks([], [])
	if neurons.index(n) == 0:
		ylabel('Rate \n (Hz)', verticalalignment = 'top', labelpad = 20)
		legend(edgecolor = None, facecolor = None, frameon = False, loc = 'lower left', bbox_to_anchor = (0.1, -0.45))	

	# Z score	
	ax = Subplot(fig, gsC[3:,i+1])
	fig.add_subplot(ax)
	simpleaxis(ax)
	z = pd.DataFrame((H0[n] - Hm[n])/Hstd.loc[n][0])
	z['filt'] = gaussFilt(z.values.flatten(), (5,))

	plot(z['filt'].loc[-500:500],  color = colors[neurons.index(n)], linewidth = 2)
	# xlabel('Time from \n $\mathbf{Sharp\ Waves\ ripples}$ (ms)', fontsize = 8)
	if neurons.index(n) == 1:
		xlabel('Time from SWRs (ms)', fontsize = 7)
	if neurons.index(n) == 0:
		ylabel('Modulation\n(z)', verticalalignment = 'bottom')
	axvline(0, color = 'grey', linewidth = 0.5)	
	if i in [0, 1]:
		ylim(-2,2)
	# yticks([-1,0,1,2,3])

# #############################################################################
# # E. MEAN SWR HD VS NO-HD
# #############################################################################
data_directory 	= '/mnt/DataGuillaume/MergedData/'
mappings = pd.read_hdf("/mnt/DataGuillaume/MergedData/MAPPING_NUCLEUS.h5")
datasets 		= np.loadtxt(data_directory+'datasets_ThalHpc.list', delimiter = '\n', dtype = str, comments = '#')
swr_mod, swr_ses 		= loadSWRMod('/mnt/DataGuillaume/MergedData/SWR_THAL_corr.pickle', datasets, return_index=True)
nbins 					= 400
binsize					= 5
times 					= np.arange(0, binsize*(nbins+1), binsize) - (nbins*binsize)/2
swr_mod 					= pd.DataFrame(	columns = swr_ses, 
										index = times,
										data = gaussFilt(swr_mod, (5,)).transpose())
swr_mod = swr_mod.drop(swr_mod.columns[swr_mod.isnull().any()].values, axis = 1)
swr_mod = swr_mod.loc[-500:500]

neurons = np.intersect1d(swr_mod.columns.values, mappings.index.values)

hd_neurons = mappings.loc[neurons][mappings.loc[neurons, 'hd'] == 1].index.values
nohd_neurons = mappings.loc[neurons][mappings.loc[neurons, 'hd'] == 0].index.values

gsm = gridspec.GridSpecFromSubplotSpec(3,1, subplot_spec = outergs[1,1], height_ratios = [0.7, 0.4, 0.6], hspace = 0.1)

subplot(gsm[0,0])
simpleaxis(gca())
m = swr_mod[hd_neurons].mean(1)
v = swr_mod[hd_neurons].sem(1)
plot(swr_mod[hd_neurons].mean(1), label = 'HD', color = 'red')
fill_between(m.index.values, m-v, m+v, color = 'red', alpha = 0.5, linewidth =0)
m = swr_mod[nohd_neurons].mean(1)
v = swr_mod[nohd_neurons].sem(1)
plot(swr_mod[nohd_neurons].mean(1), label = 'non-HD', color = 'black')
fill_between(m.index.values, m-v, m+v, color = 'grey', alpha = 0.5, linewidth = 0)
ylabel("SWRs mod.", labelpad = 2, y = 0.6)
xlim(-500,500)
legend(frameon=False,loc = 'lower left', bbox_to_anchor=(0.7,0.58),handlelength=1,ncol = 1)
xticks([], [])
axvline(0, linestyle = '--', linewidth = 1, alpha = 0.5, color = 'black')
gca().text(-0.25, 1.05, "e", transform = gca().transAxes, fontsize = 10, fontweight='bold')


subplot(gsm[1,0])
tmp = swr_mod[hd_neurons]
# idx = tmp.idxmax()
# tmp = tmp[idx.index.values[np.argsort(idx.values)]]
idx = tmp.loc[0].sort_values().index.values
tmp = tmp[idx[::-1]]
sc = imshow(tmp.T, aspect = 'auto', cmap = "coolwarm", vmin = -0.25, vmax = 0.25)
xticks([], [])
yticks([len(hd_neurons)])
cax = inset_axes(gca(), "2%", "30%",
                   bbox_to_anchor=(-0.08, 0.3, 1, 1),
                   bbox_transform=gca().transAxes, 
                   loc = 'lower left')
cb = colorbar(sc, cax = cax, orientation = 'vertical', ticks = [-0.25, 0.25])
# cb.ax.set_title('z')#, labelpad = -24, fontsize = 8)
cb.ax.xaxis.set_tick_params(pad = 1)
cb.ax.yaxis.set_ticks_position('left')

subplot(gsm[2,0])
tmp = swr_mod[nohd_neurons]
# idx = tmp.idxmax()
# tmp = tmp[idx.index.values[np.argsort(idx.values)]]
idx = tmp.loc[0].sort_values().index.values
tmp = tmp[idx[::-1]]
sc = imshow(tmp.T, aspect = 'auto', cmap = "bone", vmin = -0.4, vmax = 0.4)
xticks([0,100,200],[-500,0,500])
xlabel('Time from SWRs (ms)', fontsize = 7)
yticks([len(nohd_neurons)])
cax = inset_axes(gca(), "2%", "30%",
                   bbox_to_anchor=(-0.08, 0.3, 1, 1),
                   bbox_transform=gca().transAxes, 
                   loc = 'lower left')
cb = colorbar(sc, cax = cax, orientation = 'vertical', ticks = [-0.4, 0.4])
# cb.ax.set_title('z')#, labelpad = -24, fontsize = 8)
cb.ax.xaxis.set_tick_params(pad = 1)
cb.ax.yaxis.set_ticks_position('left')



# #############################################################################
# # F. MAPS
# #############################################################################
def softmax(x, b1 = 10.0, b2 = 0.5, lb = 0.2):
	x -= x.min()
	x /= x.max()
	return (1.0/(1.0+np.exp(-(x-b2)*b1)) + lb)/(1.0+lb)


angles = np.array([15.0, 10.0, 15.0, 20.0])
pos = [1,0,2,3]
i = 0
m = 'Mouse17'
times2 					= swr_mod.index.values
data = cPickle.load(open("../../data/maps/"+m+".pickle", 'rb'))
theta 	= data['movies']['theta']
swr 	= data['movies']['swr']
total 	= data['total']
x 		= data['x']
y 		= data['y']
headdir = data['headdir']
jpc 	= data['jpc']
jpc 	= pd.DataFrame(index = times2, data = jpc)

toplot = pd.DataFrame(index = ['Mouse12','Mouse17','Mouse20','Mouse32'], columns = pd.MultiIndex.from_product([range(3),['start','end']]))
for i,j,k in zip(range(3),[-80,120,250],[0,200,330]): 
	toplot.loc['Mouse17',(i,'start')] = j
	toplot.loc['Mouse17',(i,'end')] = k	


gsm = gridspec.GridSpecFromSubplotSpec(1,3, subplot_spec = outergs[2,0])
bound = cPickle.load(open("../../figures/figures_articles/figure1/rotated_images_"+m+".pickle", 'rb'))['bound']
newswr = []
for j in range(3):
	tmp = swr[:,:,np.where(times2 == toplot.loc[m,(j,'start')])[0][0]:np.where(times2 == toplot.loc[m,(j,'end')])[0][0]].mean(-1)
	xnew, ynew, frame = interpolate(tmp.copy(), x, y, 0.01)
	frame = gaussian_filter(frame, (10, 10))
	newswr.append(frame)
newswr = np.array(newswr)
newswr = newswr - newswr.min()
newswr = newswr / newswr.max()	
newswr = softmax(newswr, 10, 0.5, 0.0)
for j in range(3):
	subplot(gsm[0,j])
	if j == 0:
		text(-0.1, 1.05, "f", transform = gca().transAxes, fontsize = 10, fontweight='bold')
	if j == 1: 
		title("SWR modulation (Mouse 1)", pad = 2)
	noaxis(gca())
	image = newswr[j]
	h, w = image.shape
	rotated_image = np.zeros((h*3, w*3))*np.nan
	rotated_image[h:h*2,w:w*2] = image.copy() + 1.0	
	rotated_image = rotateImage(rotated_image, -angles[pos[i]])
	rotated_image[rotated_image == 0.0] = np.nan
	rotated_image -= 1.0
	tocrop = np.where(~np.isnan(rotated_image))
	rotated_image = rotated_image[tocrop[0].min()-1:tocrop[0].max()+1,tocrop[1].min()-1:tocrop[1].max()+1]			
	imshow(carte_adrien2, extent = bound_adrien, interpolation = 'bilinear', aspect = 'equal')
	im = imshow(rotated_image, extent = bound, alpha = 0.8, aspect = 'equal', cmap = 'bwr')	
	xlim(np.minimum(bound_adrien[0],bound[0]),np.maximum(bound_adrien[1],bound[1]))
	ylim(np.minimum(bound_adrien[2],bound[2]),np.maximum(bound_adrien[3],bound[3]))
	xlabel(str(toplot.loc[m,(j,'start')])+r"ms $\rightarrow$ "+str(toplot.loc[m,(j,'end')])+"ms")

	#colorbar	
	cax = inset_axes(gca(), "4%", "20%",
	                   bbox_to_anchor=(0.75, 0.0, 1, 1),
	                   bbox_transform=gca().transAxes, 
	                   loc = 'lower left')
	cb = colorbar(im, cax = cax, orientation = 'vertical', ticks = [0.25, 0.75])
	# cb.set_label('Burstiness', labelpad = -4)
	# cb.ax.xaxis.set_tick_params(pad = 1)
	# cax.set_title("Cluster 2", fontsize = 6, pad = 2.5)

# #############################################################################
# # G. Population stability
# #############################################################################
subplot(outergs[2,1])
simpleaxis(gca())

# stability HD
data2 =pd.read_hdf("../../figures/figures_articles_v4/figure2/SWR_SCALAR_PRODUCT.h5", 'w')

ax = subplot(gs_bot_right[1,0])
simpleaxis(ax)
m = data2['hd', 'mean'].loc[-500:500]
v = data2['hd', 'sem'].loc[-500:500]
plot(m, color = 'black')
fill_between(m.index.values, m+v, m-v, alpha = 0.5, color = 'grey')
# title("Only hd")

xticks([-500,0,500])
ylabel("Population\nstability")



xlabel("Time from SWRs (ms)")




# fig.subplots_adjust(wspace = 0.3, hspace= 0.3, top = 0.99, bottom = 0.05, right = 0.98, left = 0.08)
fig.subplots_adjust(top = 0.97, bottom = 0.02, right = 0.97, left = 0.03)

savefig("../../figures/figures_articles_v4/figart_2.pdf", dpi = 900, facecolor = 'white')
os.system("evince ../../figures/figures_articles_v4/figart_2.pdf &")





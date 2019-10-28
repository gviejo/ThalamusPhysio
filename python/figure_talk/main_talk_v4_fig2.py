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
# session_ex = 'Mouse17/Mouse17-130129'
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
# lfp_hpc 			= loadLFP(data_directory+session_ex+"/"+session_ex.split("/")[1]+'.eeg', n_channel, hpc_channel, float(fs), 'int16')
# rip_ep,rip_tsd 		= loadRipples(data_directory+session_ex)
# sws_ep 				= loadEpoch(data_directory+session, 'sws')
# lfp_hpc 			= lfp_hpc.restrict(sws_ep)

# data2 = cPickle.load(open("../../figures/figures_articles_v4/figure2/exemple_scalar_product.pickle", 'rb'))
# stab = data2[session_ex]


# figure()
# ax=subplot(211)
# plot(lfp_hpc)
# plot(lfp_hpc.restrict(rip_ep), '-',alpha = 0.8)
# subplot(212,sharex=ax)
# plot(stab)

# show()


# sys.exit()

# sws_ex = nts.IntervalSet(start = 3920, end = 3923, time_units = 's')
# rem_ex = nts.IntervalSet(start = 3191, end = 3195, time_units = 's')
# lfp_sws_ex 		= lfp_hpc.restrict(sws_ex)
# lfp_rem_ex 		= lfp_hpc.restrict(rem_ex)
# spikes_sws_ex 	= pd.concat({n:spikes[n].restrict(sws_ex).isnull()*n for n in spikes}, axis = 1)
# spikes_rem_ex 	= pd.concat({n:spikes[n].restrict(rem_ex).isnull()*n for n in spikes}, axis = 1)



# sys.exit()

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
	fig_height = fig_width*golden_mean*0.7          # height in inches
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

outergs = gridspec.GridSpec(1,1, figure = fig, height_ratios = [1.0], wspace = 0.3, hspace = 0.3)

gstop = gridspec.GridSpecFromSubplotSpec(1,3,subplot_spec = outergs[0,0], width_ratios = [0.15, 0.2, 0.4])#, height_ratios = [0.8, 1.0])#, width_ratios = [0.15, 0.2, 0.4]) #height_ratios = [0.8, 1.0], wspace = 0.2)#, width_ratios=[0.6,0.7,0.1,1.2],wspace = 0.1)

# #############################################
# # A. HISTOLOGY
# #############################################
# # gs = gridspec.GridSpecFromSubplotSpec(2,1,subplot_spec = outergs[0,0], height_ratios = [0.8, 1.0], wspace = 0.2)#, width_ratios=[0.6,0.7,0.1,1.2],wspace = 0.1)
# axA = fig.add_subplot(gstop[0,0])
# # noaxis(axA)
# histo = imread("../../data/histology/Mouse17/Mouse17_2_Slice7_Thalamus_Dapi_2.png")
# imshow(histo, interpolation = 'bilinear',aspect= 'equal')
# text(2500.0, 600.0, "Shanks", rotation = -10, color = 'white', fontsize = 9)
# text(1600.0, 1900.0, "AD", color = 'red', fontsize = 8)
# xticks([], [])
# yticks([], [])
# title("Mouse17")
# # axA.text(-0.0, 1.12, "a", transform = axA.transAxes, fontsize = 10, fontweight='bold')

# #############################################
# # B. MAP AD + HD
# #############################################
# carte_adrien = imread('/home/guillaume/Dropbox (Peyrache Lab)/Peyrache Lab Team Folder/Projects/HPC-Thal/Figures/ATAnatomy_ALL-01.png')
# bound_adrien = (-398/1254, 3319/1254, -(239/1254 - 20/1044), 3278/1254)
# # specific to mouse 17
# subspace = pd.read_hdf("../../figures/figures_articles_v2/figure1/subspace_Mouse17.hdf5")
# data = cPickle.load(open("../../figures/figures_articles_v2/figure1/rotated_images_Mouse17.pickle", 'rb'))
# rotated_images = data['rotated_images']
# new_xy_shank = data['new_xy_shank']
# bound = data['bound']
# data 		= cPickle.load(open("../../data/maps/Mouse17.pickle", 'rb'))
# x 			= data['x']
# y 			= data['y']*-1.0+np.max(data['y'])
# headdir 	= data['headdir']
# xy_pos = new_xy_shank.reshape(len(y), len(x), 2)

# def show_labels(ax):
# 	ax.text(0.68,	1.09,	"AM", 	fontsize = 6.5,  bbox=dict(facecolor='#C9C9C9', edgecolor = 'none', boxstyle='square,pad=-0.1'))
# 	ax.text(1.26,	1.26,	"VA",	fontsize = 6.5,  bbox=dict(facecolor='#C9C9C9', edgecolor = 'none', boxstyle='square,pad=-0.1'))
# 	ax.text(0.92,	2.05,	"AVd",	fontsize = 6.5,  bbox=dict(facecolor='#C9C9C9', edgecolor = 'none', boxstyle='square,pad=-0.1'), rotation = 50)
# 	ax.text(1.11,	1.68,	"AVv",	fontsize = 6.5,  bbox=dict(facecolor='#C9C9C9', edgecolor = 'none', boxstyle='square,pad=-0.1'))
# 	ax.text(1.28,	2.25,	"LD",	fontsize = 6.5,  bbox=dict(facecolor='#C9C9C9', edgecolor = 'none', boxstyle='square,pad=-0.1'))
# 	ax.text(0.42,	2.17,	"sm",	fontsize = 6.5,  bbox=dict(facecolor='#C9C9C9', edgecolor = 'none', boxstyle='square,pad=-0.1'))
# 	ax.text(0.20,	1.89,	"MD",	fontsize = 6.5,  bbox=dict(facecolor='#C9C9C9', edgecolor = 'none', boxstyle='square,pad=-0.1'))
# 	ax.text(-0.06,	1.58,	"PV",	fontsize = 6.5,  bbox=dict(facecolor='#C9C9C9', edgecolor = 'none', boxstyle='square,pad=-0.1'))
# 	ax.text(0.4,	1.5,	"IAD",	fontsize = 6.5,  bbox=dict(facecolor='#C9C9C9', edgecolor = 'none', boxstyle='square,pad=-0.1'), rotation = 52)
# 	return


# ses_ex = ['Mouse17/Mouse17-130129', 'Mouse17/Mouse17-130205']
# ses_mouse17 = np.array([s for s in datasets if 'Mouse17' in s])



# suB = fig.add_subplot(gstop[0,1])
# imshow(carte_adrien, extent = bound_adrien, interpolation = 'bessel', aspect = 'equal')
# i = 1
# m = 'Mouse17'

# # plotting exemples shanks
# # xx = []
# # yy = []
# # for s in ['Mouse17/Mouse17-130201']:
# # 	idx = np.where(s == ses_mouse17)[0][0]
# # 	x = xy_pos[idx,:,0]
# # 	y = xy_pos[idx,:,1]
# # 	scatter(x, y, s = 2, color = 'black')
# # 	xx.append(x[-1])
# # 	yy.append(y[-1])

# # annotate('e', xy = (xx[0]+0.05,yy[0]), xycoords='data', xytext = (xx[0]+0.5, yy[0]-0.06), textcoords='data', arrowprops=dict(arrowstyle="->"))#, horizontalalignment='right', verticalalignment='bottom')
# # annotate('d', xy = (xx[1]+0.05,yy[1]), xycoords='data', xytext = (xx[1]+0.5, yy[1]-0.06), textcoords='data', arrowprops=dict(arrowstyle="->"))#, horizontalalignment='right', verticalalignment='bottom')

# tmp2 = headdir
# tmp2[tmp2<0.05] = 0.0
# scatter(new_xy_shank[:,0], new_xy_shank[:,1], s = 1.5, color = 'black', marker = '.', 
# 	alpha = 1.0, linewidths = 0.5, label = 'shank positions')
# scatter(new_xy_shank[:,0], new_xy_shank[:,1], s = tmp2*7., label = 'HD positions',
# 	color = 'red', marker = 'o', alpha = 0.6)


# plot([2.2,2.2],[0,1], '-', linewidth = 1.3, color = 'black')
# suB.text(2.25, 0.5, "1 mm", rotation = -90)

# show_labels(suB)

# leg = legend(loc = 'lower left', fontsize = 7, framealpha=1.0, bbox_to_anchor=(0.0, -0.09)) #, title = 'HD recording sites', )

# noaxis(suB)
# leg.get_title().set_fontsize(7)
# leg.get_frame().set_facecolor('white')

# annotate('Antero-dorsal (AD)', xy=(0.9,2.4), xytext=(0.9,2.7), xycoords='data', textcoords='data',
# arrowprops=dict(facecolor='black',
# 	shrink=0.05,
# 	headwidth=3,
# 	headlength=2,
# 	width=0.3),
# fontsize = 7, ha = 'center', va = 'bottom')

# # suB.text(-0.1, 1.10, "b", transform = suB.transAxes, fontsize = 10, fontweight = 'bold')


# #############################################################################
# # C. RIPPLES MODULATION
# #############################################################################
# # gs = gridspec.GridSpecFromSubplotSpec(1,2, subplot_spec = outergs[0,1], width_ratios = [0.7, 0.3])

# gsC = gridspec.GridSpecFromSubplotSpec(5,4,subplot_spec = gstop[0,2], wspace = 0.6, width_ratios = [0.07, 1, 1, 1])
# for i, n in enumerate(neurons):	
# 	# spikes	
# 	ax = Subplot(fig, gsC[0:2,i+1])
# 	fig.add_subplot(ax)
# 	simpleaxis(ax)
# 	# if neurons.index(n) == 1:
# 	# 	text(0.5, 1.15,'SWR modulation', horizontalalignment='center', verticalalignment='center', transform=ax.transAxes, fontsize = 8)
# 	if neurons.index(n) == 0:
# 		ylabel("SWR\nevents",  multialignment='center',  labelpad = 1)
# 		# ax.text(-0.9, 1.05, "c", transform = ax.transAxes, fontsize = 10, fontweight='bold')
# 	if neurons.index(n) > 0:
# 		ax.spines['left'].set_visible(False)	
# 		yticks([], [])
# 	ax.spines['bottom'].set_visible(False)	
# 	xticks([], [])
# 	sp = spike_in_swr[n][np.arange(0, 500)]
# 	plot(sp.iloc[:,0:100], '|', markersize = 1, color = colors[neurons.index(n)], mew = 0.5)	

# 	# firing rate	
# 	ax = Subplot(fig, gsC[2,i+1])
# 	fig.add_subplot(ax)
# 	simpleaxis(ax)
# 	ax.spines['bottom'].set_visible(False)	
# 	H0[n] = gaussFilt(H0[n].values, (2,))
# 	plot(H0[n].loc[-500:500], color = colors[neurons.index(n)], label = '', linewidth = 0.7)
# 	plot(Hm[n].loc[-500:500], ':', color = colors[neurons.index(n)], label = 'Jitter', linewidth = 0.7)

# 	axvline(0, color = 'grey', linewidth = 0.5)
# 	xticks([], [])
# 	if neurons.index(n) == 0:
# 		ylabel('Rate \n (Hz)', verticalalignment = 'top', labelpad = 20)
# 		legend(edgecolor = None, facecolor = None, frameon = False, loc = 'lower left', bbox_to_anchor = (0.1, -0.45))	

# 	# Z score	
# 	ax = Subplot(fig, gsC[3:,i+1])
# 	fig.add_subplot(ax)
# 	simpleaxis(ax)
# 	z = pd.DataFrame((H0[n] - Hm[n])/Hstd.loc[n][0])
# 	z['filt'] = gaussFilt(z.values.flatten(), (5,))

# 	plot(z['filt'].loc[-500:500],  color = colors[neurons.index(n)], linewidth = 2)
# 	# xlabel('Time from \n $\mathbf{Sharp\ Waves\ ripples}$ (ms)', fontsize = 8)
# 	if neurons.index(n) == 1:
# 		xlabel('Time from SWRs (ms)', fontsize = 7)
# 	if neurons.index(n) == 0:
# 		ylabel('Modulation\n(z)', verticalalignment = 'bottom')
# 	axvline(0, color = 'grey', linewidth = 0.5)	
# 	if i in [0, 1]:
# 		ylim(-2,2)
# 	# yticks([-1,0,1,2,3])

# #############################################################################
# # D. MEAN SWR HD VS NO-HD
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


################################################################################
gsbottom = gridspec.GridSpecFromSubplotSpec(1,4, subplot_spec = outergs[0,0], wspace = 0.6, width_ratios = [-0.05, 0.4,0.4,0.3])#, width_ratios = [0.011, 0.5, 0.5])

gsm = gridspec.GridSpecFromSubplotSpec(3,1, subplot_spec = gsbottom[0,1], height_ratios = [0.7, 0.4, 0.6], hspace = 0.1)

subplot(gsm[0,0])
simpleaxis(gca())
m = swr_mod[hd_neurons].mean(1)
v = swr_mod[hd_neurons].sem(1)
plot(swr_mod[hd_neurons].mean(1), label = 'HD', color = 'red', linewidth = 1)
fill_between(m.index.values, m-v, m+v, color = 'red', alpha = 0.25, linewidth =0)
m = swr_mod[nohd_neurons].mean(1)
v = swr_mod[nohd_neurons].sem(1)
plot(swr_mod[nohd_neurons].mean(1), label = 'non-HD', color = 'black', linewidth = 1)
fill_between(m.index.values, m-v, m+v, color = 'grey', alpha = 0.25, linewidth = 0)
ylabel("SWRs mod.", labelpad = 2, y = 0.6)
xlim(-500,500)
legend(frameon=False,loc = 'lower left', bbox_to_anchor=(0.7,0.58),handlelength=1,ncol = 1)
xticks([], [])
axvline(0, linestyle = '--', linewidth = 1, alpha = 0.5, color = 'black')
gca().text(-0.3, 1.1, "d", transform = gca().transAxes, fontsize = 10, fontweight='bold')


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
sc = imshow(tmp.T, aspect = 'auto', cmap = "bone", vmin = -0.6, vmax = 0.6)
xticks([0,100,200],[-500,0,500])
xlabel('Time from SWRs (ms)', fontsize = 8)
yticks([len(nohd_neurons)])
cax = inset_axes(gca(), "2%", "30%",
                   bbox_to_anchor=(-0.08, 0.3, 1, 1),
                   bbox_transform=gca().transAxes, 
                   loc = 'lower left')
cb = colorbar(sc, cax = cax, orientation = 'vertical', ticks = [-0.4, 0.4])
# cb.ax.set_title('z')#, labelpad = -24, fontsize = 8)
cb.ax.xaxis.set_tick_params(pad = 1)
cb.ax.yaxis.set_ticks_position('left')

#############################################################################
# E. CORRELATION SHANKS EXEMPLES
#############################################################################
gsc = gridspec.GridSpecFromSubplotSpec(4,1, subplot_spec = gsbottom[0,2], hspace = 0.2, height_ratios = [0.2, 0.2, 0.07, 0.2])

hd_ex = 'Mouse17-130201'
nohd_ex = hd_ex
# nohd_ex = [s for s in datasets if 'Mouse17' in s][0]

# Exemples HD sessions
idx = [s for s in mappings.index.values if hd_ex in s]
hd_idx = mappings.loc[idx].groupby('hd').groups[1]
hd_shank_idx = mappings.loc[hd_idx].groupby('shank').groups
for i,n in enumerate(list(hd_shank_idx.keys())[0:1]):
	print(n)
	subplot(gsc[0,0])
	simpleaxis(gca())
	plot(swr_mod[hd_shank_idx[n]].loc[-500:500], color = 'red', linewidth = 1, alpha = 0.5)
	xticks([])
	title("Session 5 shank 3", loc = 'right', pad = -0.15, fontsize = 7)
	gca().text(-0.2, 1.1, "e", transform = gca().transAxes, fontsize = 10, fontweight='bold')

# Exemples non-HD session
idx = [s for s in mappings.index.values if nohd_ex in s]
idx = np.intersect1d(idx, swr_mod.columns.values)
nohd_idx = mappings.loc[idx].groupby('hd').groups[0]
nohd_shank_idx = mappings.loc[nohd_idx].groupby('shank').groups
for i,n in enumerate([list(nohd_shank_idx.keys())[4]]):
	print(n)
	subplot(gsc[1,0])
	simpleaxis(gca())
	plot(swr_mod[nohd_shank_idx[n]].loc[-500:500], color = 'black', linewidth = 1, alpha = 0.5)
	xticks([-500,0,500])
	xlabel("Time from SWRs (ms)", fontsize = 7, labelpad = -0.05)
	title("shank 5", loc = 'right', pad = -2.45, fontsize = 7)
	ylabel("SWRs mod.")
	gca().yaxis.set_label_coords(-0.15, 1.0) 

# Correlation
hd_corr = {}
for n in hd_shank_idx.keys():
	swrmod = swr_mod[hd_shank_idx[n]].values
	C = np.corrcoef(swrmod.T)
	hd_corr[n] = C[np.triu_indices_from(C,1)]
nohd_corr = {}
for n in nohd_shank_idx.keys():	
	if len(nohd_shank_idx[n]) > 1:
		swrmod = swr_mod[nohd_shank_idx[n]].values
		C = np.corrcoef(swrmod.T)
		nohd_corr[n] = C[np.triu_indices_from(C,1)]

subplot(gsc[3,0])
simpleaxis(gca())
dispersion = 0.05
side = 0.1
markersize = 3
for n in hd_corr.keys():
	x = np.ones_like(hd_corr[n])*n+np.random.randn(len(hd_corr[n]))*dispersion - side
	plot(x, hd_corr[n], 'o', color = 'red', markersize = markersize)
for n in nohd_corr.keys():
	x = np.ones_like(nohd_corr[n])*n+np.random.randn(len(nohd_corr[n]))*dispersion + side
	plot(x, nohd_corr[n], 'o', color = 'black', markersize = markersize)
ylabel("Pair\ncorrelation")
xlabel("Shanks")
xticks(np.unique(np.hstack(list(hd_corr.keys())+list(nohd_corr.keys()))), np.unique(np.hstack(list(hd_corr.keys())+list(nohd_corr.keys())))+1)


#############################################################################
# F. CORRELATION SWR MOD SHANKS
#############################################################################
gsf = gridspec.GridSpecFromSubplotSpec(3,1, subplot_spec = gsbottom[0,3], hspace = 0.5, height_ratios = [0.1, 0.5, 0.05])

data3 = cPickle.load(open('../../figures/figures_articles_v4/figure2/SWR_MOD_CORRELATION_SHANKS.pickle', 'rb'))

bins = np.linspace(-1,1,40)

hd_corr_shanks = data3['shank']['hd']
nohd_corr_shanks = data3['shank']['nohd']
subplot(gsf[1,0])
simpleaxis(gca())
hist(nohd_corr_shanks, bins=bins, weights=np.ones_like(nohd_corr_shanks)/float(len(nohd_corr_shanks)), alpha = 0.5, color = 'black', histtype='stepfilled', label = 'non-HD')
hist(hd_corr_shanks, bins=bins, weights=np.ones_like(hd_corr_shanks)/float(len(hd_corr_shanks)), alpha = 0.5, color = 'red', histtype='stepfilled', label = 'HD')
xlabel("Same shank\npair correlation")
ylabel("%")
yticks([0, 0.05, 0.1], [0, 5, 10])
legend(loc = 'lower left', fontsize = 7, framealpha=0.0, bbox_to_anchor=(0.35, 1.0)) #, title = 'HD recording sites', )
# title("Pairs/Shanks")
gca().text(-0.2, 1.1, "f", transform = gca().transAxes, fontsize = 10, fontweight='bold')


# fig.subplots_adjust(wspace = 0.3, hspace= 0.3, top = 0.99, bottom = 0.05, right = 0.98, left = 0.08)
fig.subplots_adjust(right = 0.97, left = 0.01)

savefig("../../figures/figures_articles_v4/figart_2.pdf", dpi = 900, facecolor = 'white')
os.system("evince ../../figures/figures_articles_v4/figart_2.pdf &")





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



data_directory 	= '/mnt/DataGuillaume/MergedData/'
datasets 		= np.loadtxt(data_directory+'datasets_ThalHpc.list', delimiter = '\n', dtype = str, comments = '#')
session 		= 'Mouse12/Mouse12-120810'
neurons 		= [session.split("/")[1]+"_"+str(u) for u in [38,37,40]]

path_snippet 	= "../../figures/figures_articles_v2/figure2/"
store 			= pd.HDFStore(path_snippet+'snippet_'+session.split("/")[1]+'.h5')
phase_spike 	= pd.HDFStore(path_snippet+'spikes_'+session.split("/")[1]+'.h5')
spike_in_swr 	= pd.HDFStore(path_snippet+'spikes_in_swr_'+session.split("/")[1]+'.h5')		
modulations 	= pd.HDFStore(path_snippet+'modulation_theta2_swr_Mouse17.h5')

H0 = store['H0']
Hm = store['Hm']
Hstd = store['Hstd']

# lfp_filt_hpc_swr = store['lfp_filt_hpc_swr']
# lfp_filt_hpc_theta = store['lfp_filt_hpc_theta']
# lfp_hpc_swr = store['lfp_hpc_swr']
# lfp_hpc_theta = store['lfp_hpc_theta']
phase_spike_theta = {
'Mouse12-120810_37':store['phase_spike_theta_Mouse12-120810_37'],
'Mouse12-120810_38':store['phase_spike_theta_Mouse12-120810_38'],
'Mouse12-120810_40':store['phase_spike_theta_Mouse12-120810_40']	
}
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
colors = ['#231f20', '#707174', '#abacad']

fig = figure(figsize = figsize(1.0))

outergs = gridspec.GridSpec(4,2, figure = fig, height_ratios = [0.9,1.0,-0.07,1])

#############################################################################
# A. B LFP EXEMPLE
#############################################################################
gsEx = gridspec.GridSpecFromSubplotSpec(2,2, subplot_spec = outergs[0,:], hspace = 0, wspace = 0, height_ratios = [0.4,1])

lbs = ['a', 'b']
titles = ['REM', 'NREM']
# 1 lfps
for i, lfp in zip(range(2),[lfp_rem_ex, lfp_sws_ex]):
	# ax = Subplot(fig, )
	ax = fig.add_subplot(gsEx[0,i])
	noaxis(ax)
	plot(lfp, color = 'black', linewidth = 0.4)
	title(titles[i], pad = -2)
	if i == 0:
		text(-0.15, 0.9, lbs[i], transform=ax.transAxes, fontsize = 9, fontweight='bold')
		ylabel("CA1")
	else:
		text(-0.01, 0.9, lbs[i], transform=ax.transAxes, fontsize = 9,  fontweight='bold')
		plot(rip_tsd.index.values, [lfp.max()]*2, '*', color = 'red', markersize = 5)
# 2 spikes
for i, spikes in zip(range(2), [spikes_rem_ex, spikes_sws_ex]):
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
# C. THETA PHASE MODULATION
#############################################################################
# axC = fig.add_subplot(2,2,1)
# gsC = gridspec.GridSpecFromSubplotSpec(2,4,subplot_spec = outergs[1,0], wspace = 0.6,  width_ratios = [1,1,1,0.1])
gsC = gridspec.GridSpecFromSubplotSpec(2,3,subplot_spec = outergs[1,0], wspace = 0.6)

for i, n in enumerate(neurons):	
	# spikes
	# ax = fig.add_subplot(gsC[0,i])
	# ax = Subplot(fig, )
	ax = fig.add_subplot(gsC[0,i])
	simpleaxis(ax)
	if neurons.index(n) == 1:
		text(0.5, 1.17,'Theta phase modulation', horizontalalignment='center', verticalalignment='center', transform=ax.transAxes, fontsize = 8)
	if neurons.index(n) > 0:
		ax.spines['left'].set_visible(False)	
		yticks([], [])
	if neurons.index(n) == 0:
		ylabel("Theta cycles")
		ax.text(-0.76, 1.14, "c", transform = ax.transAxes, fontsize = 9, fontweight='bold')
	ax.spines['bottom'].set_visible(False)
	sp = phase_spike[n]
	plot(sp.iloc[0:500], '|', markersize = 2, color = colors[neurons.index(n)], mew = 0.5)	
	xticks([], [])
	# xticks([0, 2*np.pi], ['0', '$2\pi$'])
	# xlabel("phase (rad)")
	# polar plot
	# ax = fig.add_subplot(gsC[1,i], projection = 'polar')	
	# axp = Subplot(fig, )
	axp = fig.add_subplot(gsC[1,i])
	# fig.add_subplot(axp, projection = 'polar')	
	simpleaxis(axp)
	tmp = phase_spike_theta[n].values
	tmp += 2*np.pi
	tmp %= 2*np.pi
	axp.hist(tmp,20, color = colors[neurons.index(n)], density = True, histtype='stepfilled')
	# a, b = np.histogram(tmp, 20)
	# polar(np.linspace(0, 2*np.pi, 20), a)
	# xlabel("Neuron "+str(neurons.index(n)+1), fontsize = 8)
	# xticks(np.arange(0, 2*np.pi, np.pi/4), ['0', '', '$\pi/2$', '', '$\pi$', '', '$3\pi/2$',''])
	xticks([0, 2*np.pi], ['0', '$2\pi$'])
	# yticks([])	
	# axp.tick_params(axis='x', pad = -5)
	# grid(linestyle = '--')
	# axp.yaxis.grid(False)
	if i ==1 : xlabel("phase (rad)")

#############################################################################
# D. RIPPLES MODULATION
#############################################################################
# axD = fig.add_subplot(2,2,2)
gsD = gridspec.GridSpecFromSubplotSpec(5,3, subplot_spec = outergs[1,1], wspace =0.6)
for i, n in enumerate(neurons):	
	# spikes
	# ax = fig.add_subplot(gsD[0:2,i])
	ax = Subplot(fig, gsD[0:2,i])
	fig.add_subplot(ax)
	simpleaxis(ax)
	if neurons.index(n) == 1:
		text(0.5, 1.23,'SWR modulation', horizontalalignment='center', verticalalignment='center', transform=ax.transAxes, fontsize = 8)
	if neurons.index(n) == 0:
		ylabel("SWR\nevents",  multialignment='center')
		ax.text(-0.85, 1.18, "d", transform = ax.transAxes, fontsize = 9, fontweight='bold')
	if neurons.index(n) > 0:
		ax.spines['left'].set_visible(False)	
		yticks([], [])
	ax.spines['bottom'].set_visible(False)	
	xticks([], [])
	sp = spike_in_swr[n][np.arange(0, 500)]
	plot(sp.iloc[:,0:300], '|', markersize = 2, color = colors[neurons.index(n)], mew = 0.5)	

	# firing rate
	# ax = fig.add_subplot(gsD[2,i])
	ax = Subplot(fig, gsD[2,i])
	fig.add_subplot(ax)
	simpleaxis(ax)
	ax.spines['bottom'].set_visible(False)	
	H0[n] = gaussFilt(H0[n].values, (2,))
	plot(H0[n].loc[-500:500], color = colors[neurons.index(n)], label = '', linewidth = 0.7)
	plot(Hm[n].loc[-500:500], ':', color = colors[neurons.index(n)], label = 'Jitter', linewidth = 0.7)

	axvline(0, color = 'grey', linewidth = 0.5)
	xticks([], [])
	if neurons.index(n) == 0:
		ylabel('Rate \n (Hz)', verticalalignment = 'top', labelpad = 18)
		legend(edgecolor = None, facecolor = None, frameon = False, loc = 'lower left', bbox_to_anchor = (0.4, -0.6))	

	# Z score
	# ax = fig.add_subplot(gsD[3:,i])
	ax = Subplot(fig, gsD[3:,i])
	fig.add_subplot(ax)
	simpleaxis(ax)
	z = pd.DataFrame((H0[n] - Hm[n])/Hstd.loc[n][0])
	z['filt'] = gaussFilt(z.values.flatten(), (10,))

	plot(z['filt'].loc[-500:500],  color = colors[neurons.index(n)], linewidth = 2)
	# xlabel('Time from \n $\mathbf{Sharp\ Waves\ ripples}$ (ms)', fontsize = 8)
	if neurons.index(n) == 1:
		xlabel('Time from SWRs (ms)', fontsize = 7)
	if neurons.index(n) == 0:
		ylabel('Modulation\n(z)', verticalalignment = 'bottom')
	axvline(0, color = 'grey', linewidth = 0.5)	
	ylim(-2,3)
	yticks([-1,0,1,2,3])

###############################################################################
# E. MAPS THETA
###############################################################################
# axmap = subplot(2,1)
# axmap = 
# noaxis(axmap)
# axE = fig.add_subplot(2,4,5)
pos_nb = (-0.15, 1.2)
outer = gridspec.GridSpecFromSubplotSpec(2,6, subplot_spec = outergs[3,:], wspace = 0.0, hspace = 0.5, width_ratios = [0.5,0.5,0.5,0.3,0.8,0.1])
axE = fig.add_subplot(outer[:,0])


noaxis(axE)
tmp = modulations['theta']
bound = (tmp.columns[0], tmp.columns[-1], tmp.index[-1], tmp.index[0])
im = imshow(tmp, extent = bound, alpha = 0.8, aspect = 'equal', cmap = 'GnBu', vmin = 0, vmax = 1)
imshow(carte_adrien2, extent = bound_adrien, interpolation = 'bilinear', aspect = 'equal')
# title("Theta spatial modulation", fontsize = 7, y = 1.3)
axE.text(0.06, 1.05, "Theta spatial\n modulation", transform = axE.transAxes, fontsize = 8, multialignment='center')
#colorbar	
cax = inset_axes(axE, "40%", "8%",
                   bbox_to_anchor=(0.2, -0.25, 1, 1),
                   bbox_transform=axE.transAxes, 
                   loc = 'lower left')
cb = colorbar(im, cax = cax, orientation = 'horizontal', ticks = [0,1])
# cb.set_label('Density (p < 0.05)' , labelpad = -0)
cb.ax.xaxis.set_tick_params(pad = 1)
cax.set_title("Density (p < 0.01)", fontsize = 7, pad = 2.5)
axE.text(pos_nb[0],pos_nb[1], "e", transform = axE.transAxes, fontsize = 9, fontweight='bold')

###############################################################################
# F. MAPS SWR POS
###############################################################################
# axF = fig.add_subplot(2,4,6)
axF = fig.add_subplot(outer[:,1])
noaxis(axF)
tmp = modulations['pos_swr']
bound = (tmp.columns[0], tmp.columns[-1], tmp.index[-1], tmp.index[0])
im = imshow(tmp, extent = bound, alpha = 0.8, aspect = 'equal', cmap = 'Reds', vmin = 0, vmax = 1)
imshow(carte_adrien2, extent = bound_adrien, interpolation = 'bilinear', aspect = 'equal')
#colorbar	
cax = inset_axes(axF, "40%", "8%",
                   bbox_to_anchor=(0.2, -0.25, 1, 1),
                   bbox_transform=axF.transAxes, 
                   loc = 'lower left')
cb = colorbar(im, cax = cax, orientation = 'horizontal', ticks = [0,1])
# cb.set_label('Density (p < 0.05)' , labelpad = -0)
cb.ax.xaxis.set_tick_params(pad = 1)
cax.set_title("Density $z_{0 ms} > $"+p_60, fontsize = 7, pad = 2.5)

axF.text(pos_nb[0],pos_nb[1], "f", transform = axF.transAxes, fontsize = 9, fontweight='bold')
# title("Ripples spatial modulation", fontsize = 7, y = 1.3)
axF.text(0.4, 1.05, "SWR spatial modulation", transform = axF.transAxes, fontsize = 8)

###############################################################################
# G. MAPS NEG SWR
###############################################################################
# axF = fig.add_subplot(2,4,8)
axG = fig.add_subplot(outer[:,2])
noaxis(axG)
tmp = modulations['neg_swr']
im = imshow(tmp, extent = bound, alpha = 0.8, aspect = 'equal', cmap = 'Reds', vmin = 0, vmax = 1)
imshow(carte_adrien2, extent = bound_adrien, interpolation = 'bilinear', aspect = 'equal')
#colorbar	
cax = inset_axes(axG, "40%", "8%",
                   bbox_to_anchor=(0.2, -0.25, 1, 1),
                   bbox_transform=axG.transAxes, 
                   loc = 'lower left')
cb = colorbar(im, cax = cax, orientation = 'horizontal', ticks = [0,1])
# cb.set_label('Density (p < 0.05)' , labelpad = -0)
cb.ax.xaxis.set_tick_params(pad = 1)
cax.set_title("$z_{0 ms} <$"+p_40, fontsize = 7, pad = 2.5)
axG.text(pos_nb[0],pos_nb[1], "g", transform = axG.transAxes, fontsize = 9, fontweight='bold')

###############################################################################
# H. THETA RIP MODULATION / NUCLEUS
###############################################################################
# axH = fig.add_subplot(2,4,8)
axH = fig.add_subplot(outer[0,4])
simpleaxis(axH)
# theta_mod, theta_ses 	= loadThetaMod('/mnt/DataGuillaume/MergedData/THETA_THAL_mod.pickle', datasets, return_index=True)
# theta 					= pd.DataFrame(	index = theta_ses['rem'], 
# 									columns = ['phase', 'pvalue', 'kappa'],
# 									data = theta_mod['rem'])
rippower 				= pd.read_hdf("../../figures/figures_articles_v2/figure2/power_ripples_2.h5")
# nucleus = np.unique(mappings['nucleus'])
df = pd.DataFrame(index = nucleus, columns = pd.MultiIndex.from_product([['theta', 'rip'], ['mean', 'sem']]))

theta2 = pd.read_hdf("/mnt/DataGuillaume/MergedData/THETA_THAL_mod_2.h5")

theta = theta2['rem']



for n in nucleus:	
	neurons = mappings[mappings['nucleus'] == n].index.values
	# df.loc[n,('theta','mean')] 	= theta.loc[neurons,'kappa'].mean(skipna=True)
	# df.loc[n,('theta','sem')] 	= theta.loc[neurons,'kappa'].sem(skipna=True)
	# df.loc[n,('rip','mean')] 	= rippower.loc[neurons].mean(skipna=True)
	# df.loc[n,('rip','sem')] 	= rippower.loc[neurons].sem(skipna=True)
	df.loc[n,('theta','mean')] 	= theta.reindex(index = neurons, columns = ['kappa']).mean(skipna=True)[0]
	df.loc[n,('theta','sem')] 	= theta.reindex(index = neurons, columns = ['kappa']).sem(skipna=True)[0]
	df.loc[n,('rip','mean')] 	= rippower.reindex(index = neurons).mean(skipna=True)
	df.loc[n,('rip','sem')] 	= rippower.reindex(index = neurons).sem(skipna=True)	

df = df.sort_values([('theta', 'mean')])

# df = df.drop(['sm', 'U'])
labels = ['Theta', 'SWR']
axHH = axH.twinx()
axHH.spines['top'].set_visible(False)
axH.spines['top'].set_visible(False)
axes = [axH, axHH]
# colors = ['royalblue', 'firebrick']
mks = ['o-', 'o-']
colors = [matplotlib.colors.to_hex(cm.GnBu(0.8)), matplotlib.colors.to_hex(cm.Reds(0.6))]
offset = 0.0
for i, k in enumerate(['theta', 'rip']):
	m = df[k]['mean'].values.astype('float32')
	s = df[k]['sem'].values.astype('float32')
	# axes[i].plot(m, np.arange(len(df)), mks[i], label = labels[i], markersize = 3, linewidth = 1, color = colors[i])
	# axes[i].fill_betweenx(np.arange(len(df)), m-s, m+s, alpha = 0.3, color = colors[i])
	axes[i].bar(np.arange(len(df))+offset, m, width = 0.4, yerr = s, label = labels[i], color = colors[i], edgecolor = 'black')
	offset +=0.4
	
axH.set_ylim(0.08, 0.32)
axHH.set_ylim(1.26, 1.9)
axH.set_xticks(np.arange(len(df))+0.2)
axH.set_xticklabels(df.index.values)
axH.set_ylabel("Theta\nmodulation ($\\kappa$)", color = colors[0], labelpad = 3.0)
axH.tick_params(axis='y', colors=colors[0], pad = 1)
axHH.set_ylabel("SWR energy (|z|)", color = colors[1], labelpad = 11, rotation = -90)
axHH.tick_params(axis='y', colors=colors[1], pad = 0)
axHH.set_yticks([1.4, 1.8])
axH.text(-0.35, 1.1, "h", transform = axH.transAxes, fontsize = 9, fontweight='bold')
axH.set_xlim(-0.2,7.6)


########################################################################
# I. SCORE CLASSIFICATION
########################################################################
axI = fig.add_subplot(outer[1,4:])
simpleaxis(gca())
gca().text(-0.3, 1.1, "i", transform = gca().transAxes, fontsize = 9, fontweight='bold')
xlabel("Nuclei")
ylabel("Classification\nscore", multialignment = 'center', labelpad = 3.0)
# title("SWR classification")
tmp = mean_score[('score', 'swr', 'mean')]
tmp2 = mean_score[('shuffle', 'swr', 'mean')]
tmp3 = (tmp-tmp2)/(1-tmp2)
tmp3 = tmp3.sort_values(ascending = False)
order = tmp3.index.values

# bar(np.arange(len(tmp)), tmp2.values, linewidth = 1, color = 'none', edgecolor = 'black', linestyle = '--')
bar(np.arange(len(tmp3)), tmp3.values, yerr = mean_score.loc[order,('score','swr','sem')], 
	linewidth = 1, color = 'none', edgecolor = 'black')
xticks(np.arange(mean_score.shape[0]), order)
# axhline(1/8, linestyle = '--', color = 'black', linewidth = 0.5)
# yticks([0, 0.2,0.4], [0, 20,40])



fig.subplots_adjust(wspace = 0.3, hspace= 0.3, top = 0.99, bottom = 0.05, right = 0.98, left = 0.08)

savefig("../../figures/figures_articles_v3/figart_2.pdf", dpi = 900, facecolor = 'white')
os.system("evince ../../figures/figures_articles_v3/figart_2.pdf &")





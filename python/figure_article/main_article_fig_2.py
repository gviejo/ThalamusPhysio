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
# start_rip, end_rip = (92150000,92950000)
# start_rip, end_rip = (92550000,92880000)
# start_theta, end_theta = (5843120000,5844075000)

path_snippet 	= "../../figures/figures_articles/figure2/"
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

carte38_mouse17 = imread('../../figures/mapping_to_align/paxino/paxino_38_mouse17_2.png')
bound_map_38 = (-2336/1044, 2480/1044, 0, 2663/1044)


###############################################################################################################
###############################################################################################################
# PLOT
###############################################################################################################
def figsize(scale):
	fig_width_pt = 483.69687                         # Get this from LaTeX using \the\textwidth
	inches_per_pt = 1.0/72.27                       # Convert pt to inch
	golden_mean = (np.sqrt(5.0)-1.0)/2.0            # Aesthetic ratio (you could change this)
	fig_width = fig_width_pt*inches_per_pt*scale    # width in inches
	fig_height = fig_width*golden_mean*1.2          # height in inches
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
from mpl_toolkits.axes_grid.inset_locator import inset_axes
colors = ['#444b6e', '#708b75', '#9ab87a']

fig = figure(figsize = figsize(1.0))

# outer = gridspec.GridSpec(2,1)

#############################################################################
# A. THETA PHASE MODULATION
#############################################################################
axA = fig.add_subplot(2,2,1)
gsA = gridspec.GridSpecFromSubplotSpec(2,4,subplot_spec = axA, wspace = 0.6,  width_ratios = [1,1,1,0.1])

for i, n in enumerate(neurons):	
	# spikes
	ax = subplot(gsA[0,i])
	simpleaxis(ax)
	if neurons.index(n) == 1:
		text(0.5, 1.17,'Theta phase modulation', horizontalalignment='center', verticalalignment='center', transform=ax.transAxes, fontsize = 7)
	if neurons.index(n) > 0:
		ax.spines['left'].set_visible(False)	
		yticks([], [])
	if neurons.index(n) == 0:
		ylabel("Theta cycles")
		ax.text(-0.6, 1.14, "A", transform = ax.transAxes, fontsize = 8)
	sp = phase_spike[n]
	plot(sp, '|', markersize = 1, color = colors[neurons.index(n)])
	xticks([0, 2*np.pi], ['0', '$2\pi$'])
	xlabel("phase (rad)")
	# polar plot
	ax = subplot(gsA[1,i], projection = 'polar')	
	tmp = phase_spike_theta[n].values
	tmp += 2*np.pi
	tmp %= 2*np.pi
	hist(tmp,20, color = colors[neurons.index(n)])
	# xlabel("Neuron "+str(neurons.index(n)+1), fontsize = 8)
	xticks(np.arange(0, 2*np.pi, np.pi/4), ['0', '', '$\pi/2$', '', '$\pi$', '', '$3\pi/2$',''])
	yticks([])	
	ax.tick_params(axis='x', pad = -5)
	grid(linestyle = '--')
	ax.yaxis.grid(False)


#############################################################################
# B. RIPPLES MODULATION
#############################################################################
axB = fig.add_subplot(2,2,2)
gsB = gridspec.GridSpecFromSubplotSpec(5,3, subplot_spec = axB, wspace =0.6)
for i, n in enumerate(neurons):	
	# spikes
	ax = subplot(gsB[0:2,i])
	simpleaxis(ax)
	if neurons.index(n) == 1:
		text(0.5, 1.23,'Ripples modulation', horizontalalignment='center', verticalalignment='center', transform=ax.transAxes, fontsize = 7)
	if neurons.index(n) == 0:
		ylabel("Ripple events")
		ax.text(-0.6, 1.18, "B", transform = ax.transAxes, fontsize = 8)
	if neurons.index(n) > 0:
		ax.spines['left'].set_visible(False)	
		yticks([], [])
	ax.spines['bottom'].set_visible(False)	
	xticks([], [])
	sp = spike_in_swr[n][np.arange(0, 500)]
	plot(sp, '|', markersize = 1, color = colors[neurons.index(n)])

	# firing rate
	ax = subplot(gsB[2,i])
	simpleaxis(ax)
	ax.spines['bottom'].set_visible(False)	
	H0[n] = gaussFilt(H0[n].values, (2,))
	plot(H0[n].loc[-500:500], color = colors[neurons.index(n)], label = '', linewidth = 0.7)
	plot(Hm[n].loc[-500:500], ':', color = colors[neurons.index(n)], label = 'Jitter', linewidth = 0.7)

	axvline(0, color = 'grey', linewidth = 0.5)
	xticks([], [])
	if neurons.index(n) == 0:
		ylabel('Firing \n rate (Hz)', verticalalignment = 'top', labelpad = 13.4)
		legend(edgecolor = None, facecolor = None, frameon = False, loc = 'lower left', bbox_to_anchor = (0.4, -0.5))	

	# Z score
	ax = subplot(gsB[3:,i])
	simpleaxis(ax)
	z = pd.DataFrame((H0[n] - Hm[n])/Hstd.loc[n][0])
	z['filt'] = gaussFilt(z.values.flatten(), (10,))

	plot(z['filt'].loc[-500:500],  color = colors[neurons.index(n)], linewidth = 2)
	# xlabel('Time from \n $\mathbf{Sharp\ Waves\ ripples}$ (ms)', fontsize = 8)
	if neurons.index(n) == 1:
		xlabel('Time from ripples (ms)', fontsize = 6)
	if neurons.index(n) == 0:
		ylabel('Modulation (z)', verticalalignment = 'bottom')
	axvline(0, color = 'grey', linewidth = 0.5)	
	ylim(-2,3)
	yticks([-1,0,1,2,3])

###############################################################################
# C. MAPS THETA
###############################################################################
axmap = fig.add_subplot(2,1,2)
outer = gridspec.GridSpecFromSubplotSpec(1,6, subplot_spec = axmap, width_ratios = [1,0.1,1,1,0.1,1])
noaxis(axmap)
# axC = fig.add_subplot(2,4,5)
axC = fig.add_subplot(outer[0,0])
cut_bound_map = (-86/1044, 2480/1044, 0, 2663/1044)

noaxis(axC)
tmp = modulations['theta']
bound = (tmp.columns[0], tmp.columns[-1], tmp.index[-1], tmp.index[0])
im = imshow(tmp, extent = bound, alpha = 0.8, aspect = 'equal', cmap = 'GnBu', vmin = 0, vmax = 1)
imshow(carte38_mouse17[:,2250:], extent = cut_bound_map, interpolation = 'bilinear', aspect = 'equal')
# title("Theta spatial modulation", fontsize = 7, y = 1.3)
axC.text(-0.05, 1.4, "Theta spatial modulation", transform = axC.transAxes, fontsize = 8)
#colorbar	
cax = inset_axes(axC, "40%", "4%",
                   bbox_to_anchor=(0.2, 1.08, 1, 1),
                   bbox_transform=axC.transAxes, 
                   loc = 'lower left')
cb = colorbar(im, cax = cax, orientation = 'horizontal', ticks = [0,1])
# cb.set_label('Density (p < 0.05)' , labelpad = -0)
cb.ax.xaxis.set_tick_params(pad = 1)
cax.set_title("Density (p < 0.01)", fontsize = 6, pad = 2.5)
pos_nb = (-0.15, 1.1)
axC.text(pos_nb[0],pos_nb[1], "C", transform = axC.transAxes, fontsize = 8)
###############################################################################
# D. MAPS SWR POS
###############################################################################
# axD = fig.add_subplot(2,4,6)
axD = fig.add_subplot(outer[0,2])
noaxis(axD)
tmp = modulations['pos_swr']
im = imshow(tmp, extent = bound, alpha = 0.8, aspect = 'equal', cmap = 'Reds', vmin = 0, vmax = 1)
imshow(carte38_mouse17[:,2250:], extent = cut_bound_map, interpolation = 'bilinear', aspect = 'equal')
#colorbar	
cax = inset_axes(axD, "40%", "4%",
                   bbox_to_anchor=(0.2, 1.08, 1, 1),
                   bbox_transform=axD.transAxes, 
                   loc = 'lower left')
cb = colorbar(im, cax = cax, orientation = 'horizontal', ticks = [0,1])
# cb.set_label('Density (p < 0.05)' , labelpad = -0)
cb.ax.xaxis.set_tick_params(pad = 1)
cax.set_title("Density $t_{0 ms} > P_{60}$", fontsize = 6, pad = 2.5)
axD.text(pos_nb[0],pos_nb[1], "D", transform = axD.transAxes, fontsize = 8)
# title("Ripples spatial modulation", fontsize = 7, y = 1.3)
axD.text(0.4, 1.4, "Ripples spatial modulation", transform = axD.transAxes, fontsize = 8)
# ###############################################################################
# # E. MAPS POS SWR
# ###############################################################################
# # axE = fig.add_subplot(2,4,7)
# axE = fig.add_subplot(outer[0,3])
# noaxis(axE)
# tmp = modulations['neu_swr']
# im = imshow(tmp, extent = bound, alpha = 0.8, aspect = 'equal', cmap = 'Greens', vmin = 0, vmax = 1)
# imshow(carte38_mouse17[:,2250:], extent = cut_bound_map, interpolation = 'bilinear', aspect = 'equal')
# title("Ripples spatial modulation", fontsize = 7, y = 1.3)
# #colorbar	
# cax = inset_axes(axE, "40%", "4%",
#                    bbox_to_anchor=(0.2, 1.08, 1, 1),
#                    bbox_transform=axE.transAxes, 
#                    loc = 'lower left')
# cb = colorbar(im, cax = cax, orientation = 'horizontal', ticks = [0,1])
# # cb.set_label('Density (p < 0.05)' , labelpad = -0)
# cb.ax.xaxis.set_tick_params(pad = 1)
# cax.set_title("$P_{40} < t_{0 ms} < P_{60}$", fontsize = 6, pad = 2.5)
# axE.text(pos_nb[0],pos_nb[1], "E", transform = axE.transAxes, fontsize = 8)
###############################################################################
# E. MAPS NEG SWR
###############################################################################
# axF = fig.add_subplot(2,4,8)
axE = fig.add_subplot(outer[0,3])
noaxis(axE)
tmp = modulations['neg_swr']
im = imshow(tmp, extent = bound, alpha = 0.8, aspect = 'equal', cmap = 'Greens', vmin = 0, vmax = 1)
imshow(carte38_mouse17[:,2250:], extent = cut_bound_map, interpolation = 'bilinear', aspect = 'equal')
#colorbar	
cax = inset_axes(axE, "40%", "4%",
                   bbox_to_anchor=(0.2, 1.08, 1, 1),
                   bbox_transform=axE.transAxes, 
                   loc = 'lower left')
cb = colorbar(im, cax = cax, orientation = 'horizontal', ticks = [0,1])
# cb.set_label('Density (p < 0.05)' , labelpad = -0)
cb.ax.xaxis.set_tick_params(pad = 1)
cax.set_title("$t_{0 ms} < P_{40}$", fontsize = 6, pad = 2.5)
axE.text(pos_nb[0],pos_nb[1], "E", transform = axE.transAxes, fontsize = 8)

###############################################################################
# F. THETA RIP MODULATION / NUCLEUS
###############################################################################
# axF = fig.add_subplot(2,4,8)
axF = fig.add_subplot(outer[0,5])
simpleaxis(axF)
# theta_mod, theta_ses 	= loadThetaMod('/mnt/DataGuillaume/MergedData/THETA_THAL_mod.pickle', datasets, return_index=True)
# theta 					= pd.DataFrame(	index = theta_ses['rem'], 
# 									columns = ['phase', 'pvalue', 'kappa'],
# 									data = theta_mod['rem'])
rippower 				= pd.read_hdf("../../figures/figures_articles/figure2/power_ripples_2.h5")
mappings = pd.read_hdf("/mnt/DataGuillaume/MergedData/MAPPING_NUCLEUS.h5")
nucleus = np.unique(mappings['nucleus'])
df = pd.DataFrame(index = nucleus, columns = pd.MultiIndex.from_product([['theta', 'rip'], ['mean', 'sem']]))

theta2 = pd.read_hdf("/mnt/DataGuillaume/MergedData/THETA_THAL_mod_2.h5")

theta = theta2['rem']

for n in nucleus:
	neurons = mappings[mappings['nucleus'] == n].index.values
	df.loc[n,('theta','mean')] = theta.loc[neurons,'kappa'].mean(skipna=True)
	df.loc[n,('theta','sem')] = theta.loc[neurons,'kappa'].sem(skipna=True)
	df.loc[n,('rip','mean')] = rippower.loc[neurons].mean(skipna=True)
	df.loc[n,('rip','sem')] = rippower.loc[neurons].sem(skipna=True)

df = df.sort_values([('rip', 'mean')])

df = df.drop(['sm', 'U'])
labels = ['Theta', 'Ripples']
axFF = axF.twiny()
axes = [axF, axFF]
colors = ['royalblue', 'firebrick']
for i, k in enumerate(['theta', 'rip']):
	m = df[k]['mean'].values.astype('float32')
	s = df[k]['sem'].values.astype('float32')
	axes[i].plot(m, np.arange(len(df)), 'o-', label = labels[i], markersize = 3, linewidth = 1, color = colors[i])
	axes[i].fill_betweenx(np.arange(len(df)), m-s, m+s, alpha = 0.3, color = colors[i])
yticks(np.arange(len(df)), df.index.values)
axF.set_xlabel(r"Theta modulation ($\kappa$)", color = colors[0])
axFF.set_xlabel("Ripples modulation (|z|)", color = colors[1])

axF.text(-0.3, 1.0, "F", transform = axF.transAxes, fontsize = 8)


fig.subplots_adjust(hspace= 1)

savefig("../../figures/figures_articles/figart_2.pdf", dpi = 900, bbox_inches = 'tight', facecolor = 'white')
os.system("evince ../../figures/figures_articles/figart_2.pdf &")





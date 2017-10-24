#!/usr/bin/env python


import numpy as np
import pandas as pd
# from matplotlib.pyplot import plot,show,draw
import scipy.io
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
start_rip, end_rip = (92550000,92880000)
start_theta, end_theta = (5843120000,5844075000)

path_snippet 	= "../figures/figures_articles/figure1/"
store 			= pd.HDFStore(path_snippet+'snippet_'+session.split("/")[1]+'.h5')

H0 = store['H0']
Hm = store['Hm']
Hstd = store['Hstd']
lfp_filt_hpc_swr = store['lfp_filt_hpc_swr']
lfp_filt_hpc_theta = store['lfp_filt_hpc_theta']
lfp_hpc_swr = store['lfp_hpc_swr']
lfp_hpc_theta = store['lfp_hpc_theta']
phase_spike_theta = {
'Mouse12-120810_37':store['phase_spike_theta_Mouse12-120810_37'],
'Mouse12-120810_38':store['phase_spike_theta_Mouse12-120810_38'],
'Mouse12-120810_40':store['phase_spike_theta_Mouse12-120810_40']	
}
spikes_swr_ex = {
'Mouse12-120810_37':store['spike_swrMouse12-120810_37'],
'Mouse12-120810_38':store['spike_swrMouse12-120810_38'],
'Mouse12-120810_40':store['spike_swrMouse12-120810_40']	
}
spikes_theta_ex = {
'Mouse12-120810_37':store['spike_thetaMouse12-120810_37'],
'Mouse12-120810_38':store['spike_thetaMouse12-120810_38'],
'Mouse12-120810_40':store['spike_thetaMouse12-120810_40']	
}
swr_ep = store['swr_ep']

generalinfo 	= scipy.io.loadmat(data_directory+session+'/Analysis/GeneralInfo.mat')
shankStructure 	= loadShankStructure(generalinfo)
if len(generalinfo['channelStructure'][0][0][1][0]) == 2:
	hpc_channel 	= generalinfo['channelStructure'][0][0][1][0][1][0][0] - 1
else:
	hpc_channel 	= generalinfo['channelStructure'][0][0][1][0][0][0][0] - 1	
spikes,shank	= loadSpikeData(data_directory+session+'/Analysis/SpikeData.mat', shankStructure['thalamus'])		
allneurons 		= [session.split("/")[1]+"_"+str(list(spikes.keys())[i]) for i in spikes.keys()]

theta_mod, theta_ses 	= loadThetaMod('/mnt/DataGuillaume/MergedData/THETA_THAL_mod.pickle', datasets, return_index=True)
theta_mod_rem 	= pd.DataFrame(index = theta_ses['rem'], columns = ['phase', 'pvalue', 'kappa'], data = theta_mod['rem'])
theta_mod_rem 	= theta_mod_rem.loc[allneurons]
theta_mod_rem['phase'] += 2*np.pi
theta_mod_rem['phase'] %= 2*np.pi
theta_mod_rem 	= theta_mod_rem.sort_values('phase')
allneurons_sorted = theta_mod_rem.index.values

###############################################################################################################
###############################################################################################################
# PLOT
###############################################################################################################
def figsize(scale):
	fig_width_pt = 483.69687                         # Get this from LaTeX using \the\textwidth
	inches_per_pt = 1.0/72.27                       # Convert pt to inch
	golden_mean = (np.sqrt(5.0)-1.0)/2.0            # Aesthetic ratio (you could change this)
	fig_width = fig_width_pt*inches_per_pt*scale    # width in inches
	fig_height = fig_width*golden_mean*2.0           # height in inches
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

mpl.use("pdf")
pdf_with_latex = {                      # setup matplotlib to use latex for output
	"pgf.texsystem": "pdflatex",        # change this if using xetex or lautex
	"text.usetex": True,                # use LaTeX to write all text
	"font.family": "serif",
	"font.serif": [],                   # blank entries should cause plots to inherit fonts from the document
	"font.sans-serif": [],
	"font.monospace": [],
	"axes.labelsize": 4,               # LaTeX default is 10pt font.
	"font.size": 7,
	"legend.fontsize": 4,               # Make the legend/label fonts a little smaller
	"xtick.labelsize": 4,
	"ytick.labelsize": 4,
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

fig = figure(figsize = figsize(0.5))
# outer = gridspec.GridSpec(3,3, wspace = 0.4, hspace = 0.5)#, height_ratios = [1,3])#, width_ratios = [1.6,0.7]) 
hrat = np.ones(11)
hrat[4] = 0.1
hrat[7] = 0.1

gs = gridspec.GridSpec(11,6, wspace = 0.3, hspace = 0.5, top = 1.0, bottom = 0.0, right = 0.9, left = 0.0, height_ratios = hrat)
# gs = gridspec.GridSpecFromSubplotSpec(1,1, subplot_spec = outer[0])

############ LFP ###########################
ax1 = subplot(gs[0:2,0:3])
plot(lfp_hpc_theta, 	color = 'black', linewidth = 0.5)
plot(lfp_filt_hpc_theta-1300.0, color = 'black', linewidth = 0.5)
noaxis(ax1)
ylabel("CA1 LFP")
title("REM", fontsize = 5, y = 0.90)
text(0.85, 0,'6-14 Hz', horizontalalignment='center', verticalalignment='center', transform=ax1.transAxes, fontsize = 4)
plot([start_theta,start_theta+100000], [-1800,-1800], '-', linewidth = 0.5, color = 'black')
text(0.1, -0.05,'100 ms', horizontalalignment='center', verticalalignment='center', transform=ax1.transAxes, fontsize = 4)

# ax1 = subplot(gs[1,0:3])
# plot(lfp_filt_hpc_theta, 		color = 'black', linewidth = 0.5)
# noaxis(ax1)
# ylabel("LFP")

ax3 = subplot(gs[0:2,3:6])
plot(lfp_hpc_swr.loc[start_rip:end_rip], color = 'black', 		linewidth = 0.5)
plot(lfp_filt_hpc_swr.loc[start_rip:end_rip]-1300, color = 'black', linewidth = 0.5)
for i in swr_ep.index.values[2:]:
	start3, end3 = swr_ep.loc[i]
	plot(lfp_hpc_swr.loc[start3:end3], linewidth = 0.5, color = 'red')
noaxis(ax3)
title("NON-REM", fontsize = 5, y = 0.9)
text(0.85, 0,'100-300 Hz', horizontalalignment='center', verticalalignment='center', transform=ax3.transAxes, fontsize = 4)
plot([start_rip,start_rip+40000], [-1800,-1800], '-', linewidth = 0.5, color = 'black')
text(0.1, -0.05,'40 ms', horizontalalignment='center', verticalalignment='center', transform=ax3.transAxes, fontsize = 4)




########### SPIKES #######################
ax2 = subplot(gs[2:4,0:3], sharex = ax1)
count = 0
msize = 3
mwidth = 0.5
for n in allneurons_sorted:	
	xt = spikes[int(n.split("_")[1])].loc[start_theta:end_theta].index.values
	if len(xt):			
		if n in neurons:			
			plot(xt, np.ones(len(xt))*count, '|', markersize = msize, markeredgewidth = mwidth, color = colors[neurons.index(n)])
		else:
			plot(xt, np.ones(len(xt))*count, '|', markersize = msize, markeredgewidth = mwidth, color = 'black')
		count+=1
noaxis(ax2)
ylabel("Thalamus")

ax4 = subplot(gs[2:4,3:6])
count = 0
for n in allneurons_sorted:	
	xt = spikes[int(n.split("_")[1])].loc[start_rip:end_rip].index.values
	count+=1
	if len(xt):			
		if n in neurons:			
			plot(xt, np.ones(len(xt))*count, '|', markersize = msize, markeredgewidth = mwidth, color = colors[neurons.index(n)])
		else:
			plot(xt, np.ones(len(xt))*count, '|', markersize = msize, markeredgewidth = mwidth, color = 'black')
		

noaxis(ax4)

##### EXEMPLES ##########################
pos = [0,2,4]
for n in neurons:
	# THETA PHASE
	ax = subplot(gs[5:7,pos[neurons.index(n)]:pos[neurons.index(n)]+2], projection = 'polar')	
	tmp = phase_spike_theta[n].values
	tmp += 2*np.pi
	tmp %= 2*np.pi
	hist(tmp,20, color = colors[neurons.index(n)])
	# xlabel("$\mathbf{Theta\ phase}$", fontsize = 8)
	xticks(np.arange(0, 2*np.pi, np.pi/4), ['0', '', '$\pi/2$', '', '$\pi$', '', '$3\pi/2$',''])
	yticks([])	
	grid(linestyle = '--')
	ax.yaxis.grid(False)
	if neurons.index(n) == 1:
		text(0.5, 1.17,'THETA PHASE MODULATION', horizontalalignment='center', verticalalignment='center', transform=ax.transAxes, fontsize = 6)


	# CROSS CORR
	ax = subplot(gs[8,pos[neurons.index(n)]:pos[neurons.index(n)]+2])
	simpleaxis(ax)
	ax.spines['bottom'].set_visible(False)	
	H0[n] = gaussFilt(H0[n].values, (2,))
	plot(H0[n].loc[-500:500], color = colors[neurons.index(n)], label = '', linewidth = 0.7)
	plot(Hm[n].loc[-500:500], ':', color = colors[neurons.index(n)], label = 'Jitter', linewidth = 0.7)

	axvline(0, color = 'grey', linewidth = 0.5)
	xticks([], [])
	if neurons.index(n) == 0:
		ylabel('Firing rate (Hz)')
	legend(edgecolor = None, facecolor = None, frameon = False)	
	if neurons.index(n) == 1:
		text(0.5, 1.23,'RIPPLES MODULATION', horizontalalignment='center', verticalalignment='center', transform=ax.transAxes, fontsize = 6)
	

	# Z
	ax = subplot(gs[9:11,pos[neurons.index(n)]:pos[neurons.index(n)]+2])
	simpleaxis(ax)
	z = pd.DataFrame((H0[n] - Hm[n])/Hstd.loc[n][0])
	z['filt'] = gaussFilt(z.values.flatten(), (10,))

	plot(z['filt'].loc[-500:500],  color = colors[neurons.index(n)], linewidth = 0.7)
	# xlabel('Time from \n $\mathbf{Sharp\ Waves\ ripples}$ (ms)', fontsize = 8)
	xlabel('Time from ripples (ms)', fontsize = 4)
	if neurons.index(n) == 0:
		ylabel('Modulation (a.u.)')
	axvline(0, color = 'grey', linewidth = 0.5)	
	ylim(-2,3)
	yticks([-1,0,1,2,3])


savefig("../figures/figures_articles/figart_1.pdf", dpi = 900, bbox_inches = 'tight', facecolor = 'white')
os.system("evince ../figures/figures_articles/figart_1.pdf &")



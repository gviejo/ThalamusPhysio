#!/usr/bin/env python
'''
    File name: main_ripp_mod.py
    Author: Guillaume Viejo
    Date created: 16/08/2017    
    Python Version: 3.5.2


'''
import sys
import numpy as np
import pandas as pd
import scipy.io
from functions import *
# from pylab import *
# import ipyparallel
from multiprocessing import Pool
import os
import neuroseries as nts
from time import time
from pylab import *
from sklearn.manifold import Isomap
from mpl_toolkits.mplot3d import Axes3D
from numba import jit



data_directory = '/mnt/DataGuillaume/MergedData/'
datasets = np.loadtxt(data_directory+'datasets_ThalHpc.list', delimiter = '\n', dtype = str, comments = '#')
datatosave = {ep:pd.DataFrame() for ep in ['wak', 'rem', 'sws']}


# session = 'Mouse17/Mouse17-130130'
# session = 'Mouse12/Mouse12-120808'
session = 'Mouse32/Mouse32-140822'

generalinfo 	= scipy.io.loadmat(data_directory+session+'/Analysis/GeneralInfo.mat')
shankStructure 	= loadShankStructure(generalinfo)
if len(generalinfo['channelStructure'][0][0][1][0]) == 2:
	hpc_channel 	= generalinfo['channelStructure'][0][0][1][0][1][0][0] - 1
else:
	hpc_channel 	= generalinfo['channelStructure'][0][0][1][0][0][0][0] - 1		
spikes,shank	= loadSpikeData(data_directory+session+'/Analysis/SpikeData.mat', shankStructure['thalamus'])		
n_channel,fs, shank_to_channel = loadXML(data_directory+session+"/"+session.split("/")[1]+'.xml')	
wake_ep 		= loadEpoch(data_directory+session, 'wake')
sleep_ep 		= loadEpoch(data_directory+session, 'sleep')
sws_ep 			= loadEpoch(data_directory+session, 'sws')
rem_ep 			= loadEpoch(data_directory+session, 'rem')
sleep_ep 		= sleep_ep.merge_close_intervals(threshold=1.e3)		
sws_ep 			= sleep_ep.intersect(sws_ep)	
rem_ep 			= sleep_ep.intersect(rem_ep)
rip_ep,rip_tsd 	= loadRipples(data_directory+session)
rip_ep			= sws_ep.intersect(rip_ep)	
rip_tsd 		= rip_tsd.restrict(sws_ep)
speed 			= loadSpeed(data_directory+session+'/Analysis/linspeed.mat').restrict(wake_ep)
hd_info 		= scipy.io.loadmat(data_directory+session+'/Analysis/HDCells.mat')['hdCellStats'][:,-1]
hd_info_neuron	= np.array([hd_info[n] for n in spikes.keys()])

spikeshd 		= {k:spikes[k] for k in np.where(hd_info_neuron==1)[0] if k not in []}
neurons 		= np.sort(list(spikeshd.keys()))

# lfp_hpc 		= loadLFP(data_directory+session+"/"+session.split("/")[1]+'.eeg', n_channel, hpc_channel, float(fs), 'int16')
# tmp = [lfp_hpc.loc[t-1e6:t+1e6] for i, t in enumerate(rip_tsd.index.values)]
# tmp = pd.concat(tmp, 0)
# tmp = tmp[~tmp.index.duplicated(keep='first')]		
# tmp.as_series().to_hdf(data_directory+session+'/'+session.split("/")[1]+'_EEG_SWR.h5', 'swr')



lfp_hpc 		= pd.read_hdf(data_directory+session+'/'+session.split("/")[1]+'_EEG_SWR.h5')

####################################################################################################################
# HEAD DIRECTION INFO
####################################################################################################################
spikeshd 		= {k:spikes[k] for k in np.where(hd_info_neuron==1)[0] if k not in []}
neurons 		= np.sort(list(spikeshd.keys()))
# sys.exit()
# position 		= pd.read_csv(data_directory+session+"/Mouse12-120808_wake.txt", delimiter = '\t', header = None, index_col = [0])
position 		= pd.read_csv(data_directory+session+"/Mouse32-140822.csv", delimiter = ',', header = None, index_col = [0])
angle 			= nts.Tsd(t = position.index.values, d = position[1].values, time_units = 's')
tcurves 		= computeAngularTuningCurves(spikeshd, angle, wake_ep, nb_bins = 60, frequency = 1/0.0256)
neurons 		= tcurves.idxmax().sort_values().index.values


####################################################################################################################
# binning data
####################################################################################################################
allrates		= {}

good_ex = (np.array([4644.8144,4924.4720,5244.9392,7222.9480,7780.2968, 11110.1888, 11292.3240, 11874.5688])*1e6).astype('int')

n_ex = 500

tmp = nts.Ts(rip_tsd.as_series().sample(n_ex-len(good_ex), replace = False).sort_index()).index.values
rip_tsd = pd.Series(index = np.hstack((good_ex, tmp)), data = np.nan)
# rip_tsd = rip_tsd.iloc[0:200]

bins_size = [200,10,100]


####################################################################################################################
# BIN WAKE
####################################################################################################################
bin_size = bins_size[0]
bins = np.arange(wake_ep.as_units('ms').start.iloc[0], wake_ep.as_units('ms').end.iloc[-1]+bin_size, bin_size)
spike_counts = pd.DataFrame(index = bins[0:-1]+np.diff(bins)/2, columns = neurons)
for i in neurons:
	spks = spikeshd[i].as_units('ms').index.values
	spike_counts[i], _ = np.histogram(spks, bins)

# allrates['wak'] = np.sqrt(spike_counts/(bins_size[0]*1e-3))
allrates['wak'] = np.sqrt(spike_counts/(bins_size[0]))

wakangle = pd.Series(index = np.arange(len(bins)-1))
tmp = angle.groupby(np.digitize(angle.as_units('ms').index.values, bins)-1).mean()
wakangle.loc[tmp.index] = tmp
wakangle.index = pd.Index(bins[0:-1] + np.diff(bins)/2.)

####################################################################################################################
# BIN SWR
####################################################################################################################
@jit(nopython=True)
def histo(spk, obins):
	n = len(obins)
	count = np.zeros(n)
	for i in range(n):
		count[i] = np.sum((spk>obins[i,0]) * (spk < obins[i,1]))
	return count


bin_size = bins_size[2]	
left_bound = np.arange(-500-bin_size/2, 500 - bin_size/4,bin_size/4)
obins = np.vstack((left_bound, left_bound+bin_size)).T

times = obins[:,0]+(np.diff(obins)/2).flatten()
tmp = []

rip_spikes = {}
# sys.exit()
tmp2 = rip_tsd.index.values/1e3
for i, t in enumerate(tmp2):
	print(i, t)
	tbins = t + obins
	spike_counts = pd.DataFrame(index = obins[:,0]+(np.diff(obins)/2).flatten(), columns = neurons)
	rip_spikes[i] = {}
	for j in neurons:
		spks = spikeshd[j].as_units('ms').index.values
		spike_counts[j] = histo(spks, tbins)
		nspks = spks - t
		rip_spikes[i][j] = nspks[np.logical_and((spks-t)>=-500, (spks-t)<=500)]
	tmp.append(np.sqrt(spike_counts/(bins_size[-1])))

allrates['swr'] = tmp



####################################################################################################################
# BIN RANDOM
####################################################################################################################
tmp = []
rnd_tsd = nts.Ts(t = np.sort(np.hstack([np.random.randint(sws_ep.loc[i,'start']+500000, sws_ep.loc[i,'end']+500000, n_ex//len(sws_ep)) for i in sws_ep.index])))
tmp3 = rnd_tsd.index.values/1000

for i, t in enumerate(tmp3):
	print(i, t)
	tbins = t + obins
	spike_counts = pd.DataFrame(index = obins[:,0]+(np.diff(obins)/2).flatten(), columns = neurons)	
	for j in neurons:
		spks = spikeshd[j].as_units('ms').index.values	
		spike_counts[j] = histo(spks, tbins)
		nspks = spks - t
	
	tmp.append(np.sqrt(spike_counts/(bins_size[-1])))
	
allrates['rnd'] = tmp



####################################################################################################################
# SMOOTHING
####################################################################################################################
tmp1 = allrates['wak'].rolling(window=200,win_type='gaussian',center=True,min_periods=1, axis = 0).mean(std=2).values

tmp3 = []
for rates in allrates['swr']:
	tmp3.append(rates.rolling(window=100,win_type='gaussian',center=True,min_periods=1,axis=0).mean(std=2).values)
tmp3 = np.vstack(tmp3)

tmp2 = []
for rates in allrates['rnd']:
	tmp2.append(rates.rolling(window=100,win_type='gaussian',center=True,min_periods=1,axis=0).mean(std=2).values)
tmp2 = np.vstack(tmp2)



n = len(tmp1)

tmp = np.vstack((tmp1, tmp3))


imap = Isomap(n_neighbors = 100, n_components = 2, n_jobs = -1).fit_transform(tmp)

iwak = imap[0:n]
isws = imap[n:]
iswr = imap[n:]

tokeep = np.where(np.logical_and(times>=-500,times<=500))[0]

iswr = iswr.reshape(len(rip_tsd),len(tokeep),2)

tmp = np.vstack((tmp1, tmp2))

imap2 = Isomap(n_neighbors = 100, n_components = 2, n_jobs = -1).fit_transform(tmp)

iwak2 = imap2[0:n]
irand = imap2[n:]
irand = irand.reshape(len(rnd_tsd), len(tokeep),2)


####################################################################################################################
# SAVING DATA
####################################################################################################################
datatosave = {
	"iwak"		: iwak,
	"iswr"		: iswr,
	"rip_tsd"	: rip_tsd,
	"rip_spikes": rip_spikes,
	"times" 	: times,
	"wakangle"	: wakangle,
	"neurons"	: neurons,
	"tcurves"	: tcurves,
	"iwak2"		: iwak2,
	"irand"		: irand
	}

import _pickle as cPickle
cPickle.dump(datatosave, open('../figures/figures_articles_v4/figure1/'+session.split("/")[1]+'.pickle', 'wb'))



####################################################################################################################
# PLOTTING
####################################################################################################################
colors = np.hstack((np.linspace(0, 1, int(len(times)/2)), np.ones(1), np.linspace(0, 1, int(len(times)/2))[::-1]))[tokeep]

colors = np.arange(len(times))[tokeep]

H = wakangle.values/(2*np.pi)

HSV = np.vstack((H, np.ones_like(H), np.ones_like(H))).T

from matplotlib.colors import hsv_to_rgb

RGB = hsv_to_rgb(HSV)



def figsize(scale):
	fig_width_pt = 483.69687                         # Get this from LaTeX using \the\textwidth
	inches_per_pt = 1.0/72.27                       # Convert pt to inch
	golden_mean = (np.sqrt(5.0)-1.0)/2.0            # Aesthetic ratio (you could change this)
	fig_width = fig_width_pt*inches_per_pt*scale    # width in inches
	fig_height = fig_width*golden_mean*1.8          # height in inches
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


from matplotlib.gridspec import GridSpec
from matplotlib import colors


for i in range(iswr.shape[0]):
# for i in range(1):
	print(i)
	fig = figure(figsize = figsize(1.0))
	gs = GridSpec(5,1, figure = fig, height_ratios = [0.1, 0.0, 0.3, 0.2, 0.7], hspace = 0)

	ax0 = subplot(gs[0,:])
	noaxis(ax0)
	lfp = lfp_hpc.loc[rip_tsd.index[i]-5e5:rip_tsd.index[i]+5e5]
	lfp = nts.Tsd(t = lfp.index.values - rip_tsd.index[i], d = lfp.values)
	plot(lfp.as_units('ms'), color = 'black')
	plot([0], [lfp.max()-50], '*', color = 'red', markersize = 5)
	ylabel('CA1', labelpad = 25)
	xlim(times[tokeep][0], times[tokeep][-1])
	

	ax1 = subplot(gs[2,:])
	simpleaxis(ax1)
	for j, n in enumerate(neurons):
		spk = rip_spikes[i][n]
		if len(spk):
			h = tcurves[n].idxmax()/(2*np.pi)
			hsv = np.repeat(np.atleast_2d(hsv_to_rgb([h,1,1])), len(spk), 0)
			scatter(spk, np.ones_like(spk)*j, c = hsv, marker = '|', s = 100, linewidth= 3)
	xlim(times[tokeep][0], times[tokeep][-1])
	ylim(-1, len(neurons)+1)
	xticks(np.arange(times[tokeep][0], times[tokeep][-1]+100, 100))
	xlabel("Time from SWR (ms)")
	ylabel("HD neurons")
	# ax1.spines['left'].set_visible(False)
	# plot([-300,-300], [0, len(neurons)], color = 'black')
	# x = times[tokeep[1:-1]]
	# y = np.ones_like(x)*-2
	# plot(x, y, '-', color = 'grey', zorder = 1)
	# scatter(x, y, c = np.arange(len(x)), zorder = 2, cmap=plt.cm.get_cmap('viridis'), s = 20)


	ax2 = subplot(gs[4,:])
	noaxis(ax2)
	ax2.set_aspect(aspect=1)
	scatter(iwak[~np.isnan(H),0], iwak[~np.isnan(H),1], c = RGB[~np.isnan(H)], marker = '.', alpha = 0.5, zorder = 2, linewidth = 0, s= 40)
	plot(iswr[i,:,0], iswr[i,:,1], alpha = 0.5, zorder = 4, color = 'grey')
	# cNorm = colors.Normalize(vmin = 0, vmax=1)
	cl = np.linspace(0, 0.7, len(tokeep))
	scatter(iswr[i,:,0], iswr[i,:,1], c =  cl, zorder = 5, cmap=plt.cm.get_cmap('bone'), s = 50, vmin = 0, vmax = 1.0)
	idx = np.where(times[tokeep] == 0)[0][0]
	plot(iswr[i,idx,0], iswr[i,idx,1], '*', color = 'red', zorder = 6, markersize = 10)
	# ax2.set_title("Head-direction manifold")
	ylabel(session+ " " + str(rip_tsd.index.values[i]/1e6) + " s " + str(i), fontsize = 6, labelpad = 40)

	# hsv
	display_axes = fig.add_axes([0.2,0.45,0.1,0.1], projection='polar')
	colormap = plt.get_cmap('hsv')
	norm = mpl.colors.Normalize(0.0, 2*np.pi)
	xval = np.arange(0, 2*pi, 0.01)
	yval = np.ones_like(xval)
	display_axes.scatter(xval, yval, c=xval, s=100, cmap=colormap, norm=norm, linewidths=0, alpha = 0.8)
	display_axes.set_yticks([])
	display_axes.set_xticks(np.arange(0, 2*np.pi, np.pi/2))
	display_axes.grid(False)

	#colorbar	
	c_map_ax = fig.add_axes([0.44, 0.59, 0.22, 0.02])
	# c_map_ax.axes.get_xaxis().set_visible(False)
	# c_map_ax.axes.get_yaxis().set_visible(False)
	# c_map_ax.set_xticklabels([times[tokeep][0], 0, times[tokeep][-1]])
	cb = mpl.colorbar.ColorbarBase(c_map_ax, cmap=plt.cm.get_cmap('bone'), orientation = 'horizontal')
	cb.ax.set_xticklabels([int(times[tokeep][0]), 0, int(times[tokeep][-1])])
	# cb.ax.set_title("Time from SWR (ms)", fontsize = 7)

	gs.update(left = 0.15, right = 0.95, bottom = 0.05, top = 0.95)

	savefig("../figures/figures_articles_v4/figure1/ex_swr_"+'{:03}'.format(i)+".pdf", dpi = 900, facecolor = 'white')
	# os.system("evince ../figures/figures_articles_v4/figure1/ex_swr_9.pdf &")


os.system("pdftk ../figures/figures_articles_v4/figure1/ex_swr_*.pdf cat output ../figures/figures_articles_v4/figure1/swr_all_exemples.pdf")
os.system("evince ../figures/figures_articles_v4/figure1/swr_all_exemples.pdf &")
os.system("rm ../figures/figures_articles_v4/figure1/ex_swr_*")
# os.system(r"cp ../figures/figures_articles_v4/figure1/swr_all_exemples.pdf /home/guillaume/Dropbox (Peyrache Lab)/swr_all_exemples.pdf")

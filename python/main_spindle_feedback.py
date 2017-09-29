#!/usr/bin/env python
'''
	File name: main_spindle_feedback.py
	Author: Guillaume Viejo
	Date created: 20/09/2017    
	Python Version: 3.5.2

to search for feedback of hippocampal spindles to thalamic spindles

'''
import numpy as np
import pandas as pd
import scipy.io
from functions import *
# from pylab import *
import ipyparallel
import os, sys
import neuroseries as nts
import time
from Wavelets import MyMorlet as Morlet


data_directory = '/mnt/DataGuillaume/MergedData/'
datasets = np.loadtxt(data_directory+'datasets_ThalHpc.list', delimiter = '\n', dtype = str, comments = '#')
datatosave = {}

allkappa = []

meankappa = []
session_rip_in_thl_spind_mod = []
session_rip_in_thl_spind_phase = {}
session_rip_in_hpc_spind_mod = []
session_rip_in_hpc_spind_phase = {}
phase = {}
session_kappa_around_swr = []
session_hpc_kappa_around_swr = []
session_spind_mod1 = {}
session_spind_mod2 = {}
session_spind_mod3 = {}
session_spind_mod4 = {}
session_spind_mod5 = {}
session_spind_mod6 = {}
session_cross_corr = {}
session_cross_corr_swr_spikes_in_thl_spindles = {}
session_cross_corr_swr_spikes_out_thl_spindles = {}

for session in datasets: 
	print("session"+session)	
	sws_ep 			= loadEpoch(data_directory+session, 'sws')
	tmp 			= np.genfromtxt("/mnt/DataGuillaume/MergedData/"+session+"/"+session.split("/")[1]+".evt.spd.thl")[:,0]
	tmp 			= tmp.reshape(len(tmp)//2,2)
	spind_thl_ep 	= nts.IntervalSet(tmp[:,0], tmp[:,1], time_units = 'ms')
	tmp 			= np.genfromtxt("/mnt/DataGuillaume/MergedData/"+session+"/"+session.split("/")[1]+".evt.spd.hpc")[:,0]
	tmp 			= tmp.reshape(len(tmp)//2,2)
	spind_hpc_ep 	= nts.IntervalSet(tmp[:,0], tmp[:,1], time_units = 'ms')	
	spind_ep 		= spind_hpc_ep.intersect(spind_thl_ep).drop_short_intervals(0.0)
	spind_thl_no_hpc = spind_thl_ep.set_diff(spind_hpc_ep).drop_short_intervals(0.0)
	spind_hpc_no_thl = spind_hpc_ep.set_diff(spind_thl_ep).drop_short_intervals(0.0)
	store 			= pd.HDFStore("../data/phase_spindles/"+session.split("/")[1]+".lfp")
	phase_hpc 		= nts.Tsd(store['phase_hpc_spindles'])
	phase_thl 		= nts.Tsd(store['phase_thl_spindles'][0])	
	store.close()	
	spikes 			= {}
	store_spike 	= pd.HDFStore("../data/spikes_thalamus/"+session.split("/")[1]+".spk")
	for n in store_spike.keys(): spikes[int(n[1:])] = nts.Ts(store_spike[n])
	store_spike.close()	

##################################################################################################
# SPINDLES MODULATION
##################################################################################################		

	spind_mod1 		= computePhaseModulation(phase_hpc, spikes, spind_hpc_ep)
	spind_mod2 		= computePhaseModulation(phase_thl, spikes, spind_thl_ep)
	spind_mod3 		= computePhaseModulation(phase_hpc, spikes, spind_ep)
	spind_mod4 		= computePhaseModulation(phase_thl, spikes, spind_ep)
	spind_mod5 		= computePhaseModulation(phase_thl, spikes, spind_thl_no_hpc)
	spind_mod6 		= computePhaseModulation(phase_hpc, spikes, spind_hpc_no_thl)
	
	kappa 			= np.vstack([spind_mod1[:,2], spind_mod3[:,2], spind_mod2[:,2], spind_mod4[:,2]]).transpose()

	kappa[np.isnan(kappa)] = 0.0

	allkappa.append(kappa)
	meankappa.append(kappa.mean(0))
	session_spind_mod1[session] = spind_mod1
	session_spind_mod2[session] = spind_mod2
	session_spind_mod3[session] = spind_mod3
	session_spind_mod4[session] = spind_mod4
	session_spind_mod5[session] = spind_mod5
	session_spind_mod6[session] = spind_mod6


##################################################################################################
# Phase of RIPPLES in SPINDLES
##################################################################################################		
	rip_ep,rip_tsd 	= loadRipples(data_directory+session)
	rip_ep			= sws_ep.intersect(rip_ep)	
	rip_tsd 		= rip_tsd.restrict(sws_ep)		

	rip_in_thl_spind_mod, rip_in_thl_spind_phase = computePhaseModulation(phase_thl, {0:rip_tsd}, spind_thl_ep, True)
	rip_in_hpc_spind_mod, rip_in_hpc_spind_phase = computePhaseModulation(phase_hpc, {0:rip_tsd}, spind_hpc_ep, True)

	session_rip_in_thl_spind_mod.append(rip_in_thl_spind_mod)
	session_rip_in_hpc_spind_mod.append(rip_in_hpc_spind_mod)
	session_rip_in_thl_spind_phase[session] = rip_in_thl_spind_phase
	session_rip_in_hpc_spind_phase[session] = rip_in_hpc_spind_phase

##################################################################################################
# Phase of Spindles between ripples
##################################################################################################		
	rip_in_spind_tsd = rip_tsd.restrict(spind_thl_ep)
	times = np.arange(0, 1005, 5) - 500
	phase_around_swr = np.zeros((len(rip_in_spind_tsd), len(times)))
	for rip_time, i in zip(rip_in_spind_tsd.as_units('ms').index.values, range(len(rip_in_spind_tsd))):
		phase_around_swr[i,:] = phase_thl.realign(nts.Ts(rip_time+times, time_units = 'ms'))

	kappa_around_swr = np.zeros(len(times))
	for t in range(len(times)):
		mu, kappa, pval = getCircularMean(phase_around_swr[:,t])
		kappa_around_swr[t] = kappa

	session_kappa_around_swr.append(kappa_around_swr)


##################################################################################################
# Phase of  hpc Spindles between ripples
##################################################################################################		
	rip_in_spind_tsd = rip_tsd.restrict(spind_hpc_ep)
	times = np.arange(0, 1005, 5) - 500
	phase_around_swr = np.zeros((len(rip_in_spind_tsd), len(times)))
	for rip_time, i in zip(rip_in_spind_tsd.as_units('ms').index.values, range(len(rip_in_spind_tsd))):
		phase_around_swr[i,:] = phase_hpc.realign(nts.Ts(rip_time+times, time_units = 'ms'))

	kappa_around_swr = np.zeros(len(times))
	for t in range(len(times)):
		mu, kappa, pval = getCircularMean(phase_around_swr[:,t])
		kappa_around_swr[t] = kappa

	session_hpc_kappa_around_swr.append(kappa_around_swr)


##################################################################################################
# CROSS CORR of hippocampal and thalamic spindles 
##################################################################################################		
	start_spindles_hpc = spind_hpc_ep.as_units('ms')['start']
	end_spindles_hpc = spind_hpc_ep.as_units('ms')['end']
	mid_spindles_hpc = start_spindles_hpc + (end_spindles_hpc - start_spindles_hpc)/2.
	start_spindles_thl = spind_thl_ep.as_units('ms')['start']
	end_spindles_thl = spind_thl_ep.as_units('ms')['end']
	mid_spindles_thl = start_spindles_thl + (end_spindles_thl - start_spindles_thl)/2.	

	bin_size 	= 5 # ms 
	nb_bins 	= 200 
	session_cross_corr[session] = np.array([
		crossCorr(start_spindles_thl, start_spindles_hpc, bin_size, nb_bins),
		crossCorr(mid_spindles_thl, mid_spindles_hpc, bin_size, nb_bins),
		crossCorr(end_spindles_thl, end_spindles_hpc, bin_size, nb_bins)
		])
	
##################################################################################################
# CROSS CORR swr/spikes  for swr in and out of spindle
##################################################################################################		
	swr_in_thl_spindle = rip_tsd.restrict(spind_thl_ep)
	swr_out_thl_spindle = rip_tsd.restrict(sws_ep.set_diff(spind_thl_ep).drop_short_intervals(0.0))
		
	bin_size 	= 5 # ms 
	nb_bins 	= 200 	
	session_cross_corr_swr_spikes_in_thl_spindles[session] = np.array([crossCorr(swr_in_thl_spindle.as_units('ms').index.values, spikes[i].as_units('ms').index.values, bin_size, nb_bins) for i in spikes.keys()])
	session_cross_corr_swr_spikes_out_thl_spindles[session] = np.array([crossCorr(swr_out_thl_spindle.as_units('ms').index.values, spikes[i].as_units('ms').index.values, bin_size, nb_bins) for i in spikes.keys()])		

##################################################################################################
# SAVING
##################################################################################################		
np.save("../data/spindle_feedback/meankappa.npy", meankappa)
np.save("../data/spindle_feedback/session_rip_in_thl_spind_mod.npy", np.array(session_rip_in_thl_spind_mod))
np.save("../data/spindle_feedback/session_rip_in_hpc_spind_mod.npy", np.array(session_rip_in_hpc_spind_mod))
np.save("../data/spindle_feedback/session_kappa_around_swr.npy", np.array(session_kappa_around_swr))
np.save("../data/spindle_feedback/session_hpc_kappa_around_swr.npy", np.array(session_hpc_kappa_around_swr))
cPickle.dump(session_rip_in_thl_spind_phase 					, open('../data/spindle_feedback/session_rip_in_thl_spind_phase.pickle', 'wb'))
cPickle.dump(session_rip_in_hpc_spind_phase 					, open('../data/spindle_feedback/session_rip_in_hpc_spind_phase.pickle', 'wb'))
cPickle.dump(phase 												, open('../data/spindle_feedback/phase.pickle', 'wb'))
cPickle.dump(session_spind_mod1 								, open('../data/spindle_feedback/session_spind_mod1.pickle', 'wb'))
cPickle.dump(session_spind_mod2 								, open('../data/spindle_feedback/session_spind_mod2.pickle', 'wb'))
cPickle.dump(session_spind_mod3 								, open('../data/spindle_feedback/session_spind_mod3.pickle', 'wb'))
cPickle.dump(session_spind_mod4 								, open('../data/spindle_feedback/session_spind_mod4.pickle', 'wb'))
cPickle.dump(session_spind_mod5 								, open('../data/spindle_feedback/session_spind_mod5.pickle', 'wb'))
cPickle.dump(session_spind_mod6 								, open('../data/spindle_feedback/session_spind_mod6.pickle', 'wb'))
cPickle.dump(session_cross_corr 								, open('../data/spindle_feedback/session_cross_corr.pickle', 'wb'))
cPickle.dump(session_cross_corr_swr_spikes_in_thl_spindles 		, open('../data/spindle_feedback/session_cross_corr_swr_spikes_in_thl_spindles ', 'wb'))
cPickle.dump(session_cross_corr_swr_spikes_out_thl_spindles 	, open('../data/spindle_feedback/session_cross_corr_swr_spikes_out_thl_spindles', 'wb'))


##################################################################################################
# THETA MODULATION
##################################################################################################		
import _pickle as cPickle	
theta_mod = cPickle.load(open('/mnt/DataGuillaume/MergedData/THETA_THAL_mod.pickle', 'rb'))



from pylab import *
mouses = ['Mouse12', 'Mouse17', 'Mouse20', 'Mouse32']

figure(figsize = (20,10))
times = np.arange(0, 1005, 5) - 500
for j in range(4):	
	index = [i for i in range(len(datasets)) if mouses[j] in datasets[i]]
	names = [datasets[i] for i in range(len(datasets)) if mouses[j] in datasets[i]]
	print(index, names)
	cmap = plt.get_cmap('autumn')	
	colors = [cmap(i) for i in np.linspace(0, 1, len(index))]	
	subplot(3,4,j+1)
	ylabel("start thl(hpc)")
	for k, ses, i in zip(index, names, range(len(index))):
		plot(times, session_cross_corr[ses][0], color = colors[i])	
	axvline(0)
	title(mouses[j])
	subplot(3,4,j+5)
	ylabel("middle thl(hpc)")
	for k, ses, i in zip(index, names, range(len(index))):
		plot(times, session_cross_corr[ses][1], color = colors[i])	
	axvline(0)
	subplot(3,4,j+9)
	ylabel("end thl(hpc)")
	for k, ses, i in zip(index, names, range(len(index))):
		plot(times, session_cross_corr[ses][2], color = colors[i])	
	axvline(0)

savefig("../figures/spindle_feedback/fig1.pdf")


def set_lines(ax):
	ax.axhline(0.0, color = 'grey', alpha = 0.5)
	ax.axhline(2*np.pi, color = 'grey', alpha = 0.5)
	ax.axhline(np.pi, linestyle = '--', color = 'grey', alpha = 0.5)
	ax.axhline(-np.pi, linestyle = '--', color = 'grey', alpha = 0.5)
	ax.set_xticks([-np.pi, 0, np.pi, 2*np.pi])	
	ax.set_xticklabels(['-pi', '0', 'pi', '2pi'])
	ax.axvline(0.0, color = 'grey', alpha = 0.5)
	ax.axvline(2*np.pi, color = 'grey', alpha = 0.5)
	ax.axvline(np.pi, linestyle = '--', color = 'grey', alpha = 0.5)
	ax.axvline(-np.pi, linestyle = '--', color = 'grey', alpha = 0.5)
	ax.set_yticks([-np.pi, 0, np.pi, 2*np.pi])
	ax.set_yticklabels(['-pi', '0', 'pi', '2pi'])
	return 

figure(figsize = (20,10))
for j in range(4):	
	index = [i for i in range(len(datasets)) if mouses[j] in datasets[i]]
	names = [datasets[i] for i in range(len(datasets)) if mouses[j] in datasets[i]]
	print(index, names)
	cmap = plt.get_cmap('autumn')
	colors = [cmap(i) for i in np.linspace(0, 1, len(index))]	

	subplot(4,4,j+1)	
	for k, ses,i in zip(index, names, range(len(index))):
		theta_mod_toplot = theta_mod[ses]['theta_mod']['wake'][:,0]
		spindles_mod_toplot = session_spind_mod1[ses][:,0]
		force = theta_mod[ses]['theta_mod']['wake'][:,2] + session_spind_mod1[ses][:,2]
		x = np.concatenate([theta_mod_toplot, theta_mod_toplot, theta_mod_toplot+2*np.pi, theta_mod_toplot+2*np.pi])
		y = np.concatenate([spindles_mod_toplot, spindles_mod_toplot + 2*np.pi, spindles_mod_toplot, spindles_mod_toplot + 2*np.pi])
		scatter(x, y, s = force*10., color = colors[i], label = 'theta wake')
	title(mouses[j])	
	set_lines(gca())
	if j == 0:
		ylabel('Hippocampal \n Spindle phase (rad)')		
	subplot(4,4,j+5)
	for k, ses,i in zip(index, names, range(len(index))):
		theta_mod_toplot = theta_mod[ses]['theta_mod']['rem'][:,0]
		spindles_mod_toplot = session_spind_mod1[ses][:,0]
		force = theta_mod[ses]['theta_mod']['rem'][:,2] + session_spind_mod1[ses][:,2]
		x = np.concatenate([theta_mod_toplot, theta_mod_toplot, theta_mod_toplot+2*np.pi, theta_mod_toplot+2*np.pi])
		y = np.concatenate([spindles_mod_toplot, spindles_mod_toplot + 2*np.pi, spindles_mod_toplot, spindles_mod_toplot + 2*np.pi])
		scatter(x, y, s = force*10., marker = '^', color = colors[i], label = 'theta rem')	
	set_lines(gca())
	if j == 0:
		ylabel('Hippocampal \n Spindle phase (rad)')
	

	subplot(4,4,j+9)
	for k, ses,i in zip(index, names, range(len(index))):
		theta_mod_toplot = theta_mod[ses]['theta_mod']['wake'][:,0]
		spindles_mod_toplot = session_spind_mod2[ses][:,0]
		force = theta_mod[ses]['theta_mod']['wake'][:,2] + session_spind_mod2[ses][:,2]
		x = np.concatenate([theta_mod_toplot, theta_mod_toplot, theta_mod_toplot+2*np.pi, theta_mod_toplot+2*np.pi])
		y = np.concatenate([spindles_mod_toplot, spindles_mod_toplot + 2*np.pi, spindles_mod_toplot, spindles_mod_toplot + 2*np.pi])
		scatter(x, y, s = force*10., color = colors[i], label = 'theta wake')
	set_lines(gca())		
	if j == 0:
		ylabel('Thalamus \n Spindle phase (rad)')				
	subplot(4,4,j+13)
	for k, ses,i in zip(index, names, range(len(index))):
		theta_mod_toplot = theta_mod[ses]['theta_mod']['rem'][:,0]
		spindles_mod_toplot = session_spind_mod2[ses][:,0]
		force = theta_mod[ses]['theta_mod']['rem'][:,2] + session_spind_mod2[ses][:,2]
		x = np.concatenate([theta_mod_toplot, theta_mod_toplot, theta_mod_toplot+2*np.pi, theta_mod_toplot+2*np.pi])
		y = np.concatenate([spindles_mod_toplot, spindles_mod_toplot + 2*np.pi, spindles_mod_toplot, spindles_mod_toplot + 2*np.pi])
		scatter(x, y, s = force*10., marker = '^', color = colors[i], label = 'theta rem')
		xlabel('Theta phase (rad)')
	set_lines(gca())
	if j == 0:
		ylabel('Thalamus \n Spindle phase (rad)')

savefig("../figures/spindle_feedback/fig2.pdf")



figure(figsize = (20,10))
times = np.arange(0, 1005, 5) - 500
for j in range(4):
	subplot(4,1,j+1)
	index = [i for i in range(len(datasets)) if mouses[j] in datasets[i]]
	cmap = plt.get_cmap('autumn')
	colors = [cmap(i) for i in np.linspace(0, 1, len(index))]
	for k in range(len(index)):
		plot(times, session_kappa_around_swr[index[k]], color = colors[k])
	title(mouses[j]+" Kappa around SWR for thalamus phases")	
	xlabel('times (ms)')
	ylabel("kappa")

savefig("../figures/spindle_feedback/fig3.pdf")


figure(figsize = (20,10))
meankappa = np.vstack(meankappa)	
for j in range(4):
	subplot(4,1,j+1)
	index = [i for i in range(len(datasets)) if mouses[j] in datasets[i]]
	cmap = plt.get_cmap('autumn')
	colors = [cmap(i) for i in np.linspace(0, 1, len(index))]
	for k in range(len(index)):
		plot(meankappa[index[k]], color = colors[k])
	title(mouses[j])	
	xticks([0,1,2,3], ['Hpc phase(spind Hpc)', 'Hpc phase(spind Hpc^THL)', 'Thl phase(spind THL)', 'THL phase (spind HPC^THL)'])
	ylabel(" mean kappa per session")	
	ylim(0, 1)

savefig("../figures/spindle_feedback/fig4.pdf")

rip_prefered_phase_in_thl_spindle = {}
rip_prefered_phase_in_hpc_spindle = {}

figure(figsize = (20,10))
for j in range(4):
	
	index = [i for i in range(len(datasets)) if mouses[j] in datasets[i]]
	print(index)
	cmap = plt.get_cmap('autumn')
	colors = [cmap(i) for i in np.linspace(0, 1, len(index))]	

	subplot(2,4,j+1, projection = 'polar')	
	for k in range(len(index)):
		hist, bin_edges = np.histogram(session_rip_in_hpc_spind_phase[datasets[index[k]]][0].values, 100, density = True)
		y = gaussFilt(hist, (5,))
		x = bin_edges[0:-1] + (bin_edges[1] - bin_edges[0])/2
		rip_prefered_phase_in_hpc_spindle[datasets[index[k]]] = x[np.argmax(y)] 
		x = list(x)
		y = list(y)
		x = x+[x[0]]
		y = y+[y[0]]
		kappa = session_rip_in_hpc_spind_mod[index[k]][0][-1]		
		plot(x, y, '-', linewidth = kappa*2, color=colors[k])		
		print(np.max(y), kappa, k)

	title(mouses[j] + "\n Ripples phase in hippocampal spindles per session")	

	subplot(2,4,j+1+4, projection = 'polar')
	for k in range(len(index)):
		hist, bin_edges = np.histogram(session_rip_in_thl_spind_phase[datasets[index[k]]][0].values, 100, density = True)
		y = gaussFilt(hist, (5,))
		x = bin_edges[0:-1] + (bin_edges[1] - bin_edges[0])/2
		rip_prefered_phase_in_thl_spindle[datasets[index[k]]] = x[np.argmax(y)] 
		x = list(x)
		y = list(y)
		x = x+[x[0]]
		y = y+[y[0]]	
		kappa = session_rip_in_thl_spind_mod[index[k]][0][-1]		
		plot(x, y, '-', linewidth = kappa*2, color=colors[k])		
	title("Ripples phase in thalamic spindles per session")		

savefig("../figures/spindle_feedback/fig5.pdf")
	


figure(figsize = (20,10))
times = np.arange(0, 1005, 5) - 500
for j in range(4):
	
	index = [i for i in range(len(datasets)) if mouses[j] in datasets[i]]
	print(index)
	names = [datasets[i] for i in range(len(datasets)) if mouses[j] in datasets[i]]
	cmap = plt.get_cmap('autumn')
	colors = [cmap(i) for i in np.linspace(0, 1, len(index))]	

	subplot(2,4,j+1)	
	for k, ses,i in zip(index, names, range(len(index))):
		tmp = []
		for n in range(len(session_cross_corr_swr_spikes_in_thl_spindles[ses])):
			y = session_cross_corr_swr_spikes_in_thl_spindles[ses][n]
			# y = session_cross_corr_swr_spikes_in_thl_spindles[ses][n] - np.mean(session_cross_corr_swr_spikes_in_thl_spindles[ses][n])
			# y = session_cross_corr_swr_spikes_in_thl_spindles[ses][n] / np.std(session_cross_corr_swr_spikes_in_thl_spindles[ses][n])
			yf = gaussFilt(y, (10,))
			tmp.append(yf)

		plot(times, np.mean(tmp, 0) , color = colors[i])
	title(mouses[j] + "\n Cross-corr swr(spikes) IN THL SPINDLE")	

	subplot(2,4,j+1+4)
	for k, ses,i in zip(index, names, range(len(index))):
		tmp = []
		for n in range(len(session_cross_corr_swr_spikes_out_thl_spindles[ses])):
			y = session_cross_corr_swr_spikes_out_thl_spindles[ses][n]
			# y = session_cross_corr_swr_spikes_out_thl_spindles[ses][n] - np.mean(session_cross_corr_swr_spikes_out_thl_spindles[ses][n])
			# y = session_cross_corr_swr_spikes_out_thl_spindles[ses][n] / np.std(session_cross_corr_swr_spikes_out_thl_spindles[ses][n])
			yf = gaussFilt(y, (10,))
			tmp.append(yf)
		plot(times, np.mean(tmp, 0) , color = colors[i])		
	title(mouses[j] + "\n Cross-corr swr(spikes) OUT THL SPINDLE")	

savefig("../figures/spindle_feedback/fig6.pdf")
	



figure(figsize = (20,10))

for j in [0]:	
	index = [i for i in range(len(datasets)) if mouses[j] in datasets[i]]
	names = [datasets[i] for i in range(len(datasets)) if mouses[j] in datasets[i]]
	print(index, names)
	cmap = plt.get_cmap('autumn')
	colors = [cmap(i) for i in np.linspace(0, 1, len(index))]	
	count = 1	
	for k, ses,i in zip(index, names, range(len(index))):
		subplot(4,4,count)	
		theta_mod_toplot = theta_mod[ses]['theta_mod']['wake'][:,0]
		spindles_mod_toplot = session_spind_mod2[ses][:,0]
		force = theta_mod[ses]['theta_mod']['wake'][:,2] + session_spind_mod2[ses][:,2]
		x = np.concatenate([theta_mod_toplot, theta_mod_toplot, theta_mod_toplot+2*np.pi, theta_mod_toplot+2*np.pi])
		y = np.concatenate([spindles_mod_toplot, spindles_mod_toplot + 2*np.pi, spindles_mod_toplot, spindles_mod_toplot + 2*np.pi])
		scatter(x, y, s = force*10., color = colors[i], label = 'theta wake')

		theta_mod_toplot = theta_mod[ses]['theta_mod']['rem'][:,0]
		spindles_mod_toplot = session_spind_mod2[ses][:,0]
		force = theta_mod[ses]['theta_mod']['rem'][:,2] + session_spind_mod2[ses][:,2]
		x = np.concatenate([theta_mod_toplot, theta_mod_toplot, theta_mod_toplot+2*np.pi, theta_mod_toplot+2*np.pi])
		y = np.concatenate([spindles_mod_toplot, spindles_mod_toplot + 2*np.pi, spindles_mod_toplot, spindles_mod_toplot + 2*np.pi])
		scatter(x, y, s = force*10., marker = '^', color = colors[i], label = 'theta rem')
		if count in np.arange(1,16,4):
			ylabel('Thalamus \n Spindle phase (rad)')			
		if count >= 13:
			xlabel('Theta phase (rad)')
		if count == 2:
			title(mouses[j])
		count += 1
savefig("../figures/spindle_feedback/fig7.pdf")



figure(figsize = (20,10))
times = np.arange(0, 1005, 5) - 500
for j in range(4):
	
	index = [i for i in range(len(datasets)) if mouses[j] in datasets[i]]
	print(index)
	names = [datasets[i] for i in range(len(datasets)) if mouses[j] in datasets[i]]
	cmap = plt.get_cmap('autumn')
	colors = [cmap(i) for i in np.linspace(0, 1, len(index))]	

	subplot(1,4,j+1)	
	for k, ses,i in zip(index, names, range(len(index))):
		tmp = []
		for n in range(len(session_cross_corr_swr_spikes_in_thl_spindles[ses])):
			# y = session_cross_corr_swr_spikes_in_thl_spindles[ses][n] - np.mean(session_cross_corr_swr_spikes_in_thl_spindles[ses][n])
			# y = session_cross_corr_swr_spikes_in_thl_spindles[ses][n] / np.std(session_cross_corr_swr_spikes_in_thl_spindles[ses][n])
			# x = session_cross_corr_swr_spikes_out_thl_spindles[ses][n] - np.mean(session_cross_corr_swr_spikes_out_thl_spindles[ses][n])
			# x = session_cross_corr_swr_spikes_out_thl_spindles[ses][n] / np.std(session_cross_corr_swr_spikes_out_thl_spindles[ses][n])
			y = session_cross_corr_swr_spikes_in_thl_spindles[ses][n] 
			x = session_cross_corr_swr_spikes_out_thl_spindles[ses][n]			
			yf = gaussFilt(y, (10,))
			xf = gaussFilt(x, (10,))
			tmp.append(yf - xf)
		
		plot(times, np.mean(tmp, 0), '-o', color = colors[i])
	
	ylabel("in - out cross corr")	
	title(mouses[j] + "\n Cross-corr swr(spikes) IN - OUT THL SPINDLE")	
	
savefig("../figures/spindle_feedback/fig8.pdf")



figure(figsize = (20,10))
times = np.arange(0, 1005, 5) - 500
for j in range(4):
	subplot(4,1,j+1)
	index = [i for i in range(len(datasets)) if mouses[j] in datasets[i]]
	cmap = plt.get_cmap('autumn')
	colors = [cmap(i) for i in np.linspace(0, 1, len(index))]
	for k in range(len(index)):
		plot(times, session_hpc_kappa_around_swr[index[k]], color = colors[k])
	title(mouses[j]+" Kappa around SWR for hippocampus phases")	
	xlabel('times (ms)')
	ylabel("kappa")

savefig("../figures/spindle_feedback/fig9.pdf")
	

figure(figsize = (20,10))
times = np.arange(0, 1005, 5) - 500
for j in [0]:	
	index = [i for i in range(len(datasets)) if mouses[j] in datasets[i]]
	names = [datasets[i] for i in range(len(datasets)) if mouses[j] in datasets[i]]
	print(index, names)
	cmap = plt.get_cmap('autumn')
	colors = [cmap(i) for i in np.linspace(0, 1, len(index))]	
	count = 1	
	for k, ses,i in zip(index, names, range(len(index))):
		subplot(4,4,count)	
		plot(times, session_kappa_around_swr[index[i]], '-', color = 'blue', label = 'spdl thl')
		plot(times, session_hpc_kappa_around_swr[index[i]], '-', color = 'green', label = 'spdl hpc')
		axvline(0.0)
		xlabel("times")
		ylabel("kappa")
		legend()
		if count == 1:
			title("kappa around swr for "+mouses[j])
		count+=1
savefig("../figures/spindle_feedback/fig10.pdf")

figure(figsize = (20,10))
times = np.arange(0, 1005, 5) - 500
for j in [1]:	
	index = [i for i in range(len(datasets)) if mouses[j] in datasets[i]]
	names = [datasets[i] for i in range(len(datasets)) if mouses[j] in datasets[i]]
	print(index, names)
	cmap = plt.get_cmap('autumn')
	colors = [cmap(i) for i in np.linspace(0, 1, len(index))]	
	count = 1	
	for k, ses,i in zip(index, names, range(len(index))):
		subplot(4,5,count)	
		plot(times, session_kappa_around_swr[index[i]], '-', color = 'blue', label = 'spdl thl')
		plot(times, session_hpc_kappa_around_swr[index[i]], '-', color = 'green', label = 'spdl hpc')
		axvline(0.0)
		xlabel("times")
		ylabel("kappa")
		legend()
		if count == 1:
			title("kappa around swr for "+mouses[j])
		count+=1
savefig("../figures/spindle_feedback/fig11.pdf")

figure(figsize = (20,10))
times = np.arange(0, 1005, 5) - 500
for j in [2]:	
	index = [i for i in range(len(datasets)) if mouses[j] in datasets[i]]
	names = [datasets[i] for i in range(len(datasets)) if mouses[j] in datasets[i]]
	print(index, names)
	cmap = plt.get_cmap('autumn')
	colors = [cmap(i) for i in np.linspace(0, 1, len(index))]	
	count = 1	
	for k, ses,i in zip(index, names, range(len(index))):
		subplot(4,5,count)	
		plot(times, session_kappa_around_swr[index[i]], '-', color = 'blue', label = 'spdl thl')
		plot(times, session_hpc_kappa_around_swr[index[i]], '-', color = 'green', label = 'spdl hpc')
		axvline(0.0)
		xlabel("times")
		ylabel("kappa")
		legend()
		if count == 1:
			title("kappa around swr for "+mouses[j])
		count+=1
savefig("../figures/spindle_feedback/fig12.pdf")

figure(figsize = (20,10))
times = np.arange(0, 1005, 5) - 500
for j in [3]:	
	index = [i for i in range(len(datasets)) if mouses[j] in datasets[i]]
	names = [datasets[i] for i in range(len(datasets)) if mouses[j] in datasets[i]]
	print(index, names)
	cmap = plt.get_cmap('autumn')
	colors = [cmap(i) for i in np.linspace(0, 1, len(index))]	
	count = 1	
	for k, ses,i in zip(index, names, range(len(index))):
		subplot(4,5,count)	
		plot(times, session_kappa_around_swr[index[i]], '-', color = 'blue', label = 'spdl thl')
		plot(times, session_hpc_kappa_around_swr[index[i]], '-', color = 'green', label = 'spdl hpc')
		axvline(0.0)
		xlabel("times")
		ylabel("kappa")
		legend()
		if count == 1:
			title("kappa around swr for "+mouses[j])
		count+=1
savefig("../figures/spindle_feedback/fig13.pdf")


figure(figsize = (20,10))
for j in range(4):	
	index = [i for i in range(len(datasets)) if mouses[j] in datasets[i]]
	names = [datasets[i] for i in range(len(datasets)) if mouses[j] in datasets[i]]
	print(index, names)
	cmap = plt.get_cmap('autumn')
	colors = [cmap(i) for i in np.linspace(0, 1, len(index))]	

	subplot(2,4,j+1)	
	for k, ses,i in zip(index, names, range(len(index))):
		theta_mod_toplot = session_spind_mod3[ses][:,0]
		spindles_mod_toplot = session_spind_mod4[ses][:,0]
		force = session_spind_mod3[ses][:,2] + session_spind_mod4[ses][:,2]
		x = np.concatenate([theta_mod_toplot, theta_mod_toplot, theta_mod_toplot+2*np.pi, theta_mod_toplot+2*np.pi])
		y = np.concatenate([spindles_mod_toplot, spindles_mod_toplot + 2*np.pi, spindles_mod_toplot, spindles_mod_toplot + 2*np.pi])
		scatter(x, y, s = force*10., color = colors[i], label = 'theta wake')
		title(mouses[j])	
		xlabel("Hippocampl spindles phase")
		ylabel("Thalamic spindles phase")
		title("HPC == THL")
	set_lines(gca())

	subplot(2,4,j+5)
	for k, ses,i in zip(index, names, range(len(index))):
		theta_mod_toplot = session_spind_mod6[ses][:,0]
		spindles_mod_toplot = session_spind_mod5[ses][:,0]
		force = session_spind_mod5[ses][:,2] + session_spind_mod6[ses][:,2]
		x = np.concatenate([theta_mod_toplot, theta_mod_toplot, theta_mod_toplot+2*np.pi, theta_mod_toplot+2*np.pi])
		y = np.concatenate([spindles_mod_toplot, spindles_mod_toplot + 2*np.pi, spindles_mod_toplot, spindles_mod_toplot + 2*np.pi])
		scatter(x, y, s = force*10., marker = '^', color = colors[i], label = 'theta rem')	
		xlabel("Hippocampl spindles phase")
		ylabel("Thalamic spindles phase")
		title("HPC != THL")
	set_lines(gca())

savefig("../figures/spindle_feedback/fig14.pdf")


figure(figsize = (20,10))
for j in range(4):	
	index = [i for i in range(len(datasets)) if mouses[j] in datasets[i]]
	names = [datasets[i] for i in range(len(datasets)) if mouses[j] in datasets[i]]
	print(index, names)
	cmap = plt.get_cmap('autumn')
	colors = [cmap(i) for i in np.linspace(0, 1, len(index))]	

	subplot(2,4,j+1)	
	for k, ses,i in zip(index, names, range(len(index))):
		theta_mod_toplot = session_spind_mod3[ses][:,0]
		spindles_mod_toplot = session_spind_mod6[ses][:,0]
		force = session_spind_mod3[ses][:,2] + session_spind_mod6[ses][:,2]
		x = np.concatenate([theta_mod_toplot, theta_mod_toplot, theta_mod_toplot+2*np.pi, theta_mod_toplot+2*np.pi])
		y = np.concatenate([spindles_mod_toplot, spindles_mod_toplot + 2*np.pi, spindles_mod_toplot, spindles_mod_toplot + 2*np.pi])
		scatter(x, y, s = force*10., color = colors[i])
		title(mouses[j])	
		xlabel("(HPC == THL)")
		ylabel("(HPC != THL)")
		title("Hippocampal spindle phase")
	set_lines(gca())

	subplot(2,4,j+5)
	for k, ses,i in zip(index, names, range(len(index))):
		theta_mod_toplot = session_spind_mod4[ses][:,0]
		spindles_mod_toplot = session_spind_mod5[ses][:,0]
		force = session_spind_mod4[ses][:,2] + session_spind_mod5[ses][:,2]
		x = np.concatenate([theta_mod_toplot, theta_mod_toplot, theta_mod_toplot+2*np.pi, theta_mod_toplot+2*np.pi])
		y = np.concatenate([spindles_mod_toplot, spindles_mod_toplot + 2*np.pi, spindles_mod_toplot, spindles_mod_toplot + 2*np.pi])
		scatter(x, y, s = force*10., marker = '^', color = colors[i])	
		xlabel("(HPC == THL)")
		ylabel("(HPC != THL)")
		title("Thalamus spindle phase")
	set_lines(gca())

savefig("../figures/spindle_feedback/fig15.pdf")



figure(figsize = (20,10))
for j in range(4):	
	index = [i for i in range(len(datasets)) if mouses[j] in datasets[i]]
	names = [datasets[i] for i in range(len(datasets)) if mouses[j] in datasets[i]]
	print(index, names)
	cmap = plt.get_cmap('autumn')
	colors = [cmap(i) for i in np.linspace(0, 1, len(index))]	

	subplot(2,4,j+1)	
	count = 0
	tmp2 = []	
	for k, ses,i in zip(index, names, range(len(index))):
		tmp = (session_spind_mod5[ses][:,0] - session_spind_mod4[ses][:,0])
		tmp = np.abs(tmp)
		tmp[tmp > np.pi] = 2*np.pi - tmp[tmp > np.pi]
		y = np.vstack((np.arange(len(tmp)),np.arange(len(tmp))))+count
		x = np.vstack((np.zeros(len(tmp)),tmp))
		plot(x, y, color = colors[i])
		tmp2.append(tmp)
		count += len(tmp)
	title("OUT - IN THL PHASE")
	tmp2 = np.hstack(tmp2)
	tmp2 = tmp2[~np.isnan(tmp2)]
	print(np.mean(tmp2[0:500]))

	subplot(2,4,j+5)
	count = 0
	tmp2 = []
	for k, ses,i in zip(index, names, range(len(index))):
		tmp = (session_spind_mod6[ses][:,0] - session_spind_mod3[ses][:,0])
		tmp = np.abs(tmp)
		tmp[tmp > np.pi] = 2*np.pi - tmp[tmp > np.pi]
		y = np.vstack((np.arange(len(tmp)),np.arange(len(tmp))))+count
		x = np.vstack((np.zeros(len(tmp)),tmp))
		plot(x, y, color = colors[i])
		count += len(tmp)	
		tmp2.append(tmp)
	title("OUT - IN HPC PHASE")
	tmp2 = np.hstack(tmp2)
	tmp2 = tmp2[~np.isnan(tmp2)]
	print(np.mean(tmp2[0:500]))
	
savefig("../figures/spindle_feedback/fig16.pdf")

sess_out = []
sess_inn = []
sess_diff = []
sess_kappa = []
sess_phase = []
sess_pvalue = []

figure(figsize = (20,10))
for j in range(4):	
	index = [i for i in range(len(datasets)) if mouses[j] in datasets[i]]
	names = [datasets[i] for i in range(len(datasets)) if mouses[j] in datasets[i]]
	print(index, names)
	cmap = plt.get_cmap('autumn')
	colors = [cmap(i) for i in np.linspace(0, 1, len(index))]	

	subplot(2,4,j+1, projection = 'polar')		
	count = 0
	for k, ses,i in zip(index, names, range(len(index))):
		out = session_spind_mod5[ses][:,0]
		inn = session_spind_mod4[ses][:,0]
		out[out<0.0] += 2*np.pi
		inn[inn<0.0] += 2*np.pi
		tmp = out - inn
		tmp[np.abs(tmp)<np.pi] *= -1.0
		tmp[tmp>np.pi] = 2*np.pi - tmp[tmp>np.pi] 
		tmp[tmp<-np.pi] = -2*np.pi - tmp[tmp<-np.pi]

		r = np.arange(len(tmp)) + count	+ 1
		for n in range(len(tmp)):					
			phase = np.linspace(out[n], out[n] + tmp[n], 100)
			plot(phase, np.ones(100)*r[n], color = colors[i])

			sess_out.append(out[n])
			sess_inn.append(inn[n])
			sess_diff.append(tmp[n])
			sess_kappa.append(session_spind_mod5[ses][n,2]+session_spind_mod4[ses][n,2])
			sess_phase.append(phase)
			sess_pvalue.append([session_spind_mod5[ses][n,1],session_spind_mod4[ses][n,1]])
			# plot(out[n], r[n], 'o')
			# plot(inn[n], r[n], '^')
		count += len(tmp)
		
	title("OUT - IN THL PHASE")
	
	subplot(2,4,j+5, projection = 'polar')
	count = 0
	for k, ses,i in zip(index, names, range(len(index))):				
		out = session_spind_mod6[ses][:,0]
		inn = session_spind_mod3[ses][:,0]
		out[out<0.0] += 2*np.pi
		inn[inn<0.0] += 2*np.pi
		tmp = out - inn
		tmp[np.abs(tmp)<np.pi] *= -1.0
		tmp[tmp>np.pi] = 2*np.pi - tmp[tmp>np.pi] 
		tmp[tmp<-np.pi] = -2*np.pi - tmp[tmp<-np.pi]

		r = np.arange(len(tmp)) + count	+ 1
		for n in range(len(tmp)):					
			phase = np.linspace(out[n], out[n] + tmp[n], 100)

			plot(phase, np.ones(100)*r[n], color = colors[i])
			# plot(out[n], r[n], 'o')
			# plot(inn[n], r[n], '^')
		count += len(tmp)
	title("OUT - IN HPC PHASE")
	
sess_out 	= np.array(sess_out )
sess_inn 	= np.array(sess_inn )
sess_diff 	= np.array(sess_diff )
sess_kappa 	= np.array(sess_kappa)
sess_phase 	= np.array(sess_phase)
sess_pvalue = np.array(sess_pvalue)
savefig("../figures/spindle_feedback/fig17.pdf")

figure()
subplot(1,1,1, projection = 'polar')
# index = np.arange(len(sess_kappa))[np.logical_and(sess_kappa>0.4, sess_kappa<1.0)]
index = np.arange(len(sess_pvalue))[np.prod((sess_pvalue < 0.0001)*1.0, axis = 1) == 1.]

r = np.arange(len(index)) + 100
for n,i in zip(index,range(len(index))):
	phase = sess_phase[n]
	plot(phase, np.ones(100)*r[i])
	# plot(out[n], r[n], 'o')
	# plot(inn[n], r[n], '^')
count += len(tmp)
title("OUT - IN THL PHASE")


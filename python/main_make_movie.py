#!/usr/bin/env python

'''
	File name: main_make_movie.py
	Author: Guillaume Viejo
	Date created: 09/10/2017    
	Python Version: 3.5.2

To make shank mapping

'''

import numpy as np
import pandas as pd
# from matplotlib.pyplot import plot,show,draw
import scipy.io
from functions import *
from pylab import *
from sklearn.decomposition import PCA
import _pickle as cPickle
import neuroseries as nts

###############################################################################################################
# LOADING DATA
###############################################################################################################
data_directory 	= '/mnt/DataGuillaume/MergedData/'
datasets 		= np.loadtxt(data_directory+'datasets_ThalHpc.list', delimiter = '\n', dtype = str, comments = '#')

theta_mod, theta_ses 	= loadThetaMod('/mnt/DataGuillaume/MergedData/THETA_THAL_mod.pickle', datasets, return_index=True)
swr_mod, swr_ses 		= loadSWRMod('/mnt/DataGuillaume/MergedData/SWR_THAL_corr.pickle', datasets, return_index=True)
spind_mod, spind_ses 	= loadSpindMod('/mnt/DataGuillaume/MergedData/SPINDLE_mod.pickle', datasets, return_index=True)

nbins 					= 400
binsize					= 5
times 					= np.arange(0, binsize*(nbins+1), binsize) - (nbins*binsize)/2
swr 					= pd.DataFrame(	index = swr_ses, 
										columns = times,
										data = swr_mod)

# filtering swr_mod
swr 				= pd.DataFrame(	index = swr.index, 
									columns = swr.columns,
									data = gaussFilt(swr.values, (10,)))

# Cut swr_mod from -500 to 500
nbins 				= 200
binsize				= 5
times 				= np.arange(0, binsize*(nbins+1), binsize) - (nbins*binsize)/2
swr 				= swr.loc[:,times]

# CHECK FOR NAN
tmp1 			= swr.index[swr.isnull().any(1).values]
# copy and delete 
if len(tmp1):
	swr_modth 	= swr.drop(tmp1)

###############################################################################################################
# MOVIE + jPCA for each animal
###############################################################################################################
mouses 	= ['Mouse12', 'Mouse17', 'Mouse20', 'Mouse32']
# times 	= np.arange(0, 1005, 5) - 500 # BAD

interval_to_cut = {	'Mouse12':[89,128],
					'Mouse17':[84,123],
					'Mouse20':[92,131],
					'Mouse32':[80,125]}
movies 	= dict.fromkeys(mouses)
rXX 	= dict.fromkeys(mouses)
maps 	= dict.fromkeys(mouses)
headdir = dict.fromkeys(mouses)
adnloc 	= dict.fromkeys(mouses)
xpos 	= dict.fromkeys(mouses)
ypos 	= dict.fromkeys(mouses)
xpos_shank = dict.fromkeys(mouses)
ypos_shank = dict.fromkeys(mouses)
xpos_phase = dict.fromkeys(mouses)
ypos_phase = dict.fromkeys(mouses)

for m in mouses:	
	depth = pd.DataFrame(index = np.genfromtxt(data_directory+m+"/"+m+".depth", dtype = 'str', usecols = 0),
						data = np.genfromtxt(data_directory+m+"/"+m+".depth", usecols = 1),
						columns = ['depth'])	
	neurons 		= np.array([n for n in swr_modth.index if m in n])
	rX,phi_swr,dynamical_system = jPCA(swr_modth.loc[neurons].values, times)
	phi_swr 		= pd.DataFrame(index = swr_modth.loc[neurons].index, data = phi_swr)
	sessions 		= np.unique([n.split("_")[0] for n in neurons])	
	swr_shank 		= np.zeros((len(sessions),8,len(times)))
	nb_bins			= interval_to_cut[m][1] - interval_to_cut[m][0]
	theta_shank 	= np.zeros((len(sessions),8,nb_bins)) # that's radian bins here
	spindle_shank 	= np.zeros((len(sessions),8,nb_bins)) # that's radian bins here
	bins_phase 		= np.linspace(-np.pi, np.pi+0.00001, nb_bins+1)
	for s in sessions:				
		shank				= loadShankMapping(data_directory+m+'/'+s+'/Analysis/SpikeData.mat').flatten()
		shankIndex 			= np.array([shank[int(n.split("_")[1])]-1 for n in neurons if s in n])		
		if np.max(shankIndex) > 8 : sys.exit("Invalid shank index for thalamus" + s)
		hd_info 			= scipy.io.loadmat(data_directory+m+'/'+s+'/Analysis/HDCells.mat')['hdCellStats'][:,-1]
		hd_info_neuron		= np.array([hd_info[int(n.split("_")[1])] for n in neurons if s in n])		
		neurons_in_session 	= np.array([n for n in neurons if s in n])		
		shank_to_neurons 	= {k:[n for n in neurons_in_session[shankIndex == k]] for k in np.unique(shankIndex)}
###########################################################################################################			
# SWR MOD
###########################################################################################################					
		# for k in shank_to_neurons.keys(): 
		# 	count_total[np.where(sessions== s)[0][0],k] = len(shank_to_neurons[k])
		# 	hd_neurons[np.where(sessions== s)[0][0],k] = np.sum(hd_info_neuron[shankIndex == k])						
		# 	amplitute['swr'][np.where(sessions==s)[0][0],k] = np.mean(swr_modth.loc[shank_to_neurons[k]].max(1) - swr_modth.loc[shank_to_neurons[k]].min(1))
		# 	mu_, kappa_, pval_ = getCircularMean(phi_swr.loc[shank_to_neurons[k]].values.flatten(), 2*np.pi, 0.0)
		# 	phase_shank['swr'][np.where(sessions==s)[0][0],k] = mu_
		# 	if np.isnan(mu_): sys.exit("mu_")			
		# 	kappa_shank['swr'][np.where(sessions==s)[0][0],k] = kappa_
		# 	coherence_shank['swr'][np.where(sessions==s)[0][0],k] = getPhaseCoherence(phi_swr.loc[shank_to_neurons[k]].values.flatten())
		# 	for t in range(len(times)):
		# 		swr_shank[np.where(sessions== s)[0][0],k,t] = np.mean(swr_modth.loc[shank_to_neurons[k],times[t]])
		
		# # positive swr mod
		# neurons_pos_in_session = np.array([n for n in neurons_pos if s in n])
		# shankIndex_pos		= np.array([shank[int(n.split("_")[1])]-1 for n in neurons_pos_in_session])
		# shank_to_neurons_pos = {k:[n for n in neurons_pos_in_session[shankIndex_pos == k]] for k in np.unique(shankIndex_pos)}
		# for k in shank_to_neurons_pos.keys():
		# 	count_positive[np.where(sessions== s)[0][0],k] = float(len(shank_to_neurons_pos[k]))
		# # negative swr mod
		# neurons_neg_in_session = np.array([n for n in neurons_neg if s in n])
		# shankIndex_neg		= np.array([shank[int(n.split("_")[1])]-1 for n in neurons_neg_in_session])
		# shank_to_neurons_neg = {k:[n for n in neurons_neg_in_session[shankIndex_neg == k]] for k in np.unique(shankIndex_neg)}
		# for k in shank_to_neurons_neg.keys():
		# 	count_negative[np.where(sessions== s)[0][0],k] = float(len(shank_to_neurons_neg[k]))
###########################################################################################################			
# THETA MOD
###########################################################################################################			
		# loading theta episodes		
		theta_ep = np.genfromtxt(data_directory+m+'/'+s+'/'+s+'_rem.evt.theta')[:,0]
		theta_ep = theta_ep.reshape(len(theta_ep)//2,2)
		theta_ep = nts.IntervalSet(theta_ep[:,0], theta_ep[:,1])
		for k in shank_to_neurons.keys():
			phi = phase.loc[shank_to_neurons[k],'theta_rem'].values.astype(np.float)
			phi = phi[~np.isnan(phi)]
			phi[phi<0.0] += 2*np.pi
			mu_, kappa_, pval_ = getCircularMean(phi, 2*np.pi, 0.0)
			phase_shank['theta'][np.where(sessions==s)[0][0],k] = mu_
			if np.isnan(mu_): sys.exit("mu_")			
			kappa_shank['theta'][np.where(sessions==s)[0][0],k] = kappa_
			coherence_shank['theta'][np.where(sessions==s)[0][0],k] = getPhaseCoherence(phi)									
			index = np.digitize(phi, bins_phase)-1
			# for t in index:
			# 	theta_shank[np.where(sessions == s)[0][0],k,t] += 1.0
				
###########################################################################################################			
# SPIND HPC MOD
###########################################################################################################			
		for k in shank_to_neurons.keys():
			phi = phase.loc[shank_to_neurons[k],'spindle_hpc'].values.astype(np.float)
			phi = phi[~np.isnan(phi)]
			phi[phi<0.0] += 2*np.pi
			mu_, kappa_, pval_ = getCircularMean(phi, 2*np.pi, 0.0)
			phase_shank['spindle'][np.where(sessions==s)[0][0],k] = mu_
			if np.isnan(mu_): sys.exit("mu_")			
			kappa_shank['spindle'][np.where(sessions==s)[0][0],k] = kappa_
			coherence_shank['spindle'][np.where(sessions==s)[0][0],k] = getPhaseCoherence(phi)									
			index = np.digitize(phi, bins_phase)-1
			# for t in index:
			# 	spindle_shank[np.where(sessions == s)[0][0],k,t] += 1.0			

	for k in ['swr', 'theta', 'spindle']:
		phase_shank[k] 	= np.flip(phase_shank[k], 1)
		amplitute[k] 	= np.flip(amplitute[k], 1)
		kappa_shank[k] 	= np.flip(kappa_shank[k], 1)
		coherence_shank[k] = np.flip(coherence_shank[k], 1)


	# saving	
	movies[m] = {	'swr'	:	np.flip(swr_shank	,1),
					'theta'	:	np.flip(theta_shank	,1),
					'spindle':	np.flip(spindle_shank,1)}
	# normalize by number of neurons per shanks 
	count_positive = count_positive/(count_total+1.0)
	count_negative = count_negative/(count_total+1.0)
	hd_neurons	= hd_neurons/(count_total+1.0)
	rXX[m]		= rX
	maps[m] 	= {	'positive':		np.flip(count_positive,1)	,
					'negative':		np.flip(count_negative	,1), 
					'total':		np.flip(count_total 	,1), 
					'amplitute':	amplitute		, 
					'phase':		phase_shank		, 
					'kappa':		kappa_shank		,
					'coherence':	coherence_shank	,					
					'x'			: 	np.arange(0.0, 8*0.2, 0.2),
					'y'			: 	depth.loc[sessions].values.flatten()
					}
	headdir[m] 	= np.flip(hd_neurons, 1)
	# # where is adn
	# ind_max = np.where(hd_neurons == np.max(hd_neurons))
	# adnloc[m] 	= [ypos[m][ind_max[0][0]], xpos[m][ind_max[1][0]]]


	


def interpolate(z, x, y, inter, bbox = None):	
	xnew = np.arange(x.min(), x.max()+inter, inter)
	ynew = np.arange(y.min(), y.max()+inter, inter)
	if bbox == None:
		f = scipy.interpolate.RectBivariateSpline(y, x, z)
	else:
		f = scipy.interpolate.RectBivariateSpline(y, x, z, bbox = bbox)
	znew = f(ynew, xnew)
	return (xnew, ynew, znew)

def filter_(z, n):
	from scipy.ndimage import gaussian_filter	
	return gaussian_filter(z, n)








def softmax(x, b1 = 20.0, b2 = 0.5):
	x -= x.min()
	x /= x.max()
	return 1.0/(1.0+np.exp(-(x-b2)*b1))

def get_rgb(mapH, mapV, mapS, bound, m):
	beta_total = {	'Mouse12':[40.0,0.25],
					'Mouse17':[40.0,0.2],
					'Mouse20':[40.0,0.2],
					'Mouse32':[100.0,0.3]}

	beta_coh = {	'Mouse12':[50.0,0.3],
					'Mouse17':[20.0,0.3],
					'Mouse20':[20.0,0.3],
					'Mouse32':[90.0,0.4]}
	# mapH : phase
	# mapV : total
	# mapS : coherence
	from matplotlib.colors import hsv_to_rgb	
	mapH -= mapH.min()
	mapH /= mapH.max()
	mapV -= mapV.min()
	mapV /= mapV.max()
	mapS -= mapS.min()
	mapS /= mapS.max()
	mapV = softmax(mapV, beta_total[m][0], beta_total[m][1])
	mapS = softmax(mapS, beta_coh[m][0], beta_coh[m][1])
	H 	= (1-mapH)*bound
	S 	= mapS	
	V 	= mapV
	HSV = np.dstack((H,S,V))	
	RGB = hsv_to_rgb(HSV)	
	return RGB

# sys.exit()

for m in mouses:
	figure()
	for k, i in zip(['swr', 'theta', 'spindle'], range(3)):
		subplot(1,3,i+1)
		xnew, ynew, total = interpolate(maps[m]['total'].copy(), maps[m]['x'], maps[m]['y'], 0.01)	
		xnew, ynew, sinn = interpolate(np.sin(maps[m]['phase'][k].copy()), maps[m]['x'], maps[m]['y'], 0.01)
		xnew, ynew, coss = interpolate(np.cos(maps[m]['phase'][k].copy()), maps[m]['x'], maps[m]['y'], 0.01)	
		sinn = gaussFilt(sinn, (10,10))
		coss = gaussFilt(coss, (10,10))	
		phi = np.arctan2(sinn, coss)	
		phi[phi<0.0] += 2*np.pi
		xnew, ynew, coh = interpolate(maps[m]['coherence'][k].copy(), maps[m]['x'], maps[m]['y'], 0.01)	
		imshow(get_rgb(phi.copy(), total.copy(), coh.copy(), 1.0, m), aspect = 'equal',origin = 'upper', extent = (xnew[0], xnew[-1], ynew[-1], ynew[0]))


		xnew, ynew, head = interpolate(headdir[m].copy(), maps[m]['x'], maps[m]['y'], 0.01)
		head[head < np.percentile(head, 90)] = 0.0
		contour(head, aspect = 'equal',origin = 'upper', extent = (xnew[0], xnew[-1], ynew[-1], ynew[0]))
		title(k)
	savefig("../figures/map_phase_"+m+"_all.pdf")














sys.exit()


# arrows

for m in mouses:
	figure()

	xnew, ynew, amp = interpolate(maps[m]['amplitute'].copy(), maps[m]['x'], maps[m]['y'], 0.01)
	amp = filter_(amp, 5)
	xnew, ynew, total = interpolate(maps[m]['total'].copy(), maps[m]['x'], maps[m]['y'], 0.01)
	total = filter_(total, 5)
	xnew, ynew, coh = interpolate(maps[m]['coherence'].copy(), maps[m]['x'], maps[m]['y'], 0.01)
	coh = filter_(coh, 5)	
	imshow(get_rgb(amp.copy(), total.copy(), coh.copy(), 0.83), aspect = 'equal',origin = 'upper', extent = (xnew[0], xnew[-1], ynew[-1], ynew[0]))

	xnew, ynew, sinn = interpolate(np.sin(maps[m]['phase'].copy()), maps[m]['x'], maps[m]['y'], 0.06)
	xnew, ynew, coss = interpolate(np.cos(maps[m]['phase'].copy()), maps[m]['x'], maps[m]['y'], 0.06)
	xnew, ynew, coh = interpolate(maps[m]['coherence'].copy(), maps[m]['x'], maps[m]['y'], 		0.06)
	coh = filter_(coh, 5)
	X, Y = np.meshgrid(xnew, ynew)
	quiver(X, Y, coss*coh, sinn*coh, units = 'xy', linewidth = 100)

	xnew, ynew, head = interpolate(headdir[m].copy(), maps[m]['x'], maps[m]['y'], 0.01)
	head[head < np.percentile(head, 90)] = 0.0
	contour(head, aspect = 'equal',origin = 'upper', extent = (xnew[0], xnew[-1], ynew[-1], ynew[0]))

	# savefig("../figures/map_phase_"+m+".pdf")
	














# from matplotlib import animation, rc
# from IPython.display import HTML, Image

# m = mouses[0]
# rc('animation', html='html5')
# fig, axes = plt.subplots(1,3)
# images1 = []

# for i,e in zip(range(3),['swr','spindle','theta']):
# 	images1.append(axes[i].imshow(movies[m][e][0], aspect = 'equal', cmap = 'jet'))		
# 	axes[i].set_title(e)
	
# def init():
# 	for i,e in zip(range(3),['swr','spindle','theta']):		
# 		images1[i].set_data(movies[m][e][0])				
# 	return images1

# def animate(t):		
# 	for i,e in zip(range(3),['swr','spindle','theta']):
# 		images1[i].set_data(movies[m][e][t])		
# 	return images1

# anim = animation.FuncAnimation(fig, animate, init_func=init,
# 						   frames=interval_to_cut[m][1]-interval_to_cut[m][0], interval=0, blit=True, repeat_delay = 0)

# show()



# anim.save('../figures/animation_swr_mod_jpca.gif', writer='imagemagick', fps=60)

####################################################################################
# MAPS
####################################################################################


# figure(figsize = (16,10))
# for i, m in zip(range(len(mouses)), mouses):
# 	subplot(1,4,i+1)
m = 'Mouse12'
imshow(get_rgb(maps[m]['amplitute'].copy(), maps[m]['total'].copy()), aspect = 'equal',origin = 'upper', extent = (xpos[m][0], xpos[m][-1], ypos[m][-1], ypos[m][0]))
xticks(xpos[m][np.arange(0, maps[m]['amplitute'].shape[1],20)])
yticks(ypos[m][np.arange(0, maps[m]['amplitute'].shape[0],20)])
X, Y = np.meshgrid(xpos_phase[m], ypos_phase[m])
quiver(X, Y, np.cos(maps[m]['phase'])*maps[m]['coherence'], np.sin(maps[m]['phase'])*maps[m]['coherence'])
headdir[m][headdir[m] < np.percentile(headdir[m], 80)] = 0.0
contour(headdir[m])
show()
sys.exit()
title(m)

savefig("../figures/map_mouse12_phase.pdf")	
# savefig("../figures/map_swr_density_neurons.pdf")
show()
sys.exit()

figure(figsize = (16,10))
for i, m in zip(range(len(mouses)), mouses):
	subplot(2,4,i+1)
	imshow(get_rgb(maps[m]['positive'], maps[m]['total']), aspect = 'equal')
	headdir[m][headdir[m] < np.percentile(headdir[m], 80)] = 0.0	
	contour(headdir[m])
	xticks(np.arange(0, maps[m]['positive'].shape[1],20), xpos[m][np.arange(0, maps[m]['positive'].shape[1],20)])
	yticks(np.arange(0, maps[m]['positive'].shape[0],20), ypos[m][np.arange(0, maps[m]['positive'].shape[0],20)])
	title(m)

	subplot(2,4,i+5)
	imshow(get_rgb(maps[m]['negative'], maps[m]['total']), aspect = 'equal')
	contour(headdir[m])		
	xticks(np.arange(0, maps[m]['negative'].shape[1],20), xpos[m][np.arange(0, maps[m]['negative'].shape[1],20)])
	yticks(np.arange(0, maps[m]['negative'].shape[0],20), ypos[m][np.arange(0, maps[m]['negative'].shape[0],20)])
	
savefig("../figures/map_swr_density_neurons.pdf")
show()

figure(figsize = (16,10))
for i, m in zip(range(len(mouses)), mouses):
	subplot(1,4,i+1)
	imshow(headdir[m], cmap = 'jet', aspect = 'equal')
	xticks(np.arange(0, headdir[m].shape[1],20), xpos[m][np.arange(0, headdir[m].shape[1],20)])
	yticks(np.arange(0, headdir[m].shape[0],20), ypos[m][np.arange(0, headdir[m].shape[0],20)])	
	title(m+"\n y="+str(np.around(adnloc[m][0],2))+"\n"+"x="+str(np.around(adnloc[m][1],2)))

savefig("../figures/map_adn_location.pdf")
show()



# SWR MOD

from matplotlib import animation, rc
from IPython.display import HTML, Image

rc('animation', html='html5')
fig, axes = plt.subplots(2,4)
lines1 = []
lines2 = []
images = []
for i in range(len(mouses)):
	lines1.append(axes[0,i].plot([],[],'o-')[0])
	lines2.append(axes[0,i].plot([],[],'o-')[0])
	axes[0,i].set_xlim(-500, 500)
	axes[0,i].set_ylim(rXX[mouses[i]].min(), rXX[mouses[i]].max())
	images.append(axes[1,i].imshow(movies[mouses[i]][0], aspect = 'auto', cmap = 'jet'))

def init():
	for i, m in zip(range(len(mouses)), mouses):
		images[i].set_data(movies[m][0])
		lines1[i].set_data(times[0], rXX[m][0,0])
		lines2[i].set_data(times[0], rXX[m][0,1])
	return images+lines1+lines2

def animate(t):		
	for i, m in zip(range(len(mouses)), mouses):
		images[i].set_data(movies[m][t])
		lines1[i].set_data(times[0:t], rXX[m][0:t,0])
		lines2[i].set_data(times[0:t], rXX[m][0:t,1])	
	return images+lines1+lines2

anim = animation.FuncAnimation(fig, animate, init_func=init,
						   frames=len(movie), interval=10, blit=True, repeat_delay = 1000)

show()

# anim.save('../figures/animation_swr_mod_jpca.gif', writer='imagemagick', fps=60)



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
import sys

###############################################################################################################
# LOADING DATA
###############################################################################################################
data_directory 	= '/mnt/DataGuillaume/MergedData/'
datasets 		= np.loadtxt(data_directory+'datasets_ThalHpc.list', delimiter = '\n', dtype = str, comments = '#')

theta_mod, theta_ses 	= loadThetaMod('/mnt/DataGuillaume/MergedData/THETA_THAL_mod.pickle', datasets, return_index=True)
swr_mod, swr_ses 		= loadSWRMod('/mnt/DataGuillaume/MergedData/SWR_THAL_corr.pickle', datasets, return_index=True)
spind_mod, spind_ses 	= loadSpindMod('/mnt/DataGuillaume/MergedData/SPINDLE_mod.pickle', datasets, return_index=True)
spike_spindle_phase 	= cPickle.load(open('/mnt/DataGuillaume/MergedData/SPIKE_SPINDLE_PHASE.pickle', 'rb'))		
spike_theta_phase 		= cPickle.load(open('/mnt/DataGuillaume/MergedData/SPIKE_THETA_PHASE.pickle', 'rb'))		

nbins 					= 400
binsize					= 5
times 					= np.arange(0, binsize*(nbins+1), binsize) - (nbins*binsize)/2

theta 					= pd.DataFrame(	index = theta_ses['rem'], 
									columns = ['phase', 'pvalue', 'kappa'],
									data = theta_mod['rem'])

# filtering swr_mod
swr 					= pd.DataFrame(	columns = swr_ses, 
										index = times,
										data = gaussFilt(swr_mod, (10,)).transpose())


# Cut swr_mod from -500 to 500
swr = swr.loc[-500:500]
# CHECK FOR NAN
tmp1 			= swr.columns[swr.isnull().any()].values
tmp2 			= theta.index[theta.isnull().any(1)].values
# CHECK P-VALUE 
tmp3	 		= theta.index[(theta['pvalue'] > 1).values].values
tmp 			= np.unique(np.concatenate([tmp1,tmp2,tmp3]))
# copy and delete 
if len(tmp):
	swr_modth 	= swr.drop(tmp, axis = 1)
	theta_modth = theta.drop(tmp, axis = 0)

swr_modth_copy 	= swr_modth.copy()
neuron_index = swr_modth.columns
times = swr_modth.loc[-500:500].index.values

###############################################################################################################
# MOVIE + jPCA for each animal
###############################################################################################################
mouses 	= ['Mouse12', 'Mouse17', 'Mouse20', 'Mouse32']
# times 	= np.arange(0, 1005, 5) - 500 # BAD

interval_to_cut = {	'Mouse12':[89,128],
					'Mouse17':[84,123],
					'Mouse20':[92,131],
					'Mouse32':[80,125]}

movies 		= dict.fromkeys(mouses)
rXX 		= dict.fromkeys(mouses)
maps 		= dict.fromkeys(mouses)
headdir 	= dict.fromkeys(mouses)
adnloc 		= dict.fromkeys(mouses)
xpos 		= dict.fromkeys(mouses)
ypos 		= dict.fromkeys(mouses)
xpos_shank 	= dict.fromkeys(mouses)
ypos_shank 	= dict.fromkeys(mouses)
xpos_phase 	= dict.fromkeys(mouses)
ypos_phase 	= dict.fromkeys(mouses)
theta_dens	= dict.fromkeys(mouses)
hd_neurons_index = []

for m in mouses:	
	print(m)
	depth = pd.DataFrame(index = np.genfromtxt(data_directory+m+"/"+m+".depth", dtype = 'str', usecols = 0),
						data = np.genfromtxt(data_directory+m+"/"+m+".depth", usecols = 1),
						columns = ['depth'])	
	neurons 		= np.array([n for n in neuron_index if m in n])
	sessions 		= np.unique([n.split("_")[0] for n in neuron_index if m in n])	
	nb_bins 		= 201
	swr_shank 		= np.zeros((len(sessions),8,nb_bins))
	# nb_bins			= interval_to_cut[m][1] - interval_to_cut[m][0]	
	theta_shank 	= np.zeros((len(sessions),8,30)) # that's radian bins here
	spindle_shank 	= np.zeros((len(sessions),8,30)) # that's radian bins here
	bins_phase 		= np.linspace(0.0, 2*np.pi+0.00001, 31)
	count_total 	= np.zeros((len(sessions),8))	
	hd_neurons 		= np.zeros((len(sessions),8))
	amplitute		= np.zeros((len(sessions),8))
	mod_theta 		= np.zeros((len(sessions),8))
###########################################################################################################			
# JPCA
###########################################################################################################				
	rX,phi_swr,dynamical_system = jPCA(swr_modth[neurons].values.transpose(), times)
	phi_swr 		= pd.DataFrame(index = neurons, data = phi_swr)
###########################################################################################################			
# VARIOUS
###########################################################################################################				
	for s in sessions:				
		generalinfo 		= scipy.io.loadmat(data_directory+m+"/"+s+'/Analysis/GeneralInfo.mat')
		shankStructure 		= loadShankStructure(generalinfo)
		spikes,shank		= loadSpikeData(data_directory+m+"/"+s+'/Analysis/SpikeData.mat', shankStructure['thalamus'])				
		hd_info 			= scipy.io.loadmat(data_directory+m+'/'+s+'/Analysis/HDCells.mat')['hdCellStats'][:,-1]
		hd_info_neuron		= np.array([hd_info[n] for n in spikes.keys()])
		shankIndex 			= np.array([shank[n] for n in spikes.keys()]).flatten()
		if np.max(shankIndex) > 8 : sys.exit("Invalid shank index for thalamus" + s)				
		shank_to_neurons 	= {k:np.array(list(spikes.keys()))[shankIndex == k] for k in np.unique(shankIndex)}		

		for k in shank_to_neurons.keys(): 						
			count_total[np.where(sessions== s)[0][0],k] = len(shank_to_neurons[k])			
			hd_neurons[np.where(sessions== s)[0][0],k] = np.sum(hd_info_neuron[shankIndex == k])
			mod_theta[np.where(sessions== s)[0][0],k] = (theta.loc[[s+'_'+str(i) for i in shank_to_neurons[k]]]['pvalue'] < 0.05).sum()
			# amplitute[np.where(sessions==s)[0][0],k] = (swr.loc[shank_to_neurons[k]].var(1)).mean()
###########################################################################################################			
# SWR MOD
###########################################################################################################		
		neurons_mod_in_s 	= np.array([n for n in neurons if s in n])								
		shank_to_neurons 	= {k:np.array([n for n in neurons_mod_in_s if shankIndex[int(n.split("_")[1])] == k]) for k in np.unique(shankIndex)}		

		for k in shank_to_neurons.keys():
			# if np.sum(hd_info_neuron[[int(n.split("_")[1]) for n in shank_to_neurons[k]]]):
			# print(s, k, len(shank_to_neurons[k]))
			# if s == 'Mouse17-130204': sys.exit()
			if len(shank_to_neurons[k]):				
				swr_shank[np.where(sessions== s)[0][0],k] = swr_modth[shank_to_neurons[k]].mean(1).values
		
###########################################################################################################			
# THETA MOD
###########################################################################################################							
		for k in shank_to_neurons.keys():
			if len(shank_to_neurons[k]):
				for n in shank_to_neurons[k]:					
					phi = spike_theta_phase['rem'][n]
					phi[phi<0.0] += 2*np.pi
					index = np.digitize(phi, bins_phase)-1
					for t in index:
					 	theta_shank[np.where(sessions == s)[0][0],k,t] += 1.0				
###########################################################################################################			
# SPIND HPC MOD
###########################################################################################################					
		for k in shank_to_neurons.keys():
			if len(shank_to_neurons[k]):
				for n in shank_to_neurons[k]:
					if n in list(spike_spindle_phase.keys()):
						phi = spike_spindle_phase['hpc'][n]
						phi[phi<0.0] += 2*np.pi
						index = np.digitize(phi, bins_phase)-1
						for t in index:
						 	spindle_shank[np.where(sessions == s)[0][0],k,t] += 1.0			

	
	for t in range(len(times)):
		swr_shank[:,:,t] = np.flip(swr_shank[:,:,t], 1)
	for t in range(theta_shank.shape[-1]):
		theta_shank[:,:,t] = np.flip(theta_shank[:,:,t], 1)
		spindle_shank[:,:,t] = np.flip(spindle_shank[:,:,t], 1)

	# saving	
	movies[m] = {	'swr'	:	swr_shank		,
					'theta'	:	theta_shank		,
					'spindle':	spindle_shank	}

	hd_neurons	= hd_neurons/(count_total+1.0)
	mod_theta 	= mod_theta/(count_total+1.0)
	rXX[m]		= rX
	maps[m] 	= {	'total':		np.flip(count_total,1), 
					'x'			: 	np.arange(0.0, 8*0.2, 0.2),
					'y'			: 	depth.loc[sessions].values.flatten()
					}
	headdir[m] 	= np.flip(hd_neurons, 1)
	theta_dens[m] = np.flip(mod_theta, 1)

for m in movies.keys():
	datatosave = {	'movies':movies[m],
					'total':maps[m]['total'],
					'x':maps[m]['x'],
					'y':maps[m]['y'],
					'headdir':headdir[m],
					'jpc':rXX[m],
					'theta_dens':theta_dens[m]

					}
	cPickle.dump(datatosave, open("../data/maps/"+m+".pickle", 'wb'))	









sys.exit()




m = 'Mouse12'
space = 0.01

thl_lines = np.load("../figures/thalamus_lines.mat.npy").sum(2)
xlines, ylines, thl_lines = interpolate(thl_lines, 	np.linspace(maps[m]['x'].min(), maps[m]['x'].max(), thl_lines.shape[1]),
 													np.linspace(maps[m]['y'].min(), maps[m]['y'].max(), thl_lines.shape[0]), 0.001)
thl_lines -= thl_lines.min()
thl_lines /= thl_lines.max()
thl_lines[thl_lines>0.6] = 1.0
thl_lines[thl_lines<=0.6] = 0.0
xnew, ynew, total = interpolate(maps[m]['total'].copy(), maps[m]['x'], maps[m]['y'], space)
# total -= total.min()
# total /= total.max()
total = softmax(total, 20.0, 0.2)


for k in movies[m].keys():
	movies[m][k] = filter_(movies[m][k], (2,2,5))

filmov = dict.fromkeys(movies[m].keys())
for k in filmov:	
	tmp = []
	for t in range(movies[m][k].shape[-1]):
		# frame = movies[m][k][:,:,t] / (maps[m]['total']+1.0)
		frame = movies[m][k][:,:,t]		
		xnew, ynew, frame = interpolate(frame, maps[m]['x'], maps[m]['y'], space)		
		tmp.append(frame)
	tmp = np.array(tmp)
	filmov[k] = filter_(tmp, 5)
	filmov[k] = filmov[k] - np.min(filmov[k])
	filmov[k] = filmov[k] / np.max(filmov[k] + 1e-8)
	filmov[k] = softmax(filmov[k], 10, 0.5)

xnew, ynew, head = interpolate(headdir[m].copy(), maps[m]['x'], maps[m]['y'], space)
head[head < np.percentile(head, 90)] = 0.0



# sys.exit()

# figure()
# index = np.arange(0,20,1)+90
 
# for i in range(len(index)):
# 	subplot(4,5,i+1)
# 	# imshow(get_rgb(filmov['swr'][index[i]].copy(), total.copy(), np.ones_like(total), 0.83),
# 	imshow(filmov['swr'][index[i]].copy(),
# 			aspect = 'auto', 
# 			origin = 'upper',
# 			cmap = 'jet', vmin = 0.0, vmax = 1.0)
# 			# extent = (xnew[0], xnew[-1], ynew[-1], ynew[0]))
# 	title("t = "+str(times[index[i]])+" ms")
# 	# contour(head, aspect = 'equal',origin = 'upper', extent = (xnew[0], xnew[-1], ynew[-1], ynew[0]))	
# 	# contour(thl_lines, aspect = 'equal', origin = 'upper', extent = (xlines[0], xlines[-1], ylines[-1], ylines[0]), colors = 'white')	
# 	# show(thl_lines, aspect = 'equal', origin = 'upper', extent = (xlines[0], xlines[-1], ylines[-1], ylines[0]))
# show()



from matplotlib import animation, rc
from IPython.display import HTML, Image

rc('animation', html='html5')
fig, axes = plt.subplots(1,1)
images = [axes.imshow(get_rgb(filmov['swr'][0].copy(), np.ones_like(total), total, 0.65), vmin = 0.0, vmax = 1.0, aspect = 'equal', origin = 'upper', extent = (xnew[0], xnew[-1], ynew[-1], ynew[0]))]
# images = [axes.imshow(filmov['swr'][0], aspect = 'equal', origin = 'upper', cmap = 'jet', vmin = 0.0, vmax = 1.0, extent = (xnew[0], xnew[-1], ynew[-1], ynew[0]))]
axes.contour(head, aspect = 'equal',origin = 'upper', extent = (xnew[0], xnew[-1], ynew[-1], ynew[0]), cmap = 'gist_gray')
axes.contour(thl_lines, aspect = 'equal', origin = 'upper', extent = (xlines[0], xlines[-1], ylines[-1], ylines[0]), colors = 'white')	
def init():
	images[0].set_data(get_rgb(filmov['swr'][0].copy(), np.ones_like(total), total, 0.65))
	# images[0].set_data(filmov['swr'][0])
	return images
		
def animate(t):
	images[0].set_data(get_rgb(filmov['swr'][t].copy(), np.ones_like(total), total, 0.65))
	# images[0].set_data(filmov['swr'][t])		
	return images
	
anim = animation.FuncAnimation(fig, animate, init_func=init,
						   frames=range(len(times)), interval=0, blit=False, repeat_delay = 5000)

anim.save('../figures/swr_mod_'+m+'.gif', writer='imagemagick', fps=60)
show()
sys.exit()








sys.exit()

from matplotlib import animation, rc
from IPython.display import HTML, Image

rc('animation', html='html5')
fig, axes = plt.subplots(1,3)
images = []
for k, i in zip(['swr', 'theta', 'spindle'], range(3)):
	images.append(axes[i].imshow(filmov[k][0], aspect = 'auto', cmap = 'jet', origin = 'upper', extent = (xnew[0], xnew[-1], ynew[-1], ynew[0])))
	contour(head, aspect = 'equal',origin = 'upper', extent = (xnew[0], xnew[-1], ynew[-1], ynew[0]))		
def init():
	for i in range(3): images[i].set_data(filmov[k][0])
	contour(head, aspect = 'equal',origin = 'upper', extent = (xnew[0], xnew[-1], ynew[-1], ynew[0]))	
	return images
		
def animate(t):
	for i in range(3): images[i].set_data(filmov[k][t])
	contour(head, aspect = 'equal',origin = 'upper', extent = (xnew[0], xnew[-1], ynew[-1], ynew[0]))	
	return images
	
anim = animation.FuncAnimation(fig, animate, init_func=init,
						   frames=range(len(times)), interval=0, blit=True, repeat_delay = 0)


sys.exit()






m = 'Mouse12'

images = []
# for i in range(len(mouses)):
# 	lines1.append(axes[0,i].plot([],[],'o-')[0])
# 	lines2.append(axes[0,i].plot([],[],'o-')[0])
# 	axes[0,i].set_xlim(-500, 500)
# 	axes[0,i].set_ylim(rXX[mouses[i]].min(), rXX[mouses[i]].max())
images.append(axes.imshow(movies[m]['spindle'][:,:,0], aspect = 'auto', cmap = 'jet'))

def init():
	# for i, m in zip(range(len(mouses)), mouses):
	# 	images[i].set_data(movies[m][0])
	# 	lines1[i].set_data(times[0], rXX[m][0,0])
	# 	lines2[i].set_data(times[0], rXX[m][0,1])
	# return images+lines1+lines2
	images[0].set_data(movies[m]['spindle'][:,:,0])
	return images

def animate(t):		
	# for i, m in zip(range(len(mouses)), mouses):
	# 	images[i].set_data(movies[m][t])
	# 	lines1[i].set_data(times[0:t], rXX[m][0:t,0])
	# 	lines2[i].set_data(times[0:t], rXX[m][0:t,1])	
	images[0].set_data(movies[m]['spindle'][:,:,t])
	return images

anim = animation.FuncAnimation(fig, animate, init_func=init,
						   frames=movies[m]['spindle'].shape[-1], interval=0, blit=True, repeat_delay = 1)

show()

# anim.save('../figures/animation_swr_mod_jpca.gif', writer='imagemagick', fps=60)



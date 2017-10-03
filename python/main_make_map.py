#!/usr/bin/env python

'''
	File name: main_make_map.py
	Author: Guillaume Viejo
	Date created: 28/09/2017    
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

###############################################################################################################
# LOADING DATA
###############################################################################################################
data_directory 			= '/mnt/DataGuillaume/MergedData/'
datasets 				= np.loadtxt(data_directory+'datasets_ThalHpc.list', delimiter = '\n', dtype = str, comments = '#')

theta_mod, theta_ses 	= loadThetaMod('/mnt/DataGuillaume/MergedData/THETA_THAL_mod.pickle', datasets, return_index=True)
swr_mod, swr_ses 		= loadSWRMod('/mnt/DataGuillaume/MergedData/SWR_THAL_corr.pickle', datasets, return_index=True)
spind_mod, spind_ses 	= loadSpindMod('/mnt/DataGuillaume/MergedData/SPINDLE_mod.pickle', datasets, return_index=True)

swr 					= pd.DataFrame(	index = swr_ses, 
										columns = np.arange(-500, 505, 5),
										data = swr_mod)

phase 					= pd.DataFrame(index = theta_ses['wake'], columns = ['theta_wake', 'theta_rem', 'spindle_hpc', 'spindle_thl', 'swr'])
phase.loc[theta_ses['wake'],'theta_wake'] = theta_mod['wake'][:,0]
phase.loc[theta_ses['rem'], 'theta_rem'] = theta_mod['rem'][:,0]
phase.loc[spind_ses['hpc'], 'spindle_hpc'] = spind_mod['hpc'][:,0]
phase.loc[spind_ses['thl'], 'spindle_thl'] = spind_mod['thl'][:,0]

pvalue 					= pd.DataFrame(index = theta_ses['wake'], columns = ['theta_wake', 'theta_rem', 'spindle_hpc', 'spindle_thl', 'swr'])
pvalue.loc[theta_ses['wake'], 'theta_wake'] = theta_mod['wake'][:,1]
pvalue.loc[theta_ses['rem'], 'theta_rem'] = theta_mod['rem'][:,1]
pvalue.loc[spind_ses['hpc'], 'spindle_hpc'] = spind_mod['hpc'][:,1]
pvalue.loc[spind_ses['thl'], 'spindle_thl'] = spind_mod['thl'][:,1]

kappa 					= pd.DataFrame(index = theta_ses['wake'], columns = ['theta_wake', 'theta_rem', 'spindle_hpc', 'spindle_thl', 'swr'])
kappa.loc[theta_ses['wake'], 'theta_wake'] = theta_mod['wake'][:,2]
kappa.loc[theta_ses['rem'], 'theta_rem'] = theta_mod['rem'][:,2]
kappa.loc[spind_ses['hpc'], 'spindle_hpc'] = spind_mod['hpc'][:,2]
kappa.loc[spind_ses['thl'], 'spindle_thl'] = spind_mod['thl'][:,2]

# filtering swr_mod
swr 				= pd.DataFrame(	index = swr.index, 
									columns = swr.columns,
									data = gaussFilt(swr.values, (10,)))

# CHECK FOR NAN
tmp1 			= swr.index[swr.isnull().any(1).values]
# copy and delete 
if len(tmp1):
	swr_modth 	= swr.drop(tmp1)

###############################################################################################################
# MOVIE + jPCA for each animal
###############################################################################################################
mouses 	= ['Mouse12', 'Mouse17', 'Mouse20', 'Mouse32']
times 	= np.arange(0, 1005, 5) - 500 # BAD

movies 	= dict.fromkeys(mouses)
rXX 	= dict.fromkeys(mouses)
maps 	= dict.fromkeys(mouses)

for m in mouses:	
	depth = pd.DataFrame(index = np.genfromtxt(data_directory+m+"/"+m+".depth", dtype = 'str', usecols = 0),
						data = np.genfromtxt(data_directory+m+"/"+m+".depth", usecols = 1),
						columns = ['depth'])	
	neurons 	= np.array([s for s in swr_modth.index if m in s])
	sessions 	= np.unique([n.split("_")[0] for n in neurons])	
	swr_shank 	= np.zeros((len(sessions),8,len(times)))
	# positive and negative modulation for each mouse
	bornsup 	= np.percentile(swr_modth.loc[neurons], 70)
	borninf 	= np.percentile(swr_modth.loc[neurons], 30)
	neurons_pos  = np.array([n for n in swr_modth.index if m in n and swr_modth.loc[n,0] > bornsup])
	neurons_neg  = np.array([n for n in swr_modth.index if m in n and swr_modth.loc[n,0] < borninf])
	count_positive = np.zeros((len(sessions),8))
	count_negative = np.zeros((len(sessions),8))

	# map neuron to a session and a shank with a dual index
	for s in sessions:				
		shank				= loadShankMapping(data_directory+m+'/'+s+'/Analysis/SpikeData.mat').flatten()
		shankIndex 			= np.array([shank[int(n.split("_")[1])]-1 for n in neurons if s in n])
		if np.max(shankIndex) > 8 : sys.exit("Invalid shank index for thalamus" + s)
		shank_to_neurons 	= {k:[s+"_"+str(i) for i in np.where(shankIndex == k)[0]] for k in np.unique(shankIndex)}		
		for k in shank_to_neurons.keys(): 
			for t in range(len(times)):
				swr_shank[np.where(sessions== s)[0][0],k,t] = swr_modth.loc[shank_to_neurons[k],times[t]].mean()
		# positive
		shankIndex_pos		= np.array([shank[int(n.split("_")[1])]-1 for n in neurons_pos if s in n])		
		shank_to_neurons_pos = {k:[s+"_"+str(i) for i in np.where(shankIndex_pos == k)[0]] for k in np.unique(shankIndex_pos)}
		for k in shank_to_neurons_pos.keys(): 
			count_positive[np.where(sessions== s)[0][0],k] += float(len(shank_to_neurons_pos[k]))
		# negative
		shankIndex_neg		= np.array([shank[int(n.split("_")[1])]-1 for n in neurons_neg if s in n])		
		shank_to_neurons_neg = {k:[s+"_"+str(i) for i in np.where(shankIndex_neg == k)[0]] for k in np.unique(shankIndex_neg)}
		for k in shank_to_neurons_neg.keys(): 			
			count_negative[np.where(sessions== s)[0][0],k] += float(len(shank_to_neurons_neg[k]))
		
	# for movies
	rX,dynamical_system = jPCA(swr_modth.loc[neurons].values, times)

	movie 		= np.array([scipy.misc.imresize(swr_shank[:,:,t], 5.0, interp = 'bilinear') for t in range(swr_shank.shape[-1])])	
	from scipy.ndimage import gaussian_filter
	movie 		= gaussian_filter(movie, 3)

	

	count_positive = scipy.misc.imresize(count_positive, 5.0)	
	count_positive = gaussian_filter(count_positive, 3)
	count_negative = scipy.misc.imresize(count_negative, 5.0)	
	count_negative = gaussian_filter(count_negative, 3)



	# saving
	movies[m] 	= movie
	rXX[m]		= rX
	maps[m] 	= {'positive':count_positive,'negative':count_negative}


figure()
for i, m in zip(range(len(mouses)), mouses):
	subplot(2,4,i+1)
	imshow(maps[m]['positive'], cmap = 'jet')
	title(m)
	subplot(2,4,i+5)
	imshow(maps[m]['negative'], cmap = 'jet')

show()





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
	images.append(axes[1,i].imshow(movies[mouses[i]][0], aspect = 'auto', cmap = 'gist_heat'))

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

anim.save('../figures/animation_swr_mod_jpca.gif', writer='imagemagick', fps=60)



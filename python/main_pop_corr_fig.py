

import numpy as np
import pandas as pd
# from matplotlib.pyplot import plot,show,draw
import scipy.io
from functions import *
from pylab import *
from sklearn.decomposition import PCA
import _pickle as cPickle
import time
import os, sys
import ipyparallel
import matplotlib.cm as cm

###############################################################################################################
# TO LOAD
###############################################################################################################
main_dir = "../data/corr_pop/"
files = os.listdir(main_dir)

ywak = []
yrem = []
yrip = []

toplot = {}

for f in files:
	data = cPickle.load(open(main_dir+f, 'rb'))
	theta_wake_corr = data['theta_wake_corr']
	rip_corr 		= data['rip_corr']
	theta_rem_corr 	= data['theta_rem_corr']
	index = np.tril_indices(len(rip_corr[0]))
	rip = np.vstack([rip_corr[0][index],rip_corr[1][index]]).transpose()

	if f == 'Mouse12-120809.pickle':
		
		toplot['wake'] = theta_wake_corr[0][1]
		toplot['rem'] = theta_rem_corr[0][1]
		toplot['rip'] = rip_corr[1]
		
	wake = []
	for i in range(len(theta_wake_corr)):
		np.fill_diagonal(theta_wake_corr[i][1], 1.0)
		index = np.tril_indices(len(theta_wake_corr[i][0]))
		wake.append(np.vstack([theta_wake_corr[i][0][index],theta_wake_corr[i][1][index]]).transpose())
	wake = np.vstack(wake)
	rem = []
	for i in range(len(theta_rem_corr)):
		np.fill_diagonal(theta_rem_corr[i][1], 1.0)
		index = np.tril_indices(len(theta_rem_corr[i][0]))
		rem.append(np.vstack([theta_rem_corr[i][0][index],theta_rem_corr[i][1][index]]).transpose())
	rem = np.vstack(rem)

	
	# remove nan
	rem = rem[~np.isnan(rem[:,1])]
	wake = wake[~np.isnan(wake[:,1])]
	rip = rip[~np.isnan(rip[:,1])]
	tt = 3.0
	# restrict to less than 3 second
	rem = rem[rem[:,0] <= tt,:]
	wake = wake[wake[:,0] <= tt,:]
	rip = rip[rip[:,0] <= tt,:]

	# average rip corr
	bins = np.arange(0.1, tt, 0.1)
	# bins[0] = 0.000001
	

	index_rip = np.digitize(rip[:,0], bins)
	index_wake = np.digitize(wake[:,0], bins)
	index_rem = np.digitize(rem[:,0], bins)
	mean_rip, mean_wake, mean_rem = ([], [], [])
	for i in range(1, len(bins)):
		mean_rip.append(np.mean(rip[index_rip == i,1]))
		mean_wake.append(np.mean(wake[index_wake == i,1]))
		mean_rem.append(np.mean(rem[index_rem == i,1]))

	# bad
	# 
	# 
	# plot(xt, mean_rip, 'o-', label = 'rip')
	# plot(xt, mean_wake, 'o-')
	# plot(xt, mean_rem, 'o-')
	# legend()
	# show()
	ywak.append(mean_wake)
	yrem.append(mean_rem)
	yrip.append(mean_rip)

bins[0] = 0.0
xt = bins[0:-1] + (bins[1] - bins[0])/2.

ywak = np.array(ywak)
yrem = np.array(yrem)
yrip = np.array(yrip)

meanywak = ywak[~np.isnan(ywak)[:,0]].mean(0)
meanyrem = yrem[~np.isnan(yrem)[:,0]].mean(0)
meanyrip = yrip[~np.isnan(yrip)[:,0]].mean(0)
varywak = ywak[~np.isnan(ywak)[:,0]].var(0)
varyrem = yrem[~np.isnan(yrem)[:,0]].var(0)
varyrip = yrip[~np.isnan(yrip)[:,0]].var(0)

######### 
# TO SAVE
#########
# tosave = {	'xt':		xt,
# 			'meanywak':	meanywak,
# 			'meanyrem':	meanyrem,
# 			'meanyrip':	meanyrip,
# 			'toplot'  : toplot,
# 			'varywak' : varywak,
# 			'varyrem' : varyrem,
# 			'varyrip' : varyrip,
# 		}
# cPickle.dump(tosave, open('../data/to_plot_corr_pop.pickle', 'wb'))


plot(xt, meanywak, 'o-', label = 'theta(wake)')
plot(xt, meanyrem, 'o-', label = 'theta(rem)')
plot(xt, meanyrip, 'o-', label = 'ripple')

figure()
subplot(1,4,1)
imshow(toplot['wake'])
title('wake')
subplot(1,4,2)
imshow(toplot['rem'][100:,100:])
title('REM')
subplot(1,4,3)
imshow(toplot['rip'][0:200,0:200])
title('RIPPLES')
subplot(1,4,4)
xtsym = np.array(list(xt[::-1]*-1.0)+[0.0]+list(xt))
meanywak = np.array(list(meanywak[::-1])+[1.0]+list(meanywak))
meanyrem = np.array(list(meanyrem[::-1])+[1.0]+list(meanyrem))
meanyrip = np.array(list(meanyrip[::-1])+[1.0]+list(meanyrip))
varywak  = np.array(list(varywak[::-1])+[0.0]+list(varywak))
varyrem  = np.array(list(varyrem[::-1])+[0.0]+list(varyrem))
varyrip  = np.array(list(varyrip[::-1])+[0.0]+list(varyrip))

colors = ['red', 'blue', 'green']

plot(xtsym, meanywak, '-', color = colors[0], label = 'theta(wake)')
plot(xtsym, meanyrem, '-', color = colors[1], label = 'theta(rem)')
plot(xtsym, meanyrip, '-', color = colors[2], label = 'ripple')
fill_between(xtsym, meanywak+varywak, meanywak-varywak, color = colors[0], alpha = 0.4)
fill_between(xtsym, meanyrem+varyrem, meanyrem-varyrem, color = colors[1], alpha = 0.4)
fill_between(xtsym, meanyrip+varyrip, meanyrip-varyrip, color = colors[2], alpha = 0.4)

legend()
xlabel('s')
ylabel('r')

savefig("../figures/fig_correlation_population.pdf", dpi = 900, bbox_inches = 'tight', facecolor = 'white')
os.system("evince ../figures/fig_correlation_population.pdf &")

# savefig("../../Dropbox (Peyrache Lab)/Peyrache Lab Team Folder/Code/AdrienDatasetThalamus/figures/fig_correlation_population.pdf", dpi = 900, bbox_inches = 'tight', facecolor = 'white')
# os.system("evince ../../Dropbox\ \(Peyrache\ \Lab)/Peyrache\ \Lab\ \Team\ \Folder/Code/AdrienDatasetThalamus/figures/fig_correlation_population.pdf &")



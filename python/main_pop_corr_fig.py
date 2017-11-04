

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
tt = 3.0
dt = 0.1
bins = np.arange(dt, tt+dt, dt)
mean_rip = pd.DataFrame(index = [f.split(".")[0] for f in files], columns = bins[1:] - dt)
mean_rem = pd.DataFrame(index = [f.split(".")[0] for f in files], columns = bins[1:] - dt)
mean_wak = pd.DataFrame(index = [f.split(".")[0] for f in files], columns = bins[1:] - dt)

for f in files:	
	store 			= pd.HDFStore("../data/corr_pop_no_hd/"+f)
	theta_wake_corr = store['wak_corr']
	theta_rem_corr 	= store['rem_corr']
	rip_corr 		= store['rip_corr']
	rip 			= store['allrip_corr']
	wake 			= store['allwak_corr']
	rem 			= store['allrem_corr']	
	store.close()
	# index = np.tril_indices(len(rip_corr))
	# rip = np.vstack([rip_corr[0][index],rip_corr[1][index]]).transpose()

	# if f == 'Mouse12-120809.pickle':
		
	# 	toplot['wake'] = theta_wake_corr[0][1]
	# 	toplot['rem'] = theta_rem_corr[0][1]
	# 	toplot['rip'] = rip_corr[1]

	# wake = []
	# for i in range(len(theta_wake_corr)):
	# 	np.fill_diagonal(theta_wake_corr[i][1], 1.0)
	# 	index = np.tril_indices(len(theta_wake_corr[i][0]))
	# 	wake.append(np.vstack([theta_wake_corr[i][0][index],theta_wake_corr[i][1][index]]).transpose())
	# wake = np.vstack(wake)
	# rem = []
	# for i in range(len(theta_rem_corr)):
	# 	np.fill_diagonal(theta_rem_corr[i][1], 1.0)
	# 	index = np.tril_indices(len(theta_rem_corr[i][0]))
	# 	rem.append(np.vstack([theta_rem_corr[i][0][index],theta_rem_corr[i][1][index]]).transpose())
	# rem = np.vstack(rem)

	
	# remove nan
	# rem = rem[~np.isnan(rem[:,1])]
	# wake = wake[~np.isnan(wake[:,1])]
	# rip = rip[~np.isnan(rip[:,1])]	
	# restrict to less than 3 second
	rem = rem[rem.index.values <= tt]
	wake = wake[wake.index.values <= tt]
	rip = rip[rip.index.values <= tt]
	rem = rem[rem.index.values > 0]
	wake = wake[wake.index.values > 0]
	rip = rip[rip.index.values > 0]


	
	index_rip = np.digitize(rip.index.values, bins)
	index_wake = np.digitize(wake.index.values, bins)
	index_rem = np.digitize(rem.index.values, bins)
	
	for i in range(len(bins)-1):		
		mean_rip.loc[f.split(".")[0]].iloc[i] = np.mean(rip.loc[index_rip == i])[0]
		mean_wak.loc[f.split(".")[0]].iloc[i] = np.mean(wake.loc[index_wake == i])[0]
		mean_rem.loc[f.split(".")[0]].iloc[i] = np.mean(rem.loc[index_rem == i])[0]



xt = np.array(list(mean_rip.columns))
meanywak = mean_wak.mean(0).values
meanyrem = mean_rem.mean(0).values
meanyrip = mean_rip.mean(0).values
stdywak = mean_wak.std(0).values
stdyrem = mean_rem.std(0).values
stdyrip = mean_rip.std(0).values

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


# plot(xt, meanywak, 'o-', label = 'theta(wake)')
# plot(xt, meanyrem, 'o-', label = 'theta(rem)')
# plot(xt, meanyrip, 'o-', label = 'ripple')


figure()
# subplot(1,4,1)
# imshow(toplot['wake'])
# title('wake')
# subplot(1,4,2)
# imshow(toplot['rem'][100:,100:])
# title('REM')
# subplot(1,4,3)
# imshow(toplot['rip'][0:200,0:200])
# title('RIPPLES')
subplot(1,1,1)
xtsym = np.array(list(xt[::-1]*-1.0)+[0.0]+list(xt))
meanywak = np.array(list(meanywak[::-1])+[1.0]+list(meanywak))
meanyrem = np.array(list(meanyrem[::-1])+[1.0]+list(meanyrem))
meanyrip = np.array(list(meanyrip[::-1])+[1.0]+list(meanyrip))
stdywak  = np.array(list(stdywak[::-1])+[0.0]+list(stdywak))
stdyrem  = np.array(list(stdyrem[::-1])+[0.0]+list(stdyrem))
stdyrip  = np.array(list(stdyrip[::-1])+[0.0]+list(stdyrip))

colors = ['red', 'blue', 'green']

plot(xtsym, meanywak, '-', color = colors[0], label = 'theta(wake)')
plot(xtsym, meanyrem, '-', color = colors[1], label = 'theta(rem)')
plot(xtsym, meanyrip, '-', color = colors[2], label = 'ripple')
legend()
fill_between(xtsym, meanywak+stdywak, meanywak-stdywak, color = colors[0], alpha = 0.4)
fill_between(xtsym, meanyrem+stdyrem, meanyrem-stdyrem, color = colors[1], alpha = 0.4)
fill_between(xtsym, meanyrip+stdyrip, meanyrip-stdyrip, color = colors[2], alpha = 0.4)

legend()
xlabel('s')
ylabel('r')

show()
# savefig("../figures/fig_correlation_population.pdf", dpi = 900, bbox_inches = 'tight', facecolor = 'white')
# os.system("evince ../figures/fig_correlation_population.pdf &")

# savefig("../../Dropbox (Peyrache Lab)/Peyrache Lab Team Folder/Code/AdrienDatasetThalamus/figures/fig_correlation_population.pdf", dpi = 900, bbox_inches = 'tight', facecolor = 'white')
# os.system("evince ../../Dropbox\ \(Peyrache\ \Lab)/Peyrache\ \Lab\ \Team\ \Folder/Code/AdrienDatasetThalamus/figures/fig_correlation_population.pdf &")



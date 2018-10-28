import ternary
import numpy as np
import pandas as pd
from functions import *
import sys
from functools import reduce
from sklearn.manifold import *
from pylab import *

############################################################################################################
# LOADING DATA
############################################################################################################
data_directory 	= '/mnt/DataGuillaume/MergedData/'
datasets 		= np.loadtxt(data_directory+'datasets_ThalHpc.list', delimiter = '\n', dtype = str, comments = '#')
burstiness 				= pd.HDFStore("/mnt/DataGuillaume/MergedData/BURSTINESS.h5")['w']
theta_mod, theta_ses 	= loadThetaMod('/mnt/DataGuillaume/MergedData/THETA_THAL_mod.pickle', datasets, return_index=True)
theta 					= pd.DataFrame(	index = theta_ses['rem'], 
									columns = ['phase', 'pvalue', 'kappa'],
									data = theta_mod['rem'])
rippower 				= pd.read_hdf("../figures/figures_articles/figure2/power_ripples_2.h5")
mappings = pd.read_hdf("/mnt/DataGuillaume/MergedData/MAPPING_NUCLEUS.h5")

############################################################################################################
# WHICH NEURONS
############################################################################################################
firing_rate = pd.read_hdf("/mnt/DataGuillaume/MergedData/FIRING_RATE_ALL.h5")
fr_index = firing_rate.index.values[((firing_rate >= 1.0).sum(1) == 3).values]
neurons = reduce(np.intersect1d, (burstiness.index.values, theta.index.values, rippower.index.values, fr_index))

nucleus = ['AD', 'AM', 'AVd', 'AVv', 'VA', 'LDvl']

neurons = np.intersect1d(neurons, mappings.index[mappings['nucleus'].isin(nucleus)])

############################################################################################################
# STACKING DIMENSIONS
############################################################################################################
data = pd.DataFrame(index = neurons)
data['burst'] = burstiness.loc[neurons, 'sws']
data['theta'] = theta.loc[neurons, 'kappa']
data['swr'] = rippower.loc[neurons]
data['costheta'] = np.cos(theta.loc[neurons,'phase'])
data['sintheta'] = np.sin(theta.loc[neurons,'phase'])



# mds = MDS(n_components = 2, metric = True).fit_transform(data.values)
# scatter(mds[:,0], mds[:,1])
# show()

tsne = TSNE(n_components = 2).fit_transform(data.values)


names = mappings.loc[neurons, 'nucleus'].copy()
colors = names.replace(to_replace=np.unique(names), value=np.arange(len(np.unique(names))))

scatter(tsne[:,0], tsne[:,1], 20, c = colors)
scatter(tsne[mappings.loc[neurons,'hd']==1][:,0], tsne[mappings.loc[neurons,'hd']==1][:,1], 2, color = 'white')


# for i, txt in enumerate(names):
# 	text(tsne[i,0], tsne[i,1], txt)

# scatter(tsne[:,0], tsne[:,1])


show()


import ternary

# Scatter Plot
scale = 100
figure, tax = ternary.figure(scale=scale)
figure.set_size_inches(10, 10)
tax.set_title("Scatter Plot", fontsize=20)
tax.boundary(linewidth=2.0)
tax.gridlines(multiple=10, color="blue")
# Plot a few different styles with a legend
tmp = [tuple(i) for i in data.values]
tax.scatter(tmp, marker='.', color='red')
tax.legend()
tax.ticks(axis='lbr', linewidth=1, multiple=5)
tax.clear_matplotlib_ticks()
tax.get_axes().axis('off')
tax.show()
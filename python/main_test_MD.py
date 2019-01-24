

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
import matplotlib.cm as cm
import os
from scipy.ndimage import gaussian_filter	





data_directory 	= '/mnt/DataGuillaume/MergedData/'
datasets 		= np.loadtxt(data_directory+'datasets_ThalHpc.list', delimiter = '\n', dtype = str, comments = '#')
swr_mod, swr_ses 		= loadSWRMod('/mnt/DataGuillaume/MergedData/SWR_THAL_corr.pickle', datasets, return_index=True)
nbins 					= 400
binsize					= 5
times 					= np.arange(0, binsize*(nbins+1), binsize) - (nbins*binsize)/2
swr_mod 					= pd.DataFrame(	columns = swr_ses, 
										index = times,
										data = gaussFilt(swr_mod, (20,)).transpose())
swr_mod = swr_mod.drop(swr_mod.columns[swr_mod.isnull().any()].values, axis = 1)
swr_mod = swr_mod.loc[-500:500]

mappings = pd.read_hdf("/mnt/DataGuillaume/MergedData/MAPPING_NUCLEUS.h5")

nbins 					= 200
binsize					= 5
times 					= np.arange(0, binsize*(nbins+1), binsize) - (nbins*binsize)/2
times2 					= swr_mod.index.values

nucleus = ['AD', 'AM', 'AVd', 'AVv', 'IAD', 'MD', 'PV', 'sm']

swr_nuc = pd.DataFrame(index = swr_mod.index, columns = pd.MultiIndex.from_product([['Mouse12', 'Mouse17', 'Mouse20', 'Mouse32'],nucleus,['mean', 'sem']]))

neurons = np.intersect1d(swr_mod.columns.values, mappings.index.values)

index = mappings.groupby(['nucleus']).groups

swr_MD = swr_mod[index['MD']]
swr_AD = swr_mod[index['AD']]

figure()
subplot(121)
plot(swr_MD)
plot(swr_MD.mean(1), color = 'black', linewidth = 5)
title("MD")
subplot(122)
plot(swr_AD)
plot(swr_AD.mean(1), color = 'black', linewidth = 5)
title("AD")
show()


data = cPickle.load(open('../figures/figures_articles_v2/figure3/dict_fig3_article.pickle', 'rb'))
allzth 			= 	data['swr_modth'	]
eigen 			= 	data['eigen'		]		
times 			= 	data['times' 		]
allthetamodth 	= 	data['theta_modth'	]		
phi 			= 	data['phi' 			]		
zpca 			= 	data['zpca'			]		
phi2			= 	data['phi2' 		]	 					
jX				= 	data['rX'			]
jscore			= 	data['jscore'		]
force 			= 	data['force'		] # theta modulation
variance 		= 	data['variance'		] # ripple modulation


# sort allzth 
index = allzth[0].sort_values().index.values
index = index[::-1]
allzthsorted = allzth.loc[index]
phi = phi.loc[index]
phi2 = phi2.loc[index]


rippower 				= pd.read_hdf("../figures/figures_articles_v2/figure2/power_ripples_2.h5")

idx_MD = swr_MD.columns.values
idx_AD = swr_AD.columns.values

figure()
scatter(rippower.loc[idx_MD], swr_MD.loc[0, idx_MD], label = 'MD')
scatter(rippower.loc[idx_AD], swr_AD.loc[0, idx_AD], label = 'AD')
legend()
xlabel("SWR power")
ylabel("z (t=0)")


figure()
hist(swr_AD.loc[-50], label = 'AD', bins = 20, alpha = 0.5)
hist(swr_MD.loc[-50], label = 'MD', bins = 20, alpha = 0.5)
legend()
xlabel("Z(t=0)")
show()
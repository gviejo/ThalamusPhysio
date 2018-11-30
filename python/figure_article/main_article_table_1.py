

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

mappings = pd.read_hdf("/mnt/DataGuillaume/MergedData/MAPPING_NUCLEUS.h5")

allnucleus = np.unique(mappings['nucleus'])

summary = pd.DataFrame(index = ['Mouse 1', 'Mouse 2', 'Mouse 3', 'Mouse 4'], columns = allnucleus, data = 0)

mouse = ['Mouse17', 'Mouse12', 'Mouse20', 'Mouse32']

for i in range(4):	
	m = mouse[i]
	groups = mappings[mappings.index.str.contains(m)].groupby('nucleus').groups
	for n in groups:
		summary.loc['Mouse '+str(i+1),n] = len(groups[n])


summary = summary.append(pd.DataFrame(summary.sum(0), columns = ['All']).T)
summary['All'] = summary.sum(1)
summary = summary.sort_values('All',1)
hd = pd.Series(index = summary.index, data= [int(mappings[mappings.index.str.contains(m)]['hd'].sum()) for m in mouse]+[int(mappings['hd'].sum())])
summary['HD'] = hd

with open("../../figures/figures_articles/table1.tex", 'w') as tf:
	tf.writelines(r"\documentclass{article}")
	tf.writelines(r"\usepackage[paperheight=1.6in,paperwidth=7.5in, margin = 0.0in]{geometry}")
	tf.writelines(r"\usepackage{booktabs}")
	tf.writelines(r"\begin{document}")				
	tf.write(summary.to_latex())
	tf.writelines(r"\end{document}")		

# os.system("pdflatex ")








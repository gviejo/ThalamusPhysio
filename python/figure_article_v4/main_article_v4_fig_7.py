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


###############################################################################################################
# TO LOAD
###############################################################################################################
#LOADING DATA 
path = '../../figures/figures_articles_v4/figure7/'
speedinfo = pd.read_hdf(path + "speedinfo_rats_box.h5")
autocorrbox = pd.read_hdf(path + "autocorrs_rats_box.h5", 'w')
databox = pd.read_hdf(path + "data_rats_box.h5", 'w')
hdinfo = pd.read_hdf(path + "hdinfo.h5")
tcurves = pd.read_hdf(path + "tuning_curves.h5")
lambdaa = pd.read_hdf(path + "lambda_rats_box.h5")
longauto = pd.read_hdf(path + "autocorrs_rats_LONG_box.h5", 'w')

ages = np.arange(12, 21, 1)


# WHICH ARE THE TRUE HD NEURONS
cond = np.logical_and(hdinfo['pval']<0.05, hdinfo['z']>10)
hdneurons = hdinfo[cond].index.values
# alln = autocorrbox.columns.values
# tmp1 = np.unique([i.split("_")[0]+"_"+i.split("_")[2] for i in hdneurons])
# tokeep = np.array([n for n in alln if n.split("_")[0]+"_"+n.split("_")[2] in tmp1])
tokeep = hdneurons

# THRESHOLD FIRING RATE
tokeep = np.intersect1d(databox[databox['frate']>1.0].index.values, tokeep)

# AGE IN AGES
tokeep = databox.loc[tokeep][np.logical_and(databox.loc[tokeep,'age']>=ages.min(), databox.loc[tokeep,'age']<=ages.max())].index.values

groups = databox.loc[tokeep].groupby('age').groups

# PCA +/- 30 ms
tmp = autocorrbox.loc[0.5:,tokeep]
tmp = tmp.rolling(window=100,win_type='gaussian',center=True,min_periods=1).mean(std=2.0)
tmp2 = tmp[::-1]
tmp2.index *= -1.0
smauto = pd.concat((tmp2, pd.DataFrame(index=[0], data=0, columns = tmp.columns), tmp), 0)
x = smauto.loc[-30:30]

pc = PCA(n_components = 2).fit_transform(x.values.T)
pc = pd.DataFrame(index = tokeep, data = pc)

variance = pd.Series(index = ages, data = [pc.loc[groups[a]].var().values.sum() for a in ages])

# TESTING FACTORS
data = pd.DataFrame(index = tokeep, columns = ['burst', 'age', 'tau', 'z'])
data['burst'] = databox.loc[tokeep,'burst']
data['age'] = databox.loc[tokeep, 'age']
data['tau'] = lambdaa.loc[tokeep, 'b']
data['z'] = hdinfo.loc[tokeep, 'z']
data['var'] = speedinfo.loc[tokeep, 'var']

import statsmodels.api as sm
X = data[['age','tau','z', 'var']].astype('float')
Y = data['burst'].astype('float')
model = sm.OLS(Y,X).fit()
model.summary()
# figure()
# for i, a in enumerate(ages):
# 	subplot(3,3,i+1)
# 	plot(np.log(lambdaa.loc[groups[a],'b'].astype('float')), np.log(databox.loc[groups[a], 'burst'].astype('float')), 'o', color = colorss[a])
# 	# xlim(0, 5)
# 	title(a)
# 	xlabel("Tau(s)")
# 	ylabel("Burstiness")
# show()

# figure(figsize = (5,20))
# for i, n in zip(range(0,len(n14)*2,2),n14):
# 	subplot(len(n14),2,i+1,projection='polar')
# 	plot(tcurves[n])
# 	xticks([])
# 	subplot(len(n14),2,i+2)
# 	plot(autocorrbox[n])
# 	title(n)



###############################################################################################################
# PLOT
###############################################################################################################
def figsize(scale):
	fig_width_pt = 483.69687                         # Get this from LaTeX using \the\textwidth
	inches_per_pt = 1.0/72.27                       # Convert pt to inch
	golden_mean = (np.sqrt(5.0)-1.0)/2.0            # Aesthetic ratio (you could change this)
	fig_width = fig_width_pt*inches_per_pt*scale    # width in inches
	fig_height = fig_width*golden_mean*1        # height in inches
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


import matplotlib as mpl
from mpl_toolkits.axes_grid1 import make_axes_locatable

# mpl.use("pdf")
pdf_with_latex = {                      # setup matplotlib to use latex for output
	"pgf.texsystem": "pdflatex",        # change this if using xetex or lautex
	# "text.usetex": True,                # use LaTeX to write all text
	# "font.family": "serif",
	"font.serif": [],                   # blank entries should cause plots to inherit fonts from the document
	"font.sans-serif": [],
	"font.monospace": [],
	"axes.labelsize": 8,               # LaTeX default is 10pt font.
	"font.size": 7,
	"legend.fontsize": 7,               # Make the legend/label fonts a little smaller
	"xtick.labelsize": 7,
	"ytick.labelsize": 7,
	"pgf.preamble": [
		r"\usepackage[utf8x]{inputenc}",    # use utf8 fonts becasue your computer can handle it :)
		r"\usepackage[T1]{fontenc}",        # plots will be generated using this preamble
		],
	"lines.markeredgewidth" : 0.2,
	"axes.linewidth"        : 0.8,
	"ytick.major.size"      : 1.5,
	"xtick.major.size"      : 1.5
	}      
mpl.rcParams.update(pdf_with_latex)
import matplotlib.gridspec as gridspec
from matplotlib.pyplot import *
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import matplotlib.cm as cmx
import matplotlib.colors as colors

cm = get_cmap('viridis')
colorss = dict(zip(ages, [cm(((a - ages.min())/(ages.max()-ages.min()))*0.9) for a in ages]))


fig = figure(figsize = figsize(1.0))
gs = gridspec.GridSpec(4,1, wspace = 0.3, hspace = 0.7, height_ratios = [1,0.4,-0.3,1])#, width_ratios = [1,0.8,1], height_ratios = [1,0.9,1])

#########################################################################
# A. EXEMPLES AUTOCORRS
#########################################################################
gsA = gridspec.GridSpecFromSubplotSpec(1,3,subplot_spec=gs[0,0])

tmp = autocorrbox.loc[0.5:,tokeep]
tmp = tmp.rolling(window=100,win_type='gaussian',center=True,min_periods=1).mean(std=1.0)
tmp2 = tmp[::-1]
tmp2.index *= -1.0
smauto = pd.concat((tmp2, pd.DataFrame(index=[0], data=0, columns = tmp.columns), tmp), 0)
x = smauto.loc[-30:30]


for i, a in zip(range(3), [14,17,20]):
	ax = subplot(gsA[0,i])
	simpleaxis(ax)
	plot(x[groups[a]], color = colorss[a], alpha = 0.8, linewidth = 1)
	title("P"+str(a))
	xlabel("Time lag (s)")
	if i == 0:
		ylabel("Autocorrelogram")
#########################################################################
# B. PCA AGES
#########################################################################
gsB = gridspec.GridSpecFromSubplotSpec(1,len(ages),subplot_spec=gs[1,0], wspace = 0)


for i,a in enumerate(ages):
	ax = subplot(gsB[0,i])
	noaxis(ax)
	plot(pc.loc[groups[a], 0], pc.loc[groups[a], 1], 'o', color = colorss[a], markersize = 3, alpha = 0.5)
	xlabel("P"+str(a))
	ylim(-30,30)
	xlim(-35,80)
	if i == 0:
		ylabel("PCA")


#########################################################################
# C. VARIOUS
#########################################################################
gsC = gridspec.GridSpecFromSubplotSpec(1,3,subplot_spec=gs[3,0], wspace = 0.4)

# VARIANCE PC
ax = subplot(gsC[0,0])
simpleaxis(ax)
for a in ages:
	bar([a], variance.loc[a], color = colorss[a])
xlabel("Age")
ylabel("Variance")

# BURSTINESS
ax = subplot(gsC[0,1])
simpleaxis(ax)
for a in ages:
	plot(databox.loc[groups[a], 'age'], np.log(databox.loc[groups[a], 'burst']), 'o', color = colorss[a], markersize = 3)
xlabel("Age")
ylabel("Burstiness")

# cax = inset_axes(ax, "25%", "25%",
#                    bbox_to_anchor=(0.7, 0.8, 1, 1),
#                    bbox_transform=ax.transAxes, 
#                    loc = 'lower left')
# plot(databox.loc[tokeep].groupby('age').var()['burst'])

# VARIANCE ANGULAR SPEED
ax = subplot(gsC[0,2])
simpleaxis(ax)

tmp = hdinfo.loc[tokeep, ['age', 'z', 'pval']]

meanz = [tmp.loc[groups[a],'z'].astype('float').mean() for a in ages]
varz = [tmp.loc[groups[a],'z'].astype('float').sem() for a in ages]

# for a in ages:
# 	plot(hdinfo.loc[groups[a], 'age'], hdinfo.loc[groups[a], 'z'], 'o', color = colorss[a], markersize = 3)
for i, a in enumerate(ages):
	bar([a], meanz[i], yerr = varz[i], color = colorss[a])


xlabel("Age")
ylabel("Rayleigh score")

# cax = inset_axes(ax, "30%", "30%",
#                    bbox_to_anchor=(0.65, 0.8, 1, 1),
#                    bbox_transform=ax.transAxes, 
#                    loc = 'lower left')

# m = speedinfo.loc[tokeep].groupby('age').mean()['var']
# v = speedinfo.loc[tokeep].groupby('age').std()['var']
# for a in ages:
# 	bar([a], m.loc[a], color = colorss[a])


subplots_adjust(top = 0.95, bottom = 0.1, right = 0.99, left = 0.07, hspace = 0.5)

savefig("../../figures/figures_articles_v4/figart_7.pdf", dpi = 900, facecolor = 'white')
os.system("evince ../../figures/figures_articles_v4/figart_7.pdf &")


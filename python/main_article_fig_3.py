

import numpy as np
import pandas as pd
# from matplotlib.pyplot import plot,show,draw
import scipy.io
from functions import *
from pylab import *
from sklearn.decomposition import PCA
import _pickle as cPickle
import matplotlib.cm as cm
import os
import neuroseries as nts


def get_xmin(ep, minutes):
	duree = (ep['end'] - ep['start'])/1000/1000/60
	starts = []
	ends = []
	count = 0.0
	for i in range(len(ep)):
		if count < minutes:
			starts.append(ep['start'].iloc[i])
			ends.append(ep['end'].iloc[i])
			count += duree[i]

	return nts.IntervalSet(starts, ends)

###############################################################################################################
# TO LOAD
###############################################################################################################
exemple = pd.HDFStore("../figures/figures_articles/figure3/Mouse12-120807_pop_pca.h5")
eigen = exemple['eigen']
posscore = exemple['posscore']
prescore = exemple['prescore']
prescore = nts.Tsd(prescore.index.values, prescore.values.flatten())
posscore = nts.Tsd(posscore.index.values, posscore.values.flatten())
exemple.close()

data_directory = '/mnt/DataGuillaume/MergedData/'
session = 'Mouse12/Mouse12-120807'
generalinfo 	= scipy.io.loadmat(data_directory+session+'/Analysis/GeneralInfo.mat')
shankStructure 	= loadShankStructure(generalinfo)	
# spikes,shank	= loadSpikeData(data_directory+session+'/Analysis/SpikeData.mat', shankStructure['thalamus'])		
wake_ep 		= loadEpoch(data_directory+session, 'wake')
sleep_ep 		= loadEpoch(data_directory+session, 'sleep')
sws_ep 			= loadEpoch(data_directory+session, 'sws')
rem_ep 			= loadEpoch(data_directory+session, 'rem')
sleep_ep 		= sleep_ep.merge_close_intervals(threshold=1.e3)		
# sws_ep 			= sleep_ep.intersect(sws_ep)	
# rem_ep 			= sleep_ep.intersect(rem_ep)
pre_ep 			= nts.IntervalSet(sleep_ep['start'][0], sleep_ep['end'][0])
post_ep 		= nts.IntervalSet(sleep_ep['start'][1], sleep_ep['end'][1])
pre_sws_ep 		= sws_ep.intersect(pre_ep)
pos_sws_ep 		= sws_ep.intersect(post_ep)
pre_sws_ep 		= get_xmin(pre_sws_ep.iloc[::-1], 30)
pos_sws_ep		= get_xmin(pos_sws_ep, 30)

store = pd.HDFStore("../figures/figures_articles/figure3/pop_phase_shift.h5")
corr = store['corr']
phi = store['phi']
store.close()

store = pd.HDFStore("../figures/figures_articles/figure3/pca_analysis_3.h5")
ripscore = {}
remscore = {}
# for i, j in zip(range(3), ['hd', 'nohd_mod', 'nohd_nomod']):
for i,j in zip(range(3),('nohd_mod', 'hd', 'nohd_nomod')):

	ripscore[i] = {'pre':store[str(j)+'pre_rip'],
					'pos':store[str(j)+'pos_rip']}
	# remscore[i] = {'pre':store[str(i)+'pre_rem'],
	# 				'pos':store[str(i)+'pos_rem']}

store.close()


###############################################################################################################
# PLOT
###############################################################################################################
def figsize(scale):
	fig_width_pt = 483.69687                         # Get this from LaTeX using \the\textwidth
	inches_per_pt = 1.0/72.27                       # Convert pt to inch
	golden_mean = (np.sqrt(5.0)-1.0)/2.0            # Aesthetic ratio (you could change this)
	fig_width = fig_width_pt*inches_per_pt*scale    # width in inches
	fig_height = fig_width*golden_mean*0.6            # height in inches
	fig_size = [fig_width*0.75,fig_height]
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
import matplotlib.patches as patches


mpl.use("pdf")
pdf_with_latex = {                      # setup matplotlib to use latex for output
	"pgf.texsystem": "pdflatex",        # change this if using xetex or lautex
	"text.usetex": True,                # use LaTeX to write all text
	"font.family": "serif",
	"font.serif": [],                   # blank entries should cause plots to inherit fonts from the document
	"font.sans-serif": [],
	"font.monospace": [],
	"axes.labelsize": 8,               # LaTeX default is 10pt font.
	"font.size": 7,
	"legend.fontsize": 7,               # Make the legend/label fonts a little smaller
	"xtick.labelsize": 4,
	"ytick.labelsize": 4,
	"pgf.preamble": [
		r"\usepackage[utf8x]{inputenc}",    # use utf8 fonts becasue your computer can handle it :)
		r"\usepackage[T1]{fontenc}",        # plots will be generated using this preamble
		],
	"lines.markeredgewidth" : 0.2,
	"axes.linewidth"        : 0.5,
	"ytick.major.size"      : 1.0,
	"xtick.major.size"      : 1.0
	}    
mpl.rcParams.update(pdf_with_latex)
import matplotlib.gridspec as gridspec
from matplotlib.pyplot import *
from mpl_toolkits.axes_grid.inset_locator import inset_axes


fig = figure(figsize = figsize(1))
gs1 = gridspec.GridSpec(1,3)
gs1.update(left = 0.05, right = 0.95, wspace = 0.3, top = 0.93, bottom = 0.65)
lw = 0.5

colors = ['#044293', '#dc3926']

##############################################################
# PRE SCORE
##############################################################
ax1 = subplot(gs1[0,0])
simpleaxis(ax1)
ax1.plot(prescore.restrict(pre_sws_ep), linewidth = lw, color = colors[0])
title("Pre Sleep")
ylim(-40, 60)
# rem_ep_pre = rem_ep.intersect(pre_xmin_ep)
# sws_ep_pre = sws_ep.intersect(pre_xmin_ep)
# [plot([rem_ep_pre['start'][i], rem_ep_pre['end'][i]], np.zeros(2)-50.0, '-', color = 'orange') for i in range(len(rem_ep_pre))]
# [plot([sws_ep_pre['start'][i], sws_ep_pre['end'][i]], np.zeros(2)-50.0, '-', color = 'red') for i in range(len(sws_ep_pre))]	


##############################################################
# EIGENVECTORS
##############################################################
inter = gridspec.GridSpecFromSubplotSpec(6, 1, subplot_spec=gs1[0,1])

ax2 = Subplot(fig, inter[0,0])
fig.add_subplot(ax2)
title('Wake')
xlim(0,1)
ylim(0,1)
noaxis(ax2)

for i in range(len(eigen))[0:5]:
	# ax = inter[i,0]
	ax = Subplot(fig, inter[i+1,0])
	fig.add_subplot(ax)
	noaxis(ax)
	axhline(0, linewidth = 0.4, color = 'grey')
	for j in eigen.loc[i].index.values:
		ax.plot([j,j], [0, eigen.loc[i,j]], '-', linewidth = lw, color = 'black', markersize = 1)
		ax.plot([j], [eigen.loc[i,j]], 'o', linewidth = lw, color = 'black', markersize = 1)
	ylabel("PC "+str(i+1), rotation = 'horizontal', fontsize = 4)

xlabel("Neurons", fontsize = 5)
##############################################################
# POST SCORE
##############################################################
ax3 = subplot(gs1[0,2])
simpleaxis(ax3)
ax3.plot(posscore.restrict(pos_sws_ep), linewidth = lw, color = colors[1])
title("Post Sleep")
ylim(-40, 60)
# rem_ep_pos = rem_ep.intersect(post_xmin_ep)
# sws_ep_pos = sws_ep.intersect(post_sws_ep)
# [plot([rem_ep_pos['start'][i], rem_ep_pos['end'][i]], np.zeros(2)-50.0, '-', color = 'orange') for i in range(len(rem_ep_pos))]
# [plot([sws_ep_pos['start'][i], sws_ep_pos['end'][i]], np.zeros(2)-50.0, '-', color = 'red') for i in range(len(sws_ep_pos))]	


##############################################################
# ARROWS
##############################################################
ax1tr = ax1.transData
ax2tr = ax2.transData
ax3tr = ax3.transData
figtr = fig.transFigure.inverted()
ptB = figtr.transform(ax2tr.transform((0.2,1)))
ptE = figtr.transform(ax1tr.transform((5e9,50)))
style="simple,head_width=2,head_length=3"
kw = dict(arrowstyle=style, color="k")
arrow = matplotlib.patches.FancyArrowPatch(
    ptB, ptE, transform=fig.transFigure,  # Place arrow in figure coord system
    fc = "None", connectionstyle="arc3,rad=0.4", alpha = 0.3,
    mutation_scale = 3., **kw)
fig.patches.append(arrow)

ptB = figtr.transform(ax2tr.transform((0.8,1)))
ptE = figtr.transform(ax3tr.transform((0.93e10,50)))
style="<->,head_width=2,head_length=3"
arrow = matplotlib.patches.FancyArrowPatch(
    ptB, ptE, transform=fig.transFigure,  # Place arrow in figure coord system
    fc = "None", connectionstyle="arc3,rad=-0.4", alpha = 0.3,
    mutation_scale = 3., **kw)
fig.patches.append(arrow)

ax2.text(0.1, 0.5, 'PCA(population activity)', fontsize = 4)
ax1.text(3.5e9, 45, 'Reactivation Score', fontsize = 4)




##############################################################
# NOHD SWR MOD
##############################################################
gs2 = gridspec.GridSpec(1,3)
gs2.update(left = 0.07, right = 0.8, wspace = 0.24, bottom = 0.1, top = 0.5)

ax = subplot(gs2[0,0])
simpleaxis(ax)
times = ripscore[0]['pre'].index.values
labels = ['Pre-sleep', 'Post-sleep']


for k,i in zip(['pre', 'pos'],range(2)):
	for j in ripscore[0][k].columns:
		ripscore[0][k][j] = gaussFilt(ripscore[0][k][j], (1,))

	plot(ripscore[0][k].mean(1).loc[-500:500], linewidth = 1, color = colors[i], label = labels[i])
	sem = ripscore[0][k].sem(1).loc[-500:500]
	fill_between(sem.index.values, ripscore[0][k].mean(1).loc[-500:500] - sem, ripscore[0][k].mean(1).loc[-500:500] + sem, facecolor = colors[i], alpha = 0.2, linewidth = lw)


axvline(0, linewidth =1, color = 'grey', alpha = 0.5)
ylabel("SPWR score", fontsize = 5)
# legend(loc = 'upper left', edgecolor = None, facecolor = None, frameon = False)
# text(1800,0.90, '$\mathbf{Non\ HD\ Neurons}$')
xlabel("Time from SPWR (s)", fontsize = 5)
title("Theta modulated \n Non HD Neurons", fontsize = 5, y = 0.95)


##############################################################
# HD SWR
##############################################################
ax = subplot(gs2[0,1])
simpleaxis(ax)
for k,i in zip(['pre', 'pos'],range(2)):
	for j in ripscore[1][k].columns:
		ripscore[1][k][j] = gaussFilt(ripscore[1][k][j], (1,))

	plot(ripscore[1][k].mean(1).loc[-500:500], linewidth = 1, color = colors[i], label = labels[i])
	sem = ripscore[1][k].sem(1).loc[-500:500]
	fill_between(sem.index.values, ripscore[1][k].mean(1).loc[-500:500] - sem, ripscore[1][k].mean(1).loc[-500:500] + sem, facecolor = colors[i], alpha = 0.2, linewidth = lw)


# legend(loc = 'upper left', edgecolor = None, facecolor = None, frameon = False)
axvline(0, linewidth =1, color = 'grey', alpha = 0.5)
xlabel("Time from SPWR (s)", fontsize = 5)
# text(2000,3.4, '$\mathbf{HD\ Neurons}$')
title("HD Neurons", fontsize = 5, y = 0.95)


##############################################################
# HD SWR NO MOD
##############################################################
ax = subplot(gs2[0,2])
simpleaxis(ax)
for k,i in zip(['pre', 'pos'],range(2)):
	for j in ripscore[2][k].columns:
		ripscore[2][k][j] = gaussFilt(ripscore[2][k][j], (1,))

	plot(ripscore[2][k].mean(1).loc[-500:500], linewidth = 1, color = colors[i], label = labels[i])
	sem = ripscore[2][k].sem(1).loc[-500:500]
	fill_between(sem.index, ripscore[2][k].mean(1).loc[-500:500] - sem, ripscore[2][k].mean(1).loc[-500:500] + sem, facecolor = colors[i], alpha = 0.2, linewidth = lw)


# legend(loc = 'upper left', edgecolor = None, facecolor = None, frameon = False)
axvline(0, linewidth =1, color = 'grey', alpha = 0.5)
xlabel("Time from SPWR (s)", fontsize = 5)
# text(2000,3.4, '$\mathbf{HD\ Neurons}$')
title("Non Theta modulated \n Non HD Neurons", fontsize = 5, y = 0.95)


# ##############################################################
# # COVARIANCE
# ##############################################################
gs3 = gridspec.GridSpec(2,1)
gs3.update(left = 0.9, right = 0.99, hspace = 0.5,  bottom = 0.06, top = 0.5)
# ax = subplot(gs3[0,0])
# simpleaxis(ax)
# # scatter(corr['pre'], corr['pos'], s = 1)
# xlabel('$R_{pre}$')
# ylabel('$R_{post}$')



# ##############################################################
# # PHASE DIFFERENCE
# ##############################################################
ax = subplot(gs3[0,0])
simpleaxis(ax)
x = phi['phi']-phi['phipre']
y = phi['phi']-phi['phipos']
x[(x < -np.pi)] += 2*np.pi
y[(y < -np.pi)] += 2*np.pi
x[(x > np.pi)] -= 2*np.pi
y[(y > np.pi)] -= 2*np.pi
# scatter(x, y, s = 1)
scatter(phi['phipre'], phi['phipos'], s = 1)
xlabel('$\phi_{pre}$')
ylabel('$\phi_{post}$')


##############################################################
# TIME MAX
##############################################################
# ax = subplot(gs3[2,0])
# simpleaxis(ax)




savefig("../figures/figures_articles/figart_3.pdf", dpi = 900, facecolor = 'white')
os.system("evince ../figures/figures_articles/figart_3.pdf &")


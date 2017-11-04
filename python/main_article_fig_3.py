

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

	# plot([wake_ep['start'][0], wake_ep['end'][0]], np.zeros(2), '-', color = 'blue', label = 'wake')
	# [plot([wake_ep['start'][i], wake_ep['end'][i]], np.zeros(2), '-', color = 'blue') for i in range(len(wake_ep))]
	# plot([sleep_ep['start'][0], sleep_ep['end'][0]], np.zeros(2), '-', color = 'green', label = 'sleep')
	# [plot([sleep_ep['start'][i], sleep_ep['end'][i]], np.zeros(2), '-', color = 'green') for i in range(len(sleep_ep))]	
	# plot([rem_ep['start'][0], rem_ep['end'][0]],  np.zeros(2)+0.1, '-', color = 'orange', label = 'rem')
	# [plot([rem_ep['start'][i], rem_ep['end'][i]], np.zeros(2)+0.1, '-', color = 'orange') for i in range(len(rem_ep))]
	# plot([sws_ep['start'][0], sws_ep['end'][0]],  np.zeros(2)+0.1, '-', color = 'red', label = 'sws')
	# [plot([sws_ep['start'][i], sws_ep['end'][i]], np.zeros(2)+0.1, '-', color = 'red') for i in range(len(sws_ep))]	
	# legend()

###############################################################################################################
# TO LOAD
###############################################################################################################
exemple = pd.HDFStore("../figures/figures_articles/figure3/Mouse12-120807_pop_pca.h5")
eigen = exemple['eigen']
posscore = exemple['posscore']
prescore = exemple['prescore']
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
pre_xmin_ep = pre_ep.copy()
pre_xmin_ep['start'] = pre_xmin_ep['end'][0] - 60*60*1000*1000
post_xmin_ep = post_ep.copy()
post_xmin_ep['end'] = post_xmin_ep['start'][0] + 60*60*1000*1000
pre_xmin_ep = pre_ep.copy()
pre_xmin_ep['start'] = pre_xmin_ep['end'][0] - 60*60*1000*1000
post_xmin_ep = post_ep.copy()
post_xmin_ep['end'] = post_xmin_ep['start'][0] + 60*60*1000*1000


store = pd.HDFStore("../figures/figures_articles/figure3/pop_phase_shift.h5")
corr = store['corr']
phi = store['phi']
store.close()

store = pd.HDFStore("../figures/figures_articles/figure3/pca_analysis.h5")
ripscore = {}
remscore = {}
for i in range(2):	
	ripscore[i] = {'pre':store[str(i)+'pre_rip'],
					'pos':store[str(i)+'pos_rip']}
	remscore[i] = {'pre':store[str(i)+'pre_rem'],
					'pos':store[str(i)+'pos_rem']}

store.close()


###############################################################################################################
# PLOT
###############################################################################################################
def figsize(scale):
	fig_width_pt = 483.69687                         # Get this from LaTeX using \the\textwidth
	inches_per_pt = 1.0/72.27                       # Convert pt to inch
	golden_mean = (np.sqrt(5.0)-1.0)/2.0            # Aesthetic ratio (you could change this)
	fig_width = fig_width_pt*inches_per_pt*scale    # width in inches
	fig_height = fig_width*golden_mean*1            # height in inches
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
gs1.update(left = 0.05, right = 0.7, wspace = 0.3, top = 0.95, bottom = 0.75)
lw = 0.5

colors = ['#044293', '#dc3926']

##############################################################
# PRE SCORE
##############################################################
ax1 = subplot(gs1[0,0])
simpleaxis(ax1)
ax1.plot(prescore, linewidth = lw, color = colors[0])
title("Pre Sleep")
ylim(-40, 60)
# rem_ep_pre = rem_ep.intersect(pre_xmin_ep)
# sws_ep_pre = sws_ep.intersect(pre_xmin_ep)
# [plot([rem_ep_pre['start'][i], rem_ep_pre['end'][i]], np.zeros(2)-50.0, '-', color = 'orange') for i in range(len(rem_ep_pre))]
# [plot([sws_ep_pre['start'][i], sws_ep_pre['end'][i]], np.zeros(2)-50.0, '-', color = 'red') for i in range(len(sws_ep_pre))]	


##############################################################
# EIGENVECTORS
##############################################################
inter = gridspec.GridSpecFromSubplotSpec(len(eigen)+1, 1, subplot_spec=gs1[0,1])

ax2 = Subplot(fig, inter[0,0])
fig.add_subplot(ax2)
title('Wake')
xlim(0,1)
ylim(0,1)
noaxis(ax2)

for i in range(len(eigen)):
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
ax3.plot(posscore, linewidth = lw, color = colors[1])
title("Post Sleep")
ylim(-40, 60)
rem_ep_pos = rem_ep.intersect(post_xmin_ep)
sws_ep_pos = sws_ep.intersect(post_xmin_ep)
[plot([rem_ep_pos['start'][i], rem_ep_pos['end'][i]], np.zeros(2)-50.0, '-', color = 'orange') for i in range(len(rem_ep_pos))]
[plot([sws_ep_pos['start'][i], sws_ep_pos['end'][i]], np.zeros(2)-50.0, '-', color = 'red') for i in range(len(sws_ep_pos))]	


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
ptE = figtr.transform(ax3tr.transform((0.86e10,50)))
style="<->,head_width=2,head_length=3"
arrow = matplotlib.patches.FancyArrowPatch(
    ptB, ptE, transform=fig.transFigure,  # Place arrow in figure coord system
    fc = "None", connectionstyle="arc3,rad=-0.4", alpha = 0.3,
    mutation_scale = 3., **kw)
fig.patches.append(arrow)

ax2.text(0.1, 0.5, 'PCA(population activity)', fontsize = 4)
ax1.text(3.5e9, 45, 'Reactivation Score', fontsize = 4)
##############################################################
# NOHD SWR
##############################################################
gs2 = gridspec.GridSpec(2,3)
gs2.update(left = 0.05, right = 0.7, wspace = 0.5, bottom = 0.06, top = 0.65)
ax = subplot(gs2[0,0:2])
simpleaxis(ax)
times = ripscore[0]['pre'].columns
labels = ['Pre-sleep', 'Post-sleep']
for k,i in zip(['pre', 'pos'],range(2)):
	plot(times, gaussFilt(ripscore[0][k].mean(0), (1,)), linewidth = 1, color = colors[i], label = labels[i])
axvline(0, linewidth =1, color = 'grey', alpha = 0.5)
title("SPWR score", y = 0.95)
legend(loc = 'upper left', edgecolor = None, facecolor = None, frameon = False)
text(1800,0.850, '$\mathbf{Non\ HD\ Neurons}$')
##############################################################
# HD SWR
##############################################################
ax = subplot(gs2[1,0:2])
simpleaxis(ax)
for k,i in zip(['pre', 'pos'],range(2)):
	plot(times, gaussFilt(ripscore[1][k].mean(0), (1,)), linewidth = 1, color = colors[i], label = labels[i])
legend(loc = 'upper left', edgecolor = None, facecolor = None, frameon = False)
axvline(0, linewidth =1, color = 'grey', alpha = 0.5)
xlabel("Time from SPWR (s)", fontsize = 5)
text(2000,3.4, '$\mathbf{HD\ Neurons}$')
# ##############################################################
# # NOHD THETA
# ##############################################################
# ax = subplot(gs2[0,2])
# simpleaxis(ax)
# y = [remscore[0][k]['mean'].mean() for k in ['pre', 'pos']]
# bar([0,1], y)
# xticks([0,1], ['pre', 'pos'])
# title("REM score", y = 0.95)

# ##############################################################
# # HD THETA
# ##############################################################
# ax = subplot(gs2[1,2])
# simpleaxis(ax)
# y = [remscore[1][k]['mean'].mean() for k in ['pre', 'pos']]
# bar([0,1], y)
# xticks([0,1], ['pre', 'pos'])

##############################################################
# COVARIANCE
##############################################################
gs3 = gridspec.GridSpec(3,1)
gs3.update(left = 0.75, right = 1, hspace = 0.5)
ax = subplot(gs3[0,0])
simpleaxis(ax)
# scatter(corr['pre'], corr['pos'], s = 1)


##############################################################
# PHASE DIFFERENCE
##############################################################
ax = subplot(gs3[1,0])
simpleaxis(ax)
x = phi['phi']-phi['phipre']
y = phi['phi']-phi['phipos']
x[(x < -np.pi)] += 2*np.pi
y[(y < -np.pi)] += 2*np.pi
x[(x > np.pi)] -= 2*np.pi
y[(y > np.pi)] -= 2*np.pi
# scatter(x, y, s = 1)
scatter(phi['phipre'], phi['phipos'], s = 1)


# ax1 = subplot(gs[0:,0])
# plot(snippet_filt_rem*0.01 + 50, color = 'black', linewidth = 0.5)
# # noaxis(ax1)
# ylabel("CA1 LFP")
# title("REM", fontsize = 5, y = 0.90)
# text(0.85, 0,'6-14 Hz', horizontalalignment='center', verticalalignment='center', transform=ax1.transAxes, fontsize = 4)
# plot([start_theta,start_theta+100000], [-0,-0], '-', linewidth = 0.5, color = 'black')
# text(0.1, -0.05,'100 ms', horizontalalignment='center', verticalalignment='center', transform=ax1.transAxes, fontsize = 4)
# p, t_theta = getPeaksandTroughs(nts.Tsd(t = snippet_filt_rem.index.values, d = snippet_filt_rem.values.flatten()), 5)
# for i in t_theta.index.values:
# 	plot([i,i], [0,78], color = 'grey', linewidth = 0.5)
# for i in range(len(p)-1):
# 	text(p.index.values[i]-1000, -4, str(i+1))

# count = 0
# msize = 3
# mwidth = 0.5
# for n in allneurons:	
# 	xt = spikes[int(n.split("_")[1])].loc[start_theta:end_theta].index.values
# 	if len(xt):			
# 		plot(xt, np.ones(len(xt))*count, '|', markersize = msize, markeredgewidth = mwidth, color = 'black')
# 	count+=1

# ypos = np.linspace(-10, -60, len(p))
# xpos = p.index.values - 2000
# for i in range(len(p)-1):
# 	text(xpos[0] - 100000, ypos[i], str(i+1))

# for i in range(len(p)-1):
# 	for j in range(len(p)-1):
# 		plot([xpos[i], xpos[i] + 80000], [ypos[j], ypos[j]], '-', linewidth = 0.5, color = 'black')
# 		plot([xpos[i], xpos[i]], [ypos[j], ypos[j]+5], '-', linewidth = 0.5, color = 'black')

# 		x, y = pop_rem.iloc[i+1].values, pop_rem.iloc[j+1].values
# 		x = x - x.min()
# 		y = y - y.min()
# 		x = x / x.max()
# 		y = y / y.max()
# 		x = x * 80000
# 		y = y * 5
# 		x = x + xpos[i]
# 		y = y + ypos[j]		
# 		scatter(x, y, 0.1, color = 'black')



# # ylabel("Thalamus")
# # ylabel("Correlation")
# # offset = 68800
# # style="<->,head_width=2,head_length=3"
# # kw = dict(arrowstyle=style, color="k")
# # ax1.add_patch(patches.FancyArrowPatch((t_theta.index.values[1]-offset,-1), (t_theta.index.values[2]-offset, -1),connectionstyle="arc3,rad=.5", **kw))
# # ax1.add_patch(patches.FancyArrowPatch((t_theta.index.values[3]-offset,-1), (t_theta.index.values[4]-offset,-1),connectionstyle="arc3,rad=.5", **kw))
# # ax1.add_patch(patches.FancyArrowPatch((t_theta.index.values[1]-offset,-40), (t_theta.index.values[4]-offset,-40),connectionstyle="arc3,rad=.5", **kw))


# ylim(-70, 60)




# ############ second column ###########################




# ax2 = subplot(gs[0:,1])
# plot(lfp_filt_hpc_swr.loc[start_rip:end_rip], color = 'black', linewidth = 0.5)
# noaxis(ax2)
# title("NON-REM", fontsize = 5, y = 0.9)
# text(0.85, 0,'100-300 Hz', horizontalalignment='center', verticalalignment='center', transform=ax2.transAxes, fontsize = 4)
# plot([start_rip,start_rip+40000], [-1,-1], '-', linewidth = 0.5, color = 'black')
# text(0.1, -0.05,'40 ms', horizontalalignment='center', verticalalignment='center', transform=ax2.transAxes, fontsize = 4)


# count = 0
# for n in allneurons:
# 	xt = spikes[int(n.split("_")[1])].loc[start_rip:end_rip].index.values
# 	count+=1
# 	if len(xt):					
# 		plot(xt, np.ones(len(xt))*count, '|', markersize = msize, markeredgewidth = mwidth, color = 'black')
		



# ########### POP CORR #######################
# # ax3 = subplot(gs[2,0], sharex = ax1)



# # ax4 = subplot(gs[2,1])
# #noaxis(ax4)









# ###############################################################################
# ax = subplot(gs[0,2])
# # axp = imshow(toplot['rem'][100:,100:], origin = 'lower', cmap = 'gist_heat')
# axp = imshow(rem_to_plot, origin = 'lower', cmap = 'gist_heat')
# ylabel("Theta Cycle")
# xlabel("Theta Cycle")
# # title('REM SLEEP')
# # cbaxes = fig.add_axes([0.32, 0.12, 0.01, 0.15])
# # cb = colorbar(axp, cax = cbaxes, orientation = 'vertical', cmap = 'gist_heat')
# # cbaxes.title.set_text('r')

# ###############################################################################
# # ax = subplot(gs[0, 1])
# # imshow(toplot['wake'], interpolation = None, origin = 'lower', cmap = 'gist_heat')
# # ylabel('Theta Cycle')
# # xlabel('Theta Cycle')
# # title("WAKE")

# ###############################################################################
# ax = subplot(gs[1, 2])
# imshow(toplot['rip'][0:200,0:200], origin = 'lower', cmap = 'gist_heat')
# # title('SHARP-WAVES RIPPLES')
# xlabel('SWR')
# ylabel('SWR')

# ###############################################################################
# ax = subplot(gs[2,2])
# simpleaxis(ax)
# # xtsym = np.array(list(xt[::-1]*-1.0)    +list(xt))
# # meanywak = np.array(list(meanywak[::-1])+list(meanywak))
# # meanyrem = np.array(list(meanyrem[::-1])+list(meanyrem))
# # meanyrip = np.array(list(meanyrip[::-1])+list(meanyrip))
# # varywak  = np.array(list(varywak[::-1]) +list(varywak))
# # varyrem  = np.array(list(varyrem[::-1]) +list(varyrem))
# # varyrip  = np.array(list(varyrip[::-1]) +list(varyrip))

# colors = ['red', 'blue', 'green']
# colors = ['#231123', '#af1b3f', '#ccb69b']

# # plot(xtsym, meanywak, '-', color = colors[0], label = 'theta(wake)')
# # plot(xtsym, meanyrem, '-', color = colors[1], label = 'theta(rem)')
# # plot(xtsym, meanyrip, '-', color = colors[2], label = 'ripple')
# # fill_between(xtsym, meanywak+varywak, meanywak-varywak, color = colors[0], alpha = 0.4)
# # fill_between(xtsym, meanyrem+varyrem, meanyrem-varyrem, color = colors[1], alpha = 0.4)
# # fill_between(xtsym, meanyrip+varyrip, meanyrip-varyrip, color = colors[2], alpha = 0.4)
# plot(xtime, meanyrem, '-', color = colors[1], label = 'REM')
# plot(xtime, meanywak, '-', color = colors[0], label = 'WAKE')
# plot(xtime, meanyrip, '-', color = colors[2], label = 'SWR')
# fill_between(xtime, meanyrem+varyrem, meanyrem-varyrem, color = colors[1], alpha = 0.4)
# fill_between(xtime, meanywak+varywak, meanywak-varywak, color = colors[0], alpha = 0.4)
# fill_between(xtime, meanyrip+varyrip, meanyrip-varyrip, color = colors[2], alpha = 0.4)

# xtime = xtime[::-1]*-1.0
# meanywak = meanywak[::-1]
# meanyrem = meanyrem[::-1]
# meanyrip = meanyrip[::-1]
# varywak = varywak[::-1]
# varyrem = varyrem[::-1]
# varyrip = varyrip[::-1]
# plot(xtime, meanyrem, '-', color = colors[1])
# plot(xtime, meanywak, '-', color = colors[0])
# plot(xtime, meanyrip, '-', color = colors[2])
# fill_between(xtime, meanyrem+varyrem, meanyrem-varyrem, color = colors[1], alpha = 0.4)
# fill_between(xtime, meanywak+varywak, meanywak-varywak, color = colors[0], alpha = 0.4)
# fill_between(xtime, meanyrip+varyrip, meanyrip-varyrip, color = colors[2], alpha = 0.4)


# axvline(0, linestyle = '--', color = 'grey')

# # ylim(0.0, 0.3)

# legend(edgecolor = None, facecolor = None, frameon = False)
# xlabel('Time between events (s) ')
# ylabel("r")


savefig("../figures/figures_articles/figart_3.pdf", dpi = 900, facecolor = 'white')
os.system("evince ../figures/figures_articles/figart_3.pdf &")




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

###############################################################################################################
# TO LOAD
###############################################################################################################
data_directory 	= '/mnt/DataGuillaume/MergedData/'
session 		= 'Mouse12/Mouse12-120810'
generalinfo 	= scipy.io.loadmat(data_directory+session+'/Analysis/GeneralInfo.mat')
shankStructure 	= loadShankStructure(generalinfo)
if len(generalinfo['channelStructure'][0][0][1][0]) == 2:
	hpc_channel 	= generalinfo['channelStructure'][0][0][1][0][1][0][0] - 1
else:
	hpc_channel 	= generalinfo['channelStructure'][0][0][1][0][0][0][0] - 1	
spikes,shank	= loadSpikeData(data_directory+session+'/Analysis/SpikeData.mat', shankStructure['thalamus'])		
allneurons 		= [session.split("/")[1]+"_"+str(list(spikes.keys())[i]) for i in spikes.keys()]

#################
# POPULATION ACTIVITY AND EPISODE
store 			= pd.HDFStore("../data/population_activity/"+session.split("/")[1]+".h5")
theta_rem_ep 	= store['theta_rem_ep']
rip_ep 			= store['rip_ep']
theta_rem_pop 	= store['rem']
store.close()
store 			= pd.HDFStore("../data/corr_pop/"+session.split("/")[1]+".h5")
rem_corr 		= store['rem_corr']
wak_corr 		= store['wak_corr']
rip_corr 		= store['rip_corr']
store.close()
################
# CHOOSE THE EXAMPLE
i1, i2 = (343,503) # the image
rem_to_plot = rem_corr.values[i1:i2,i1:i2]
rem_t1, rem_t2 = (rem_corr.index.values[i1],rem_corr.index.values[i2])
rem_ex = (i1+62,i1+68) # the snippet
start_theta, end_theta = (rem_corr.index.values[rem_ex[0]],rem_corr.index.values[rem_ex[1]]) 
store = pd.HDFStore('../figures/figures_articles/figure3/'+session.split("/")[1]+'.h5')
snippet_rem = store['theta_rem']
snippet_filt_rem = store['theta_filt_rem']
store.close()

pop_rem = theta_rem_pop.loc[start_theta:end_theta]

start_rip, end_rip = (92550000,92880000)
store 			= pd.HDFStore("../figures/figures_articles/figure1/"+'snippet_'+session.split("/")[1]+'.h5')
lfp_filt_hpc_swr = store['lfp_filt_hpc_swr']
swr_ep = store['swr_ep']
store.close()


###############
# THE STATISTICS
data = cPickle.load(open('../data/to_plot_corr_pop.pickle', 'rb'))
xtime	 	= data	['xt'	  ]
meanywak 	= data['meanywak']
meanyrem 	= data['meanyrem']
meanyrip 	= data['meanyrip']
toplot 	= data['toplot'  ]
varywak		= data['varywak']
varyrem		= data['varyrem']
varyrip		= data['varyrip']









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
	"xtick.labelsize": 6,
	"ytick.labelsize": 6,
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
# outer = gridspec.GridSpec(3,3, wspace = 0.4, hspace = 0.5)#, height_ratios = [1,3])#, width_ratios = [1.6,0.7]) 
gs = gridspec.GridSpec(3,3, wspace = 0.35, hspace = 0.35)#, wspace = 0.4, hspace = 0.4)
# gs = gridspec.GridSpecFromSubplotSpec(1,1, subplot_spec = outer[0])

############ first column ###########################
ax1 = subplot(gs[0:,0])
plot(snippet_filt_rem*0.01 + 50, color = 'black', linewidth = 0.5)
# noaxis(ax1)
ylabel("CA1 LFP")
title("REM", fontsize = 5, y = 0.90)
text(0.85, 0,'6-14 Hz', horizontalalignment='center', verticalalignment='center', transform=ax1.transAxes, fontsize = 4)
plot([start_theta,start_theta+100000], [-0,-0], '-', linewidth = 0.5, color = 'black')
text(0.1, -0.05,'100 ms', horizontalalignment='center', verticalalignment='center', transform=ax1.transAxes, fontsize = 4)
p, t_theta = getPeaksandTroughs(nts.Tsd(t = snippet_filt_rem.index.values, d = snippet_filt_rem.values.flatten()), 5)
for i in t_theta.index.values:
	plot([i,i], [0,78], color = 'grey', linewidth = 0.5)
for i in range(len(p)-1):
	text(p.index.values[i]-1000, -4, str(i+1))

count = 0
msize = 3
mwidth = 0.5
for n in allneurons:	
	xt = spikes[int(n.split("_")[1])].loc[start_theta:end_theta].index.values
	if len(xt):			
		plot(xt, np.ones(len(xt))*count, '|', markersize = msize, markeredgewidth = mwidth, color = 'black')
	count+=1

ypos = np.linspace(-10, -60, len(p))
xpos = p.index.values - 2000
for i in range(len(p)-1):
	text(xpos[0] - 100000, ypos[i], str(i+1))

for i in range(len(p)-1):
	for j in range(len(p)-1):
		plot([xpos[i], xpos[i] + 80000], [ypos[j], ypos[j]], '-', linewidth = 0.5, color = 'black')
		plot([xpos[i], xpos[i]], [ypos[j], ypos[j]+5], '-', linewidth = 0.5, color = 'black')

		x, y = pop_rem.iloc[i+1].values, pop_rem.iloc[j+1].values
		x = x - x.min()
		y = y - y.min()
		x = x / x.max()
		y = y / y.max()
		x = x * 80000
		y = y * 5
		x = x + xpos[i]
		y = y + ypos[j]		
		scatter(x, y, 0.1, color = 'black')



# ylabel("Thalamus")
# ylabel("Correlation")
# offset = 68800
# style="<->,head_width=2,head_length=3"
# kw = dict(arrowstyle=style, color="k")
# ax1.add_patch(patches.FancyArrowPatch((t_theta.index.values[1]-offset,-1), (t_theta.index.values[2]-offset, -1),connectionstyle="arc3,rad=.5", **kw))
# ax1.add_patch(patches.FancyArrowPatch((t_theta.index.values[3]-offset,-1), (t_theta.index.values[4]-offset,-1),connectionstyle="arc3,rad=.5", **kw))
# ax1.add_patch(patches.FancyArrowPatch((t_theta.index.values[1]-offset,-40), (t_theta.index.values[4]-offset,-40),connectionstyle="arc3,rad=.5", **kw))


ylim(-70, 60)




############ second column ###########################




ax2 = subplot(gs[0:,1])
plot(lfp_filt_hpc_swr.loc[start_rip:end_rip], color = 'black', linewidth = 0.5)
noaxis(ax2)
title("NON-REM", fontsize = 5, y = 0.9)
text(0.85, 0,'100-300 Hz', horizontalalignment='center', verticalalignment='center', transform=ax2.transAxes, fontsize = 4)
plot([start_rip,start_rip+40000], [-1,-1], '-', linewidth = 0.5, color = 'black')
text(0.1, -0.05,'40 ms', horizontalalignment='center', verticalalignment='center', transform=ax2.transAxes, fontsize = 4)


count = 0
for n in allneurons:
	xt = spikes[int(n.split("_")[1])].loc[start_rip:end_rip].index.values
	count+=1
	if len(xt):					
		plot(xt, np.ones(len(xt))*count, '|', markersize = msize, markeredgewidth = mwidth, color = 'black')
		



########### POP CORR #######################
# ax3 = subplot(gs[2,0], sharex = ax1)



# ax4 = subplot(gs[2,1])
#noaxis(ax4)









###############################################################################
ax = subplot(gs[0,2])
# axp = imshow(toplot['rem'][100:,100:], origin = 'lower', cmap = 'gist_heat')
axp = imshow(rem_to_plot, origin = 'lower', cmap = 'gist_heat')
ylabel("Theta Cycle")
xlabel("Theta Cycle")
# title('REM SLEEP')
# cbaxes = fig.add_axes([0.32, 0.12, 0.01, 0.15])
# cb = colorbar(axp, cax = cbaxes, orientation = 'vertical', cmap = 'gist_heat')
# cbaxes.title.set_text('r')

###############################################################################
# ax = subplot(gs[0, 1])
# imshow(toplot['wake'], interpolation = None, origin = 'lower', cmap = 'gist_heat')
# ylabel('Theta Cycle')
# xlabel('Theta Cycle')
# title("WAKE")

###############################################################################
ax = subplot(gs[1, 2])
imshow(toplot['rip'][0:200,0:200], origin = 'lower', cmap = 'gist_heat')
# title('SHARP-WAVES RIPPLES')
xlabel('SWR')
ylabel('SWR')

###############################################################################
ax = subplot(gs[2,2])
simpleaxis(ax)
# xtsym = np.array(list(xt[::-1]*-1.0)    +list(xt))
# meanywak = np.array(list(meanywak[::-1])+list(meanywak))
# meanyrem = np.array(list(meanyrem[::-1])+list(meanyrem))
# meanyrip = np.array(list(meanyrip[::-1])+list(meanyrip))
# varywak  = np.array(list(varywak[::-1]) +list(varywak))
# varyrem  = np.array(list(varyrem[::-1]) +list(varyrem))
# varyrip  = np.array(list(varyrip[::-1]) +list(varyrip))

colors = ['red', 'blue', 'green']
colors = ['#231123', '#af1b3f', '#ccb69b']

# plot(xtsym, meanywak, '-', color = colors[0], label = 'theta(wake)')
# plot(xtsym, meanyrem, '-', color = colors[1], label = 'theta(rem)')
# plot(xtsym, meanyrip, '-', color = colors[2], label = 'ripple')
# fill_between(xtsym, meanywak+varywak, meanywak-varywak, color = colors[0], alpha = 0.4)
# fill_between(xtsym, meanyrem+varyrem, meanyrem-varyrem, color = colors[1], alpha = 0.4)
# fill_between(xtsym, meanyrip+varyrip, meanyrip-varyrip, color = colors[2], alpha = 0.4)
plot(xtime, meanyrem, '-', color = colors[1], label = 'REM')
plot(xtime, meanywak, '-', color = colors[0], label = 'WAKE')
plot(xtime, meanyrip, '-', color = colors[2], label = 'SWR')
fill_between(xtime, meanyrem+varyrem, meanyrem-varyrem, color = colors[1], alpha = 0.4)
fill_between(xtime, meanywak+varywak, meanywak-varywak, color = colors[0], alpha = 0.4)
fill_between(xtime, meanyrip+varyrip, meanyrip-varyrip, color = colors[2], alpha = 0.4)

xtime = xtime[::-1]*-1.0
meanywak = meanywak[::-1]
meanyrem = meanyrem[::-1]
meanyrip = meanyrip[::-1]
varywak = varywak[::-1]
varyrem = varyrem[::-1]
varyrip = varyrip[::-1]
plot(xtime, meanyrem, '-', color = colors[1])
plot(xtime, meanywak, '-', color = colors[0])
plot(xtime, meanyrip, '-', color = colors[2])
fill_between(xtime, meanyrem+varyrem, meanyrem-varyrem, color = colors[1], alpha = 0.4)
fill_between(xtime, meanywak+varywak, meanywak-varywak, color = colors[0], alpha = 0.4)
fill_between(xtime, meanyrip+varyrip, meanyrip-varyrip, color = colors[2], alpha = 0.4)


axvline(0, linestyle = '--', color = 'grey')

# ylim(0.0, 0.3)

legend(edgecolor = None, facecolor = None, frameon = False)
xlabel('Time between events (s) ')
ylabel("r")


savefig("../figures/figures_articles/figart_3.pdf", dpi = 900, bbox_inches = 'tight', facecolor = 'white')
os.system("evince ../figures/figures_articles/figart_3.pdf &")


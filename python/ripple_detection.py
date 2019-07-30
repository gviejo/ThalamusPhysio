import numpy as np
import pandas as pd
from pylab import *
from matplotlib import pyplot as plt
import neuroseries as nts
#from matplotlib import pyplot
from functions import downsampleDatFile
from functions import loadLFP
from functions import loadBunch_Of_LFP
import scipy
import scipy.signal
import sys
from sklearn.decomposition import PCA
#Path for the directory where the data is stored
data_directory = r'D:\Mouse12-120808'
pos_dir = r'D:\Mouse12-120808\Analysis\Mouse12-120808_wake.txt'
pos_dir1 = r'D:\Mouse12-120808\Analysis\Mouse12-120808_pretraining_sleep.txt'
lfp_dir = r'D:\Mouse12-120808\Mouse12-120808.lfp'                                  #path for downsampled(at 1250 hz) LFP file
#lfp_seq	= loadBunch_Of_LFP(lfp_dir,start = 0.0, stop = 1000.0, channel = 64)		#Function to convert LFP file into NTS Series Frame...Function(time in seconds)

def bandPass_filt(lfp_unfiltered, low_cut, high_cut, sampling_freq = 1250, order = 4):
	#4th order Butterworth Bandpass filter for filtering the LFP data(single channel) between the frequency range of 150-250 Hz.
	#Returns a numpy array of band-filtered signal(Not the Pandas dataframe)
	signal_un = lfp_unfiltered.values
	nyq = 0.5*sampling_freq
	low = low_cut/nyq
	high = high_cut/nyq
	b,a = scipy.signal.butter(order, [low, high], btype = 'bandpass')
	filt_band_signal = scipy.signal.lfilter(b, a, signal_un)
	return filt_band_signal

#Buggy Implementation of Neuroseries for implementing ripple time epochs....Don't ever use Neuroseries this way. The way Neuroseries IntervalSet works is that it stores only the intersection of two consecutive intervals.
'''
def make_Epochs(start, end):
	#Function to make an nts.IntervalSet dataframe with starting and ending epochs
	import numpy as np
	import neuroseries as nts
	#Firstly, check whether both the lists are of same size or not
	if not (len(start) == len(end)):
		print("Start and End array lists are not of same dimension. Epochs IntervalSet can't be developed.")
		sys.exit()
	else:
		eps = nts.IntervalSet(start, end)
		#print(eps)
		return eps
'''

def make_Epochs(start, end):
	#Function to make an nts.IntervalSet dataframe with starting and ending epochs
	import numpy as np
	import neuroseries as nts
	#Firstly, check whether both the lists are of same size or not
	if not (len(start) == len(end)):
		print("Start and End array lists are not of same dimension. Epochs IntervalSet can't be developed.")
		sys.exit()
	else:
		nts_array = []
		for i in range(len(start)):
			nts_array.append(nts.IntervalSet(start[i], end[i]))	
		#print(eps)
		return nts_array

def plot_scatter_3d(matrix, color_val, colorbar = True):
	import matplotlib.pyplot as plt
	from mpl_toolkits import mplot3d
	fig = plt.figure()
	ax = plt.axes(projection = "3d")
	p = ax.scatter3D(matrix[:,0], matrix[:,1], matrix[:,2], c= color_val, cmap = 'hsv')
	if colorbar == True:
		fig.colorbar(p)
	plt.show()

def entropy_matrix(entropy_decoding2):
	feature_matrix = np.array([])
	l = len(entropy_decoding2[0].values)

	for i in range(len(entropy_decoding2)):
		
		if i == 0:
			feature_matrix = np.append(feature_matrix, entropy_decoding2[i].values)
		else:
			if len(entropy_decoding2[i].values) == l:
				feature_matrix = np.vstack((feature_matrix, entropy_decoding2[i].values))
	return feature_matrix

def DTW_Matrix(feature_matrix):
	from fastdtw import fastdtw
	from scipy.spatial.distance import euclidean
	print("Starting Dynamic Time Warping between signals..")
	dtw_matrix = np.zeros(((feature_matrix.shape[0]), (feature_matrix.shape[0])))
	for i in range(feature_matrix.shape[0]):
		for j in range(i, (feature_matrix.shape[0]), 1):
			d, path = fastdtw(feature_matrix[i,:], feature_matrix[j,:], dist = euclidean)
			dtw_matrix[i][j] = d

	print("Successfully developed matrix.")		
	return dtw_matrix

def writeNeuroscopeEvents(path, ep, name):
	f = open(path, 'w')
	for i in range(len(ep)):
		f.writelines(str(ep.as_units('ms').iloc[i]['start']) + " "+name+" start "+ str(1)+"\n")
		#f.writelines(str(ep.as_units('ms').iloc[i]['peak']) + " "+name+" start "+ str(1)+"\n")
		f.writelines(str(ep.as_units('ms').iloc[i]['end']) + " "+name+" end "+ str(1)+"\n")
	f.close()		
	return

def correct_Epochs(position_data):
	#Load the proper epoch timings where the position was correctly detected in the experiment.
	#position_data : nts.TsdFrame of position data in the format --> [time, x, y, angle]
	import neuroseries as nts
	ep = nts.IntervalSet(position_data.index.values[0], position_data.index.values[-1])
	return ep

def LFP_Feature_Matrix(lfp_raw, ripple_events, epoch, warping = False):
	print("Starting Ripple LFP Extraction..")
	ripples = ripple_events.values
	mid = 1000000*(ripples[:,1])
	#print(mid)
	arrr = 1000000*(ripples[:,2] - ripples[:,1])
	arrl = 1000000*(ripples[:,1] - ripples[:,0])
	#jl  = np.argmax(arrl)
	max_diffl = np.max(arrl)
	max_diffr = np.max(arrr)
	#addl = max_diffl - arrl
	#addr = max_diffr - arrr
	ripples_mod = np.array([(mid - max_diffl) + epoch['start'].values[0], (mid + max_diffr) + epoch['start'].values[0]])
	ripples_mod = ripples_mod.T
	#print(ripples_mod)
	rip_ep = make_Epochs(ripples_mod[:,0], ripples_mod[:,1])

	features = np.array([])
	for i in range(len(rip_ep.index.values)):
		series = (lfp_raw.restrict(rip_ep.iloc[[i]]).values)
		if i == 0:
			features = np.append(features, series)
		else:	
			try:
				features = np.vstack((features, series))				
			except:
				if warping == False:
					print("Not appended "+str(i))
					continue
				elif warping == True:
					print("To DO")
					sys.exit()	
	feature_map = np.asarray(features)
	return feature_map

def load_Positions(path, ep):
	#Load the position HDF file as TsDFrame
	#Epoch to be selected from all the recording time. Available options : 'wake', 'presleep', 'postsleep'
	import os
	import neuroseries as nts
	if not os.path.exists(path):
		print("The path "+path+" doesn't exist; Exiting ...")
		sys.exit()    
	new_path = os.path.join(path, 'Analysis/')
	if os.path.exists(new_path):
		new_path    = os.path.join(path, 'Analysis/')
		store = pd.HDFStore(new_path + 'Positions.h5')
		pos = store['positions']
		try:
			presl = (pos.loc[ep])
			positions_vs_t = nts.TsdFrame(t = (presl['timestamp']).values, d = presl[['x','y','angle']].values, columns  = presl[['x','y','angle']].columns, time_units = 's')
			return positions_vs_t
		except:
			print("Specific epoch name didn't exist is the position data. Exiting.....")
			sys.exit()

def load_ripples(data_dir):
	import os
	if data_dir == None:
		print("The path "+ str(data_dir) +" doesn't exist; Please provide the correct path directory and try again...")
		sys.exit()   
	new_path = os.path.join(data_dir, 'Analysis/')

	if os.path.exists(new_path):
		new_path    = os.path.join(data_dir, 'Analysis/')
		if os.path.isfile(os.path.join(new_path, 'Swr_Ripple.h5')):
			print("Loading Ripple data from the file")
			rpt_frame = pd.read_hdf(new_path + 'Swr_Ripple.h5')
			return rpt_frame

def plot_sharing_calc(bin_size_un, bin_size_ov, interval):
	d1 = bin_size_un/2
	d2 = bin_size_ov/2
	d1 = d1*1000 + interval   # microseconds
	d2 = d2*1000 + interval + 20*1000   # microseconds
	return d1,d2

def ripple_Decoding(ripple_event, data_dir, epoch, lfp, save_pdf = False):

	#Purpose: Decoding the ripple event to the Head-Direction angle data
	'''
	NOTE ON function arguments:
	ripple_event : pandas Dataframe with info about ripple events in the following form : [start time, peak time, end time, Average Power, Instantanous Frequency of ripple].
	data_dir 	 : path to the directory where all the session data is stored.
	epoch        : name of epoch in which decoding is needed to be done.
	lfp          : Downasampled neuroseries Time Series with single channel. It can be obtained using loadLFP() function to obtain a time series from a single channel. 
	'''

	from matplotlib.backends.backend_pdf import PdfPages
	import itertools
	
	bin_size_un = 20 #ms
	bin_size_ov = 100 #ms
	percent_overlap = 0.75
	d1,d2 = plot_sharing_calc(bin_size_un, bin_size_ov, (500 * 1000))

	ripple_event = ripple_event.values
	#Loading the correct angular position wrt time from Positions.h5 file.
	positions = load_Positions(data_directory, epoch)
	ep = correct_Epochs(positions)
	#Loading the spike and shank data
	from wrappers import loadSpikeData, loadEpoch, loadPosition, loadXML
	spikes, shank  = loadSpikeData(data_dir)
	n_channels, fs, shank_to_channel = loadXML(data_dir)
	
	#Loading the wakefulness period data for decoding as it is required to calculate tuning curves and occupancy
	pos_wake = load_Positions(data_directory, 'wake')
	wake_ep = correct_Epochs(pos_wake)
	
	#Computing Angular Tuning curves for the detection of head-direction cells using Reyleigh's test.
	#NOTE: Use wakefulness position data to calculate tuning curves and occupancy
	from functions import computeAngularTuningCurves
	tuning_curves = computeAngularTuningCurves(spikes, pos_wake['angle'], wake_ep, 61)    #spikes: spike data, angle : angle data from all the, no of bins for angles

	#Defining correct HD Cells using Reileigh's test on angular tuning curves.
	from functions import findHDCells
	hd_idx, stat = findHDCells(tuning_curves)
	tuning_curves = tuning_curves[hd_idx]

	#To calculate prior using Poisson spiking assumption 	
	from functions import decodeHD, decodeHD_overlap2
	spikes_hd = {k:spikes[k] for k in hd_idx}
	occupancy = np.histogram(pos_wake['angle'], np.linspace(0, 2*np.pi, 61), weights = np.ones_like(pos_wake['angle'])/float(len(pos_wake['angle'])))[0]

	#Here, we will try to decode the HD angle during wake state to see whether the decoding priors like Ocuupancy map and tuning curves are correct or not.
	wake_dec, prob_wake_dec = decodeHD(tuning_curves, spikes_hd, wake_ep, occupancy, 200)
	posterior_wake = nts.Tsd(t = prob_wake_dec.index.values, d = prob_wake_dec.max(1).values, time_units = 'ms')

	'''
	figure()
	subplot(2,1,1)
	subplot(2,1,1).set_title('Bayesian Decoding during Wakefulness')
	plot(pos_wake['angle'].restrict(wake_ep), label = 'True HD')
	plot(wake_dec, label = 'Decoded HD(Bayesian)')
	legend()
	subplot(2,1,2)
	subplot(2,1,2).set_title('Posterior Probability(Bayesian Decoding)')
	matshow(prob_wake_dec)
	legend()
	show()	
	'''
	
	print(ripple_event.shape)
	#So far, we have calculated the general information that we need for decoding such as occupancy map and spikes of HD cells. They are 
	#important irrespective of the event we are trying to decode.
	#Now, we will calculate the decoding and probable_angle(wrt time) for each ripple event 
	d = 500 * 1000
	ripple_epochs1 = make_Epochs((ep['start'].values[0] + (ripple_event[:,1]*1000000) - d1), (ep['start'].values[0] + (ripple_event[:,1]*1000000) + d1))           #ripple_event is in seconds here...
	ripple_epochs2 = make_Epochs((ep['start'].values[0] + (ripple_event[:,1]*1000000) - d2), (ep['start'].values[0] + (ripple_event[:,1]*1000000) + d2))
	ripple_epochs = make_Epochs((ep['start'].values[0] + (ripple_event[:,1]*1000000) - d), (ep['start'].values[0] + (ripple_event[:,1]*1000000) + d))
	proba_angle = []      #Since there are multiple events to decode, the proba_angle will now be a list of decodings for each of the ripple event
	decoded  = []
	#e = 67

	#signal = signal.as_units('s')

	if save_pdf == True:
		print("Initiating Ripple Plotting..")
		with PdfPages("Ripple_plots_all.pdf") as Pdf:

			#Some preprocessing to display LFP together...
			low_cut = 200
			high_cut = 400
			frequency = 1250
			
			signal = bandPass_filt(lfp_unfiltered = lfp, low_cut = low_cut, high_cut = high_cut, sampling_freq = frequency)
			ts = np.arange(len(signal))/frequency
			signal = nts.TsdFrame(ts, signal, time_units = 's')

			for e in range(5):
			#for e in range(len(ripple_epochs.index.values)):
				#Decoding using Non-overlapping equal width bins
				decoding, p_angle = decodeHD(tuning_curves, spikes_hd, ripple_epochs1[e], occupancy, bin_size_un)
				#posterior_prob = nts.Tsd(t = p_angle.index.values, d = p_angle.max(1).values, time_units = 'ms')
				
				p_mat = p_angle.values
				entropy1 = -(p_angle*np.log2(p_angle)).sum(1)

				#print(entropy1)
				#entropy1 = nts.Tsd(t = p_angle.index.values, d = ent1, time_units = 'ms')
				
				#Decoding using Overlapping bins 
				decoding2, p_angle2 = decodeHD_overlap2(tuning_curves, spikes_hd, ripple_epochs2[e], occupancy, bin_size_ov, percent_overlap = percent_overlap)
				

				if type(decoding2).__name__ == 'NoneType':
					continue
				p_mat2 = p_angle2.values
				entropy2 = -(p_angle2*np.log2(p_angle2)).sum(1) 
				
				#print(entropy2)	
				#posterior_prob2 = nts.Tsd(t = p_angle2.index.values, d = p_angle2.max(1).values, time_units = 'ms')
				#entropy2 = nts.Tsd(t = p_angle2.index.values, d = [-p*math.log(p) for p in posterior_prob2.values], time_units = 'ms')

				frac = (1/len(spikes_hd))
				ind_dix = pd.DataFrame(hd_idx, index = [tuning_curves[k].idxmax() for k in hd_idx], columns = ["cell_id"])
				#ind_dix = {tuning_curves[k].idxmax():k for k in hd_idx}
				w = ind_dix.sort_index()
				#print(hd_idx)

				lfp_view = signal.restrict(ripple_epochs[e])
				#lfp_view = signal[((ep.as_units('s')['start'].values[0]) + (ripple_event[int(e),0]) - resolution) : ((ep.as_units('s')['start'].values[0]) + (ripple_event[int(e),2]) + resolution)]
			
				print("Plotting Figure "+ str(e+1)+".")		
				figure()
				ax1 = plt.subplot(4,1,1)
				ax1.set_title('LFP Bandpassed('+ str(low_cut) + '-' + str(high_cut) + 'Hz)')
				ax1.set_ylabel('mV')
				ax1.plot(lfp_view)
				ax1.plot(lfp.restrict(ripple_epochs[e]))
				
				ax2 = plt.subplot(4,1,2, sharex = ax1)
				[[ax2.axvline(_x, linewidth=1, color='r', ymin = i*frac, ymax = (i+1)*frac) for _x in spikes_hd[col].restrict(ripple_epochs[e]).index.values] for col,i in itertools.zip_longest(w['cell_id'].values, np.arange(len(spikes_hd)))]
				#[ax2.axhline((2*i +1)*frac/2, linewidth=1, color='black') for i in range(len(spikes_hd)+1)]
				ax2.axis('tight')
				ax2.set_yticks([(2*i + 1)*frac/2 for i in range(len(hd_idx)+1)])
				ax2.set_yticklabels([str('%.2f'%i) for i in w.index.values])
				#ax2.set_title('Rastor Plot')
				ax2.set_ylabel('HD Neuron acc. Preferred Angle')
				
				ax3 = plt.subplot(4,1,3, sharex = ax2)
				#ax3.set_title('Bayesian Decoding')
				ax3.plot(positions['angle'].restrict(ripple_epochs[e]), label = 'True HD')
				ax3.plot(decoding, label = 'Decoded HD')
				ax3.set_ylabel('Decoded Angle(Radians)')
				ax3.plot(decoding2, label = 'Decoded HD(overlapping)')
				#ax3.axvline(x = (ripple_epochs.iloc[[e]])['start'].values[0], color = 'green', label = 'Starting point of SWR')
				#ax3.axvline(x = (ripple_epochs.iloc[[e]])['end'].values[0], color = 'red', label = 'End point of SWR')
				ax3.legend()
				
				ax4 = plt.subplot(4,1,4)
				#ax4.set_title('Posterior Prob(Bayesian)')
				ax4.plot(entropy1, label = 'Entropy')
				ax4.plot(entropy2, label = 'Entropy(Overlapping)')
				ax4.set_ylabel('Entropy of Decoding')	
				#ax4.axvline(x = (ripple_epochs.iloc[[e]])['start'].values[0], color = 'green', label = 'Starting point of SWR')
				#ax4.axvline(x = (ripple_epochs.iloc[[e]])['end'].values[0], color = 'red', label = 'End point of SWR')
				ax4.legend()
				mng = plt.get_current_fig_manager()
				mng.window.showMaximized()
				plt.pause(0.2)
				try:
					Pdf.savefig(dpi = 600)
				except ValueError:
					continue
				#legend()
				#show()
				
				plt.close()

		print("All Plotted. Done!")	
	#sys.exit()

	entropy_un = []
	entropy_b = []
	decoded2 = []
	decoded = []
	print("trying decoding")
	
	for i in range(len(ripple_epochs2)):
		#decoding, p_angle = decodeHD(tuning_curves, spikes_hd, ripple_epochs1[i], occupancy, bin_size_un)		
		#decoded.append(decoding)
		
		#decoding, p_angle = decodeHD(tuning_curves, spikes_hd, ripple_epochs.iloc[[i]], occupancy, 20)
		decoding2, p_angle2 = decodeHD_overlap2(tuning_curves, spikes_hd, ripple_epochs2[i], occupancy, bin_size_ov, percent_overlap = percent_overlap)
		print(len(decoding2))
		decoded2.append(decoding2)
		#entropy1 = -(p_angle*np.log2(p_angle)).sum(1)
		#entropy_un.append(entropy1)	
		#decoded.append(decoding)
		#proba_angle.append(p_angle)
		#entropy1 = -(p_angle*np.log2(p_angle)).sum(1)
		#entropy_un.append(entropy1)	
		p_mat2 = p_angle2.values
		entropy2 = -(p_angle2*np.log2(p_angle2)).sum(1)
		entropy_b.append(entropy2)
		print("Done with "+ str(i))
	

	return entropy_b, decoded2

def decoding_overlap(tuning_curves, bin_size, spikes, neuron_order, t, px):
	#My way of defining overlapping bins
	#bin_size = 40 # ms
	bins = np.arange(0, 2000+2*bin_size, bin_size) - 1000 - bin_size/2
	obins = np.vstack((bins-bin_size/2,bins)).T.flatten()
	obins = np.vstack((obins,obins+bin_size)).T
	times = obins[:,0]+(np.diff(obins)/2).flatten()
	
	
	# My function to compute the histogram 
	def histo(spk, obins):
		n = len(obins)
		count = np.zeros(n)
		for i in range(n):
			count[i] = np.sum((spk>obins[i,0]) * (spk < obins[i,1]))
		return count
	# When I do the binning for one SWR time t

	spike_counts = pd.DataFrame(index = times, columns = neuron_order)
	tbins = t + obins
	for k in neuron_order:
		spike_counts[k] = histo(spikes[k].as_units('ms').index.values, tbins)

	tcurves_array = tuning_curves.values
	spike_counts_array = spike_counts.values
	proba_angle = np.zeros((spike_counts.shape[0], tuning_curves.shape[0]))
	
	part1 = np.exp(-(bin_size/1000)*tcurves_array.sum(1))	
	part2 = px
	
	for i in range(len(proba_angle)):
		part3 = np.prod(tcurves_array**spike_counts_array[i], 1)
		p = part1 * part2 * part3
		proba_angle[i] = p/p.sum() #Normalization process here
	
	#print(spike_counts)
	proba_angle  = pd.DataFrame(index = spike_counts.index.values, columns = tuning_curves.index.values, data= proba_angle)	
	# proba_angle = proba_angle.astype('float')		
	decoded = nts.Tsd(t = proba_angle.index.values, d = proba_angle.idxmax(1).values, time_units = 'ms')
	return decoded, proba_angle	

def findRipples(lfp = None, frequency = 1250.0, plotting = False, noise_ch = None, minRipLen = 20, maxRipLen = 350, minInterRippleInterval = 30, low_thresFactor = 2, high_thresFactor = 7, sd = 0, save_output = False, data_dir = None):
	'''
	#NOTE on SWR Parameters           
	#Ripple Envelope must exceed low_thresFactor*stdev 
	#Ripple peak must exceed high_thresFactor*stdev
	Arguments:
	lfp     			   : neuroseries Time Series with downsampled data from single channel.
	frequency 			   : Sampling frequency of the signal. Default = 1250 Hz after downsampling from 20,000Hz.
	plotting  			   : Set True if you want to display the start & end points of SWRs on graph.
	noise_ch  			   : Use of noise channel to further eliminate the noisy events that look like ripples.
	minRipLen              : Minimum ripple length
	maxRipLen              : Maximum ripple length
	minInterRippleInterval : minimum distance between two successive ripples.
	low_thresFactor        : lower threshold for power factor
	high_thresFactor       : higher threshold for power factor
	sd                     : Standard Deviation
	save_output            : Use if you want to save the ripple output and don't want to waste time by running the detection code everytime. 
	data_dir               : directory where all the session data is stored.
	'''

	#Declaring array to store ripple start, peak, end and peak Normalized Power event.	
	ripples = np.array([])
	low_cut = 150
	high_cut = 300
	windowLength = np.floor(frequency/1250*11)
	import os
	if data_dir == None:
		print("The path "+ str(data_dir) +" doesn't exist; Please provide the correct path directory and try again...")
		sys.exit()   
	new_path = os.path.join(data_dir, 'Analysis/')

	if os.path.exists(new_path):
		new_path    = os.path.join(data_dir, 'Analysis/')
		if os.path.isfile(os.path.join(new_path, 'Swr_Ripple.h5')):
			print("Loading Ripple data from the file")
			rpt_frame = pd.read_hdf(new_path + 'Swr_Ripple.h5')
			ripples = rpt_frame.values

			if type(lfp).__name__ == 'NoneType':
				return rpt_frame
			elif type(lfp).__name__ != 'NoneType':	
				signal = bandPass_filt(lfp_unfiltered = lfp, low_cut = low_cut, high_cut = high_cut, sampling_freq = frequency)
					
				squared_signal = np.square(signal)     #Need to convert pandas dataframe to array before squaring
					
				ts = np.arange(len(squared_signal))/frequency
				ts = ts.reshape((len(ts),1))
				
				window = np.ones((int(windowLength),1))/windowLength       #Floor function returns a float value. Needs to be converted to int
				window = window.flatten()
				shift = int((len(window)-1)/2)    #Shift value for the signal
				#Now, we will apply Normalized Linear Transformation on the signal
				#Here:
				# y0 : Signal after transformation
				# zf : z-values of signal transformation
				zi0 = scipy.signal.lfilter_zi(window,1)
				y0, zf = scipy.signal.lfilter(window, 1, squared_signal, zi = zi0)
				y1, zf1 = scipy.signal.lfilter(window, 1, signal, zi = zi0) 
				#Picking up the desired range of values according to the shift values   
				normalized_SquareSignal = np.append(y0[shift+1:],zf[0:shift+1])
				normalized_signal = np.append(y1[shift+1:],zf1[0:shift+1])
				if not (sd):
					sd = np.std(normalized_SquareSignal)
				meanA = np.mean(normalized_SquareSignal)	
				A = normalized_SquareSignal.reshape((len(normalized_SquareSignal),1))
				normalized_SquareSignal = np.divide((A - np.tile(meanA, [len(A),1])),(np.tile(sd, [len(A),1])))
				normalized_SquareSignal = normalized_SquareSignal.flatten()

		elif not os.path.isfile(os.path.join(new_path, 'Swr_Ripple.h5')):
			print("Ripple file not Found. Starting Analysis from scratch...")	

			#Signal Preprocessing
			signal = bandPass_filt(lfp_unfiltered = lfp, low_cut = low_cut, high_cut = high_cut, sampling_freq = frequency)
				
			squared_signal = np.square(signal)     #Need to convert pandas dataframe to array before squaring
				
			ts = np.arange(len(squared_signal))/frequency
			ts = ts.reshape((len(ts),1))
			
			window = np.ones((int(windowLength),1))/windowLength       #Floor function returns a float value. Needs to be converted to int
			window = window.flatten()
			shift = int((len(window)-1)/2)    #Shift value for the signal
			#Now, we will apply Normalized Linear Transformation on the signal
			#Here:
			# y0 : Signal after transformation
			# zf : z-values of signal transformation
			zi0 = scipy.signal.lfilter_zi(window,1)
			y0, zf = scipy.signal.lfilter(window, 1, squared_signal, zi = zi0)
			y1, zf1 = scipy.signal.lfilter(window, 1, signal, zi = zi0) 
			#Picking up the desired range of values according to the shift values   
			normalized_SquareSignal = np.append(y0[shift+1:],zf[0:shift+1])
			normalized_signal = np.append(y1[shift+1:],zf1[0:shift+1])
			if not (sd):
				sd = np.std(normalized_SquareSignal)
			meanA = np.mean(normalized_SquareSignal)	
			A = normalized_SquareSignal.reshape((len(normalized_SquareSignal),1))
			normalized_SquareSignal = np.divide((A - np.tile(meanA, [len(A),1])),(np.tile(sd, [len(A),1])))

			######################################################l############################################################
			#Round1 : Detecting Ripple Periods by thresholding normalized signal
			print("\nRound 1 : Detecting Ripple Periods by thresholding.")
			normalized_SquareSignal = normalized_SquareSignal.flatten()
			thresholded = np.where(normalized_SquareSignal > low_thresFactor, 1,0)
			start = np.nonzero(np.where(np.diff(thresholded) > 0, 1, 0))
			stop = np.nonzero(np.where(np.diff(thresholded) < 0, 1, 0))
			if len(stop) == len(start)-1:
				start = start[0:]

			if len(stop)-1 == len(start):
				stop = stop[1:]

			firstPass = np.vstack((start,stop))
			firstPass = firstPass.transpose()
			
			#print(firstPass)	
			print("After detection by thresholding: " + str(len(firstPass)) + " events.")

			####################################################################################################################
			#Round 2 : Excluding ripples whose length < minRipLen
			print("\nRound 2 : Excluding ripples whose length is less than Minimum Ripple Length and greater than Maximum Ripple Length.")
			if firstPass.size != 0:
				stop = np.array(stop)     #Converting tuples to numpy array for vectorization
				start = np.array(start)   #Converting tuples to numpy array for vectorization
				l = (stop - start)
				
				minIx = np.nonzero(np.where(l[0] < minRipLen/1000*frequency, 1, 0))
				minIx = minIx[0]
				stop = np.delete(stop,minIx)
				start = np.delete(start, minIx)
				
				l1 = (stop - start)
				minIx = np.nonzero(np.where(l1 > maxRipLen/1000*frequency, 1, 0))
				minIx = minIx[0]
				stop = np.delete(stop, minIx)
				start = np.delete(start, minIx)

				firstPass = np.vstack((start, stop))
				firstPass = firstPass.transpose()
			if firstPass.size == 0:
				print("Detection by threshold failed!")
				return
			else:
				#print(firstPass)
				print("Detection after thresholding: " + str(len(firstPass)) + " events.")

			####################################################################################################################
			#Round 3 : Merging ripples if inter-ripple period is too short
			print("\nRound 3 : Merging ripples if Inter-Ripple period is too short")
			minInterRippleSamples = minInterRippleInterval/1000*frequency
			secondPass = np.array([])
			ripple = firstPass[0,:]
			for i in range(1,len(firstPass)):
				if (firstPass[i,0] - ripple[1]) < minInterRippleSamples:
					ripple = np.append(ripple[0], firstPass[i,1])
				else:
					if secondPass.size != 0:
						secondPass = np.vstack((secondPass, ripple))
						ripple = firstPass[i,:]
					elif secondPass.size == 0:
						secondPass = np.append(secondPass, ripple)
						ripple = firstPass[i,:]
			secondPass = np.vstack((secondPass, ripple))

			if secondPass.size == 0:
				print("Ripple Merge Failed!")
				ripple = np.array([])
				return
			else:
				#print(secondPass)
				print("After Ripple Merge: "+ str((secondPass.shape)[0]) + " events.")
			
			#####################################################################################################################
			#Round 4: Discard Ripples with a peak power < high_thresFactor....
			print("\nRound 4: Discard Ripples with a peak power < High Threshold Factor")
			thirdPass = np.array([])
			peakNormalized_Power = np.array([])

			for i in range(0,len(secondPass)):                                                                          #Find a way to vectorize this segment for better speed performance.
				maxValue = np.amax(normalized_SquareSignal[int(secondPass[i,0]):int(secondPass[i,1] + 1)])
				maxIndex = np.argmax(normalized_SquareSignal[int(secondPass[i,0]): int(secondPass[i,1] + 1)])
			
				if maxValue > high_thresFactor:
					if thirdPass.size == 0:
						thirdPass = np.append(thirdPass, secondPass[i,:])
					elif thirdPass.size != 0:
						thirdPass = np.vstack((thirdPass, secondPass[i,:]))	
					peakNormalized_Power = np.append(peakNormalized_Power, maxValue)

			if thirdPass.size == 0:
				print("Peak Thresholding has failed. Ripple Detection Aborted.")
				return
			else:
				if len(thirdPass) == 2:
					thirdPass = thirdPass.reshape(1,2)
				#print(thirdPass)
				print("After Peak Thresholding: "+str((thirdPass.shape)[0])+" events.")

			#######################################################################################################################
			#Round 5: Detect Negative Peak for each ripple
			print("\nDetecting Negative Peaks in each ripple....")
			peakPosition = np.zeros((len(thirdPass),1))
			for i in range(0, len(thirdPass)):
				minValue = np.min(signal[int(thirdPass[i,0]):int(thirdPass[i,1] + 1)])
				minIndex = np.argmin(signal[int(thirdPass[i,0]):int(thirdPass[i,1] + 1)])
				#print(minIndex)
				peakPosition[i] = minIndex + thirdPass[i,0] 
			print("Okay! Its done.")	
				

			#######################################################################################################################
			#Round 6: Selection of ripples by means of their average inter-peak intervals...
			print("\nCalculating average frequency of ripples by the means of their Average Inter-Peak Intervals.")	
			fqcy = np.zeros((len(thirdPass),1))
			for i in range(len(thirdPass)):
				peakIx = scipy.signal.find_peaks(x = -signal[int(thirdPass[i,0]):int(thirdPass[i,1] + 1)], distance = 4.0, threshold = 0.0)
				#print(peakIx[1]['right_thresholds'])
				peakIx = peakIx[0]	
				if (not (peakIx.size == 0)) and (peakIx.size != 1) :
					fqcy[i] = frequency/np.median(np.diff(peakIx))
				else:
					print("This is something weird now...")	 

			#######################################################################################################################
			#Final Round : Filtering using Noisy channel signal recorded outside the hippocampus 
			#noise_ch : Noise signal array
			#		  : Ripple-Band filtered signal used to exclude ripple-like noise

			ripples = np.array([thirdPass[:,0]/frequency, peakPosition.flatten()/frequency, thirdPass[:,1]/frequency, peakNormalized_Power, fqcy.flatten()])
			
			ripples = ripples.transpose()
			print("\n")
			#print(ripples)
			print("Ripples Detected Successfully !!")	

			if save_output == True and data_dir != None:
				rpt_ep = make_Epochs(ripples[:,0], ripples[:,2])
				writeNeuroscopeEvents(new_path + "Swr_Ripple.evt.rip", rpt_ep, "SWR Ripple event")	
				rpt_frame = pd.DataFrame(ripples, columns = ["start", "peak", "end", "Peak_Power", "Instantanous_Frequency"])
				rpt_frame.to_hdf(new_path + 'Swr_Ripple.h5', mode = 'w', key = 'positions')		
				print("Results saved  Successfully!!")
	
	########################
	#Plooting the results
	if plotting == True :
		ts = np.arange(len(signal))/frequency
		print("Loading Plots; Please Wait...")
		figure()
		subplot(3,1,1)
		subplot(3,1,1).set_title("Bandpassed LFP(" + str(low_cut) +" - "+str(high_cut)+" Hz)")
		plot(ts, signal)
		[axvline(_x, linewidth=1, color='g') for _x in ripples[:,0]]
		[axvline(_x, linewidth=1, color='r') for _x in ripples[:,2]]
		subplot(3,1,2)
		subplot(3,1,2).set_title("Squared Signal")
		plot(ts, squared_signal)
		[axvline(_x, linewidth=1, color='g') for _x in ripples[:,0]]
		[axvline(_x, linewidth=1, color='r') for _x in ripples[:,2]]
		subplot(3,1,3)
		subplot(3,1,3).set_title("Normalized Squared Signal")
		plot(ts, normalized_SquareSignal)
		[axvline(_x, linewidth=1, color='g') for _x in ripples[:,0]]
		[axvline(_x, linewidth=1, color='r') for _x in ripples[:,2]]
		show()
		
	return rpt_frame		

def concat_Positions(presleep, postsleep, wake, path):
	#Function to concatenate all these position files as one single HDF file for pandas
	#Parameters:-
	#presleep : path to file with presleep epoch head-direction positions
	#postsleep : path to file with postsleep epoch head-direction positions
	#wake : path to file with wakefulness epoch head-direction positions
	#path : path to the directory with all the data files

	pos_data0 = pd.read_csv(presleep, sep = '\t', names = ["timestamp", "x", "y", "angle"])
	pos_data1 = pd.read_csv(postsleep, sep = '\t', names = ["timestamp", "x", "y", "angle"])
	pos_data2 = pd.read_csv(wake, sep = '\t', names = ["timestamp", "x", "y", "angle"])
	import os
	if not os.path.exists(path):
		print("The path "+path+" doesn't exist; Exiting ...")
		sys.exit()    
	new_path = os.path.join(path, 'Analysis/')
	if os.path.exists(new_path):

		new_path    = os.path.join(path, 'Analysis/')
		fft = pd.concat([pos_data0, pos_data1, pos_data2], keys = {'presleep':pos_data0, 'postsleep': pos_data1, 'wake' : pos_data2})
		fft.to_hdf(new_path + 'Positions.h5', mode = 'w', key = 'positions')
	return None

def ripple_spiking_statisitics(ripple_events, data_directory,position_data, ep, bin_size): 
	#For inspecting the spiking profile for different SWR events..
	#Useful for inspecting the no of ripple events passing a specific statistical criterion
	from wrappers import loadSpikeData
	from functions import computeAngularTuningCurves, findHDCells
	spikes, shanks = loadSpikeData(data_directory)
	tuning_curves = (computeAngularTuningCurves(spikes, position_data, ep, 61))
	hd_idx, stats = findHDCells(tuning_curves)
	ripple_events = ripple_events.values
	spikes_hd = {k:spikes[k] for k in hd_idx}
	#print(spikes_hd)
	d = 1000000/2
	ripple_epochs = make_Epochs((ep['start'].values[0] + (ripple_events[:,1]*1000000) - d), (ep['start'].values[0] + (ripple_events[:,1]*1000000) + d))
	#print(ripple_epochs.as_units('ms'))
	val_array = []
	edges = []
	for i in range(len(ripple_epochs)):
		val = 0
		for k in hd_idx:
			spk = spikes_hd[k].restrict(ripple_epochs.iloc[[i]]).as_units('ms').index.values
			val_emp, edge = np.histogram(spk, np.arange(ripple_epochs.as_units('ms').iloc[[i]].start.values[0], ripple_epochs.as_units('ms').iloc[[i]].end.values[0], bin_size))
			print(len(np.nonzero(val_emp)))
			val = val + val_emp
		val_array.append(val)
		edges.append(edge)
		print(val_array)
	return val_array, edges

def calc_Angular_velocity(decoded_angles):
	#Calculate angular_velocity as a function of time...
	ang_vector = []
	for i in range(len(decoded_angles)):
		ang_vector.append(nts.TsdFrame((decoded_angles[i]).index.values[:-1],np.diff(decoded_angles[i]), time_units = 'us'))

	ts = []
	for i in range(len(ang_vector)):
		ts.append((ang_vector[i]).index.values)

	ang_vec = []
	for i in range(len(decoded_angles)):
		y1 = np.absolute(np.diff(decoded_angles[i]))
		ang_vec.append(nts.TsdFrame(np.arange(len(ts[i])), np.where(y1> np.pi, 2*np.pi - y1, y1)/np.diff(decoded_angles[i].index.values)))
	return ang_vec	

def discontinous_decoding_array(spikes, decoding_array, bin_size, neuron_order, t, tuning_curves, occupancy):
	#Computing a discontinuos decoding by eliminating the sections where there are no spikes 
	#t: ripple peak time
	#bin_size : 40 ms
	bins = np.arange(0, 2000+2*bin_size, bin_size) - 1000 - bin_size/2
	obins = np.vstack((bins-bin_size/2,bins)).T.flatten()
	obins = np.vstack((obins,obins+bin_size)).T
	times = obins[:,0]+(np.diff(obins)/2).flatten()	
	
	def histo(spk, obins):
		n = len(obins)
		count = np.zeros(n)
		for i in range(n):
			count[i] = np.sum((spk > obins[i,0]) * (spk < obins[i,1]))
		return count
	
	spike_counts = pd.DataFrame(index = times, columns = neuron_order)
	tbins = t + obins
	for k in neuron_order:
		spike_counts[k] = histo(spikes[k].as_units('ms').index.values, tbins)

	#print(spike_counts)	
	#return spike_counts

	dec, par = decoding_overlap(tuning_curves, bin_size, spikes_hd, hd_idx, rip[0], occupancy)
	dec = dec.as_units('ms')
	cnt = 0
    for i in range(len(spike_counts.index.values)):
    	if len(np.nonzero(spike_counts.iloc[[i]].values)[0]) == 0:
    		cnt += 1
    		dec.iloc[[i]] = np.nan

    return dec		

def make_pandas_matrix(ang_arr):
	ang_arr = calc_Angular_velocity(angle_decoding2)

	ang_matrix = np.array([])
	for i in range(len(ang_arr)):
		if i == 0:
			ang_matrix = np.append(ang_matrix, (ang_arr[i].values).T)
		else:
			ang_matrix = np.vstack((ang_matrix, (ang_arr[i].values).T))

	angular_frame = pd.DataFrame(ang_matrix)
	return angular_frame	

lfp_tot = loadLFP(lfp_dir)     # total length = 17681.2 s
#lfp_tot	= loadBunch_Of_LFP(lfp_dir,start = 0.0, stop = 1000.0, channel = 64)		#Function to convert LFP file into NTS Series Frame...Function(time in seconds)

#Note: Run this command if there is no Positions.h5 file present in Analysis folder.
'''
concat_Positions(r'D:\Mouse12-120808\Analysis\Mouse12-120808_pretraining_sleep.txt', r'D:\Mouse12-120808\Analysis\Mouse12-120808_posttraining_sleep.txt', 
																			r'D:\Mouse12-120808\Analysis\Mouse12-120808_wake.txt', r'D:\Mouse12-120808')
'''
#Loading the position data from the 'Positions.h5' file in Analysis folder.
presleep = load_Positions(data_directory, 'presleep')
#Since epochs loaded from the loadEpoch() function can be mismatched in some cases, this function will generate them from position file.
presleep_ep = correct_Epochs(presleep)  

postsleep = load_Positions(data_directory, 'postsleep')
postsleep_ep = correct_Epochs(postsleep)  
#First lets find rip-rip ripples using findRipples() function
#ripples1 = findRipples(lfp_tot.loc[int(presleep_ep['start'].values[0]) : int(presleep_ep['end'].values[0])], plotting = True, data_dir = data_directory, save_output = True)        #For the pre-training sleep period


#ripples2 = findRipples(lfp_tot.loc[int(postsleep_ep['start'].values[0]) : int(postsleep_ep['end'].values[0])], plotting = True, data_dir = data_directory, save_output = True)

#ripples1 = findRipples(plotting = False, data_dir = data_directory, save_output = True)
ripples2 = load_ripples(data_directory)
#ripple_event = ripples1.values
#ripple_epochs = make_Epochs((presleep_ep['start'].values[0] + (ripple_event[:,0]*1000000)), (presleep_ep['start'].values[0] + (ripple_event[:,2]*1000000)))



entropy_decoding2, angle_decoding2 = ripple_Decoding(ripples2, data_directory, 'postsleep', lfp_tot, save_pdf = False)

'''
feature_mat = entropy_matrix(entropy_decoding2)
fmat2 = feature_mat/np.log2(40)
max_val = np.max(feature_mat)
max_val = np.max(fmat2)
fmat3 = -fmat2 + max_val
'''

'''
l = max([len(i) for i in angle_decoding2])
angv = []
for i in angle_decoding2:
	if len(i) == l:
		angv.append(i)
	else:
		print("Ripple Discarded.")	
'''

ang_arr = calc_Angular_velocity(angle_decoding2)

ang_matrix = np.array([])
for i in range(len(ang_arr)):
	if i == 0:
		ang_matrix = np.append(ang_matrix, (ang_arr[i].values).T)
	else:
		ang_matrix = np.vstack((ang_matrix, (ang_arr[i].values).T))


angular_frame = pd.DataFrame(ang_matrix)
#angular_frame = 1000*angular_frame

'''
feature_mat1 = feature_mat - np.min(feature_mat)
feature_mat1 = feature_mat1/np.max(feature_mat1)
fmat = pd.DataFrame(feature_mat1,columns = [np.linspace(-500,500,40)])
m = fmat.max(1)
t = m.sort_values(0)
fmat = fmat.reindex(t.index.values)
fmat.index = np.arange(len(t.index.values))
'''
rip = ripples2.values
rip = rip[:,1]*1000 + postsleep_ep.as_units('ms')['start'].values[0]


sys.exit()
feature_matrix = Feature_Matrix(lfp_tot, ripples1, presleep_ep) 
print(feature_matrix.shape)
#sys.exit()
pca = PCA(n_components =3)
pca_map = pca.fit_transform(feature_matrix)

freq = np.delete(ripples1['Instantanous_Frequency'].values,[82,84,85,91,92,93,95,96,97])
plot_scatter_3d(pca_map, color_val = freq)

#findRipples will return a dataframe containing information about the SWRs. Feed it to ripple_Decoding function. Now sit back and relax.
#presleep = ripple_Decoding(ripples1, data_directory, 'postsleep', lfp_tot, save_pdf = True)


sys.exit()

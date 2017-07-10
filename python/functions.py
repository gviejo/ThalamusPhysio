import numpy as np


def gaussFilt(X, wdim = (1)):
	'''
		Gaussian Filtering in 1 or 2d.		
		Made to fit matlab
	'''
	from scipy.signal import gaussian
	from scipy.signal import convolve2d

	def conv2(x, y, mode='same'):
		return np.rot90(convolve2d(np.rot90(x, 2), np.rot90(y, 2), mode=mode), 2)	

	if len(wdim) == 1:
		N1 = wdim[0]
		print("TODO 1D gaussFilt")
		return X
	elif len(wdim) == 2:
		l1, l2 = X.shape
		N1, N2 = wdim		
		# create bordered matrix
		Xf = np.flipud(X)
		bordered_X = np.vstack([
				np.hstack([
					np.fliplr(Xf),Xf,np.fliplr(Xf)
				]),
				np.hstack([
					np.fliplr(X),X,np.fliplr(X)
				]),
				np.hstack([
					np.fliplr(Xf),Xf,np.fliplr(Xf)
				]),
			])
		# gaussian windows
		N1 = N1*10
		N2 = N2*10
		S1 = (N1-1)/float(2*5)
		S2 = (N2-1)/float(2*5)
		gw = np.vstack(gaussian(N1,S1))*gaussian(N2,S2)
		gw = gw/gw.sum()
		# convolution
		filtered_X = conv2(bordered_X, gw, mode ='same')
		return filtered_X[l1:l1+l1,l2:l2+l2]
	else :
		print("Error, dimensions larger than 2")
		return

def derivative(x, f):
	''' 
		Compute the derivative of a time serie
		Used for jPCA
	'''
	from scipy.stats import linregress
	fish = np.zeros(len(f))
	slopes_ = []
	tmpf = np.hstack((f[-1],f,f[0:3]))
	binsize = x[1]-x[0]	
	tmpx = np.hstack((np.array([x[0]-binsize]),x,np.array([x[-1]+i*binsize for i in range(1,4)])))	
	# plot(x, f, 'o')
	for i in range(len(f)):
		slope, intercept, r_value, p_value, std_err = linregress(tmpx[i:i+3], tmpf[i:i+3])
		slopes_.append(slope)	
		# plot(tmpx[i:i+3], tmpx[i:i+3]*slope+intercept, '-')
	return np.array(slopes_)/binsize

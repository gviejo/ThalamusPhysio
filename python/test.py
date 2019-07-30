'''
========================================================
Demonstration of advanced quiver and quiverkey functions
========================================================

Known problem: the plot autoscaling does not take into account
the arrows, so those on the boundaries are often out of the picture.
This is *not* an easy problem to solve in a perfectly general way.
The workaround is to manually expand the axes.
'''
import matplotlib.pyplot as plt
import numpy as np
from numpy import ma

X, Y = np.meshgrid(np.arange(0, 2 * np.pi, .2), np.arange(0, 2 * np.pi, .1))
# X = np.arange(0, 2*np.pi, 0.1)
# Y = np.arange(0, 2*np.pi, 0.2)
U = np.ones(X.shape)
V = np.ones(Y.shape)*0.0
# U = np.cos(X)
# V = np.sin(Y)

plt.figure()
plt.title('Arrows scale with plot width, not view')
Q = plt.quiver(X, Y, U, V, units='width')
# qk = plt.quiverkey(Q, 0.9, 0.9, 2, r'$2 \frac{m}{s}$', labelpos='E',
#                    coordinates='figure')

# plt.figure()
# plt.title("pivot='mid'; every third arrow; units='inches'")
# Q = plt.quiver(X[::3, ::3], Y[::3, ::3], U[::3, ::3], V[::3, ::3],
#                pivot='mid', units='inches')
# # qk = plt.quiverkey(Q, 0.9, 0.9, 1, r'$1 \frac{m}{s}$', labelpos='E',
# #                    coordinates='figure')
# # plt.scatter(X[::3, ::3], Y[::3, ::3], color='r', s=5)

# plt.figure()
# plt.title("pivot='tip'; scales with x view")
# M = np.hypot(U, V)
# Q = plt.quiver(X, Y, U, V, M, units='x', pivot='tip', width=0.022,
#                scale=1 / 0.15)
# # qk = plt.quiverkey(Q, 0.9, 0.9, 1, r'$1 \frac{m}{s}$', labelpos='E',
# #                    coordinates='figure')
# # plt.scatter(X, Y, color='k', s=5)

plt.show()

# from pylab import *
# import matplotlib.cm as cm
# import matplotlib.gridspec as gridspec
# from matplotlib.pyplot import *
# from mpl_toolkits.axes_grid.inset_locator import inset_axes

# figure()

# gs = gridspec.GridSpec(6, 3)

# ax1 = plt.subplot(gs[0:2,0])

# ax2 = plt.subplot(gs[0, 1])
# ax3 = plt.subplot(gs[1, 1])

# ax4 = plt.subplot(gs[0:2,2])

# ax5 = plt.subplot(gs[2:,0:2])

# ax6 = plt.subplot(gs[2:4, 2])
# ax7 = plt.subplot(gs[4:6, 2])

# show()
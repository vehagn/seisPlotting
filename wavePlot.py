#!/usr/bin/env python
import rsf.api as rsf
import numpy as np
import pylab as pl
import sys as sys
import os
from matplotlib.colors import LinearSegmentedColormap

baseFolder  = '/home/vegahag/msim-ext/AGU2017/gullfaks/'

dataFolder  = baseFolder + 'vp_8Hz-zdir-long/'
modelFolder = baseFolder + 'model/'
saveFolder  = baseFolder + 'vp_8Hz-zdir-long/img/'

model = 'vp'
fileName = 'szz-pforw+0001'
#fileName = 'szz-pback+0001'

if not os.path.exists(saveFolder):
    os.makedirs(saveFolder)

fin   = rsf.Input(dataFolder+fileName+'.rsf')
[t,n,m] = fin.shape()

data  = np.zeros([t,n,m],'f')
fin.read(data)

data = np.array(data.transpose())
data = data/np.max(np.abs(data)) # Normalise data

pl.rc('text', usetex=True)

dt = 0.00008*120 # [s]

fin   = rsf.Input(modelFolder+model+'.rsf')
model = np.zeros([n,m],'f')
fin.read(model)
model = np.array(model.transpose())

cdict = {'red':  ((0.0, 0.0, 0.0),
                   (1.0, 1.0, 1.0)),

         'green': ((0.0, 0.0, 0.0),
                   (1.0, 1.0, 1.0)),

         'blue':  ((0.0, 0.0, 0.0),
                   (1.0, 1.0, 1.0)),

         'alpha': ((0.0, 1.0, 1.0),
                   (0.5, 0.0, 0.0),
                   (1.0, 1.0, 1.0))
        }

grayAlpha = LinearSegmentedColormap('gAlpha', cdict)
it = 0
for i in range(0,t,5):
#for i in range(t-1,-1,-5):
    print 'Frame {} of {}'.format(i,t)
#cLim = pow(1./(1 + i),0.7)
    cLim = 0.3
    fig = pl.figure()
#    img = pl.imshow(data[:,:,i],interpolation='nearest')
    fig.set_size_inches(10,3)
    ax  = pl.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)
    mod = ax.imshow(model)
#    cbar = fig.colorbar(mod,
#            ax=ax,
#            orientation='horizontal',
#            shrink=0.8,
#            pad=0.01,
#            aspect=18.0)
#    ax.plot([200,600], [250,250], color='c', lw=50, alpha=0.3)
#    ax.xaxis.tick_top()
#    ax.xaxis.set_label_position('top')
    img = ax.imshow(data[:,:,i],interpolation='nearest', cmap=grayAlpha)
    img.set_clim([-cLim,cLim])
    #ax.annotate('$P$',xy=(5,30),fontsize=30, color='white')
    time = '{: 6.1f} ms'.format(1000*dt*(i))
#time = '{: 6.1f} ms'.format(1000*dt*(t-(i+1)))
    ax.annotate(time, xy=(995,295), fontsize=15, color='teal',horizontalalignment='right')
    pl.savefig(saveFolder+fileName+'_'+'{0:04d}'.format(it)+'.png')#,bbox_inches='tight')
    pl.close()
    it += 1
    #pl.show()

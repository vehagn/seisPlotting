#!/usr/bin/env python
import abp
import numpy as np
import pylab as pl
import sys as sys
import os
from matplotlib.colors import ListedColormap

nx   = 501;
ny   = 150;
npml =  20;

dx = 20.0
dy = 20.0
dt = 0.001# [s]

fPclip = 0.3

fieldType = 'syy'
timeStep = 3000

model = 'Gull0km/'
project = '/home/vegahag/msim-ext/SEG2018/'+model
save = 'SEG2018/'+model

folder = project
modelFolder = project

if not os.path.exists(save):
    os.makedirs(save)

rh = abp.read_data_2d(modelFolder+"rh-true.bin",nx,ny);
vp = abp.read_data_2d(modelFolder+"vp-true.bin",nx,ny);
vs = abp.read_data_2d(modelFolder+"vs-true.bin",nx,ny);

model = vp

field = abp.read_data_2d(modelFolder+fieldType+'-snaps-'+str(timeStep)+'.bin',nx+2*npml, ny+2*npml)
field = field[npml:-npml,npml:-npml]

fldAbsMax = abs(field).max()

# Choose colormap
cmapBase = pl.cm.gray
cmapAlpha = cmapBase(np.arange(cmapBase.N))
cmapAlpha[:,-1] = np.block([np.linspace(1, 0, cmapBase.N/2),  np.linspace(0,1, cmapBase.N/2)])
cmapAlpha = ListedColormap(cmapAlpha)

pl.rc('text', usetex=True)

fig = pl.figure()
ax = pl.Axes(fig, [0.,0.,1.,1.])
ax.set_axis_off()
fig.add_axes(ax)

mod = ax.imshow(model, cmap='viridis', interpolation='none', extent=[0,nx*dx,ny*dy,0], aspect=1)
img = ax.imshow(field, cmap=cmapAlpha, interpolation='none', extent=[0,nx*dx,ny*dy,0], aspect=1)
img.set_clim([-fPclip*fldAbsMax,fPclip*fldAbsMax])
#ax.annotate('$P$',xy=(5,30),fontsize=30, color='white')
time = '{: 6.1f} s'.format(dt*timeStep)
#time = '{: 6.1f} ms'.format(1000*dt*(t-(i+1)))
ax.annotate(time, xy=(9900,2900), fontsize=15, color='teal',horizontalalignment='right')

pl.savefig(save+fieldType+'-'+str(timeStep)+'.pdf',bbox_inches='tight')

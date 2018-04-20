#!/usr/bin/env python
import abp
import numpy as np
import pylab as pl
import sys as sysi
import os
#import matplotlib.pyplot as mpl

# Plot the parameter models including differences

nx = 501;
ny = 150;

dx = 20.0;
dy = 20.0;

def normalize(x):
    if x.max() != x.min():
        return (x - x.min())/(x.max()-x.min())
    else:
        return (x - x.min())

prefix = 'Gradient '

model = 'Grad5km/'
project = '/home/vegahag/msim-ext/SEG2018/'+model
save = 'SEG2018/'+model

folder = project
modelFolder = project

if not os.path.exists(save):
    os.makedirs(save)

rh = abp.read_data_2d(modelFolder+"rh-true.bin",nx,ny);
vp = abp.read_data_2d(modelFolder+"vp-true.bin",nx,ny);
vs = abp.read_data_2d(modelFolder+"vs-true.bin",nx,ny);

model = np.zeros([ny,nx,3],'f')
model[:,:,0] = rh
model[:,:,1] = vp
model[:,:,2] = vs

param = [
         #'rho-bg','vp-bg','vs-bg',
         'rh-true','vp-true','vs-true',
         #'rho-diff','vp-diff'
         ]

title = [
         #'$\mathbf{\\rho}$ background','$\mathbf{v_p}$ background','$\mathbf{v_s}$ background',
         '$\mathbf{\\rho}$ model [kg/m\\textsuperscript{3}]',
         '$\mathbf{v_p}$ model [m/s]',
         '$\mathbf{v_s}$ model [m/s]',
         #'$\mathbf{\\rho}$ difference','$\mathbf{v_p}$ difference'
         ]

nParam = len(param)

pl.rc('text', usetex=True)
# Plot
for i in range(0,nParam):
    fig = pl.figure()
    image = pl.imshow(model[:,:,i], extent=[0,nx*dx,ny*dy,0], interpolation='none')
    ax = pl.gca()

    image.set_cmap('viridis')
    ax.xaxis.tick_top()
    ax.xaxis.set_label_position('top')

    pl.xlabel("$x$ (m)")
    pl.ylabel("$y$ (m)")

    # Colorbar
    cb = pl.colorbar(image,
            ax=ax,
            orientation='horizontal',
 #           ticks=([-1, -0.5, 0, 0.5, 1]),
            #ticks=(),
            shrink=1,
            pad=0.01,
            aspect=20.0)
    cb.set_label(prefix+title[i])

    # Markers
    pl.scatter(5000, 80, marker='*', color='xkcd:red', s=30, zorder=1)
    pl.scatter(  80, 80, marker='*', color='xkcd:yellow', s=30, zorder=1)
    pl.plot([5000,5000], [500,3000], color='w', linewidth=2, zorder=1,alpha=0.5)

    pl.xlim([0,nx*dx]);
    pl.ylim([ny*dy,0]);

    pl.savefig(save+param[i]+'.pdf',bbox_inches='tight')
    #pl.show()

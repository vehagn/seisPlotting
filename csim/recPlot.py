#!/usr/bin/env python
import abp
import numpy as np
import pylab as pl
import sys as sysi
import os
#import matplotlib.pyplot as mpl

# Plot the parameter models including differences

nx =   501;
nt = 10000;

dx = 20.0;
dt = 0.001;

pclip = .1;
tpow  = 1.8

model = 'Gull0km/'
project = '/home/vegahag/msim-ext/SEG2018/'+model
save = 'SEG2018/'+model

folder = project
modelFolder = project

if not os.path.exists(save):
    os.makedirs(save)

tru = abp.read_data_2d(modelFolder+"data-true.bin",nt,nx);
mod = abp.read_data_2d(modelFolder+"data-mod.bin",nt,nx);
res = abp.read_data_2d(modelFolder+"data-res.bin",nt,nx);

rec = np.zeros([nt,nx,3],'f')
rec[:,:,0] = tru.transpose()
rec[:,:,1] = mod.transpose()
rec[:,:,2] = res.transpose()

param = [
         'rec-tru','rec-mod','rec-res',
         ]

title = [
         'True',
         'Backgound',
         'Residual',
         ]

nParam = len(param)

# tpow scale
for t in range(0,nt):
    rec[t,:,:] = rec[t,:,:]*(t*dt)**(tpow)


pl.rc('text', usetex=True)
# Plot
for i in range(0,nParam):
    absmax = abs(rec[:,:,i]).max()

    print absmax
    fig = pl.subplot(1,3,i+1)
    #fig = pl.figure()
    image = pl.imshow(rec[:,:,i],
                vmin=-pclip*absmax,
                vmax=+pclip*absmax,
                extent=[0,nx*dx,nt*dt,0],
                aspect=(2*9*nx*dx)/(16*nt*dt),
                interpolation='none'
                )

    ax = pl.gca()

    image.set_cmap('seismic')
    ax.xaxis.tick_top()
    ax.xaxis.set_label_position('top')

    pl.xlabel("\Large $x$ (m)")
    if (i==0):
        pl.ylabel("\Large $t$ (s)")

    # Hacking colorbar to get title below
    cb = pl.colorbar(image, ax=ax, orientation='horizontal', ticks=(), shrink=.0001, pad=0.01, aspect=1.0)
    cb.outline.set_visible(False)
    cb.set_label('\Large '+title[i])

    # Markers
    #pl.scatter(5000, 1500, marker='*', color='r', s=30)
    #pl.plot([0,10000], [10,10], color='r', linewidth=3, zorder=1)

pl.savefig(save+'rec.pdf',bbox_inches='tight')
    #pl.show()

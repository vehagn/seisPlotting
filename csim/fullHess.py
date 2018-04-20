#!/usr/bin/env python
import abp
import numpy as np
import pylab as pl
import sys as sysi
import os
from matplotlib.colors import ListedColormap
# Plot the parameter models including differences

nx = 501;
ny = 150;

dx = 20.0;
dy = 20.0;

par  = 'vp'
pert = 'vp'
pos  = 125

vLim = 0.03

def normalize(x):
    if x.max() != x.min():
        return (x - x.min())/(x.max()-x.min())
    else:
        return (x - x.min())

prefix = 'Hessian '

model = 'Gull5km/'
project = '/home/vegahag/msim-ext/SEG2018/maur/'+model
save = 'SEG2018/'+model

folder = project
modelFolder = project

if not os.path.exists(save):
    os.makedirs(save)

H1 = abp.read_data_2d(modelFolder+"H1-"+par+"-d"+pert+"-"+'{0:03.0f}'.format(pos)+".bin",nx,ny);
H2 = abp.read_data_2d(modelFolder+"H2-"+par+"-d"+pert+"-"+'{0:03.0f}'.format(pos)+".bin",nx,ny);
H3 = abp.read_data_2d(modelFolder+"H3-"+par+"-d"+pert+"-"+'{0:03.0f}'.format(pos)+".bin",nx,ny);
vp = abp.read_data_2d(modelFolder+"vp-true.bin",nx,ny);

Hess = np.zeros([ny,nx,4],'f')
Hess[:,:,0] = H1
Hess[:,:,1] = H2
Hess[:,:,2] = H3
Hess[:,:,3] = H1 + H2 + H3

param = [
         'H1','H2','H3','H'
         ]

title = [
         #'$\mathbf{\\rho}$ background','$\mathbf{v_p}$ background','$\mathbf{v_s}$ background',
         '$\mathbf{H1}$',
         '$\mathbf{H2}$',
         '$\mathbf{H3}$',
         '$\mathbf{H}$',
         #'$\mathbf{\\rho}$ difference','$\mathbf{v_p}$ difference'
         ]

nParam = len(param)

maxAbs = abs(Hess).max()
Hess = Hess/maxAbs

# Choose colormap
cmapBase = pl.cm.RdBu

cmapAlpha = cmapBase(np.arange(cmapBase.N))
cmapAlpha[:,-1] = np.block([np.linspace(1, 0, cmapBase.N/2),  np.linspace(0,1, cmapBase.N/2)])
cmapAlpha = ListedColormap(cmapAlpha)

pl.rc('text', usetex=True)
# Plot
for i in range(0,nParam):
    fig = pl.figure()

    #Hess[:,:,i] = Hess[:,:,i]).max()

    mod = pl.imshow(vp, cmap='gray', extent=[0,nx*dx,ny*dy,0], interpolation='none',alpha=1)
    img = pl.imshow(Hess[:,:,i], cmap=cmapAlpha, extent=[0,nx*dx,ny*dy,0], interpolation='none',vmin=-vLim, vmax=vLim)
    ax = pl.gca()

    #img.set_cmap('RdBu')
    ax.xaxis.tick_top()
    ax.xaxis.set_label_position('top')

    pl.xlabel("$x$ (m)")
    pl.ylabel("$y$ (m)")

    # Colorbar
    cb = pl.colorbar(img,
            ax=ax,
            orientation='horizontal',
 #           ticks=([-1, -0.5, 0, 0.5, 1]),
            #ticks=(),
            shrink=1,
            pad=0.01,
            aspect=20.0)
    cb.set_label(prefix+title[i])

    # Markers
    #pl.scatter(5000, 80, marker='*', color='xkcd:red', s=30, zorder=1)
    #pl.scatter(  80, 80, marker='*', color='xkcd:yellow', s=30, zorder=1)
    #pl.plot([5000,5000], [500,3000], color='w', linewidth=2, zorder=1,alpha=0.5)

    pl.xlim([0,nx*dx]);
    pl.ylim([ny*dy,0]);
    pl.savefig(save+param[i]+'-'+par+'-d'+pert+'-'+'{0:03.0f}'.format(pos*dy)+'m.pdf',bbox_inches='tight')
    #pl.show()

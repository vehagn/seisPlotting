#!/usr/bin/env python
import abp
import numpy as np
import pylab as pl
import os
#import matplotlib.pyplot as mpl

nx = 501;
ny = 150;

dx = 20.0;

#step = 10;

def normalize(x):
    if x.max() != x.min():
        return (x - x.min())/(x.max()-x.min())
    else:
        return (x)

model = 'Gull0km/'

cLim = 1;
vLim = 0.03;

project = '/home/vegahag/msim-ext/SEG2018/maur/'+model
save = 'SEG2018/'+model

xStart = nx/2-1;
xStop  = xStart+1;

yStart =  24  #ny/2-step-1;
yStop  = 149  #ny/2+step;

folder = project
modelFolder = project

if not os.path.exists(save):
    os.makedirs(save)

#rh = abp.read_data_2d(modelFolder+"model-rh-true.bin",nx,ny);
#vp = abp.read_data_2d(modelFolder+"model-vp-true.bin",nx,ny);
#vs = abp.read_data_2d(modelFolder+"model-vs-true.bin",nx,ny);
#
#rhNorm = normalize(rh)
#vpNorm = normalize(vp)
#vsNorm = normalize(vs)
#
#model = np.zeros([nx,ny,3],'f')
#model[:,:,0] = rh
#model[:,:,1] = vp
#model[:,:,2] = vs

kernel = ['H1','H2','H3','H12','H123']
param  = ['rh','vp','vs']
perts  = ['rh','vp','vs']
title  = ['$\mathbf{H_1}$','$\mathbf{H_2}$','$\mathbf{H_3}$','$\mathbf{H_1}+\mathbf{H_2}$','$\mathbf{H}$']

# plus one to get the correct indices
yPos = np.arange(yStart,yStop)
xPos = np.arange(xStart,xStop)
#500-1

nParam = len(param)
nPerts = len(perts)
nPos   = len(yPos)
mPos   = len(xPos)
nKern  = len(kernel)
nLen   = np.max([nPos,mPos])
hessian = np.zeros([nParam*nPos,nParam*nPos],'f')

Hess    = np.zeros([nParam*nLen,nParam*nLen,nKern],'f')

print 'Starting to read data.'
# Read in data
for k in range(0,nKern-2):
    for n in range(0,nPos):
        for m in range(0,mPos):
            for i in range(0,nParam):
                for j in range(0,nPerts):
                    # Read data
                    # PARAM AND PERT SWITCHED!!
                    toRead = '{0}-{1}-d{2}-{3:03d}.bin'.format(kernel[k],param[j],perts[i],yPos[n])

                    temp = abp.read_data_2d(folder+toRead,nx,ny)

                    if nPos > mPos:
                        Hess[j*nLen:(j+1)*nLen, i*nLen+n,k] = temp[yPos,xPos]
                    else:
                        Hess[j*nLen:(j+1)*nLen, i*nLen+m,k] = temp[yPos,xPos]


## Remove diagonals
#    if k != 3 and k != 4:
#        for i in range(0,nParam*nLen):
#            Hess[i,i,k] = 0.0
#            if  (i+nLen < nParam*nLen):
#                Hess[i+nLen,i,k] = 0.0
#                Hess[i,i+nLen,k] = 0.0
#            if (i+2*nLen < nParam*nLen):
#                Hess[i+2*nLen,i,k] = 0.0
#                Hess[i,i+2*nLen,k] = 0.0
#
## Average diagonals
#    if k != 3 and k != 4:
#        for i in range(1,nParam*nPos):
#            Hess[i,i,k] = 0.5*(Hess[i-1,i,k] + Hess[i,i-1,k])
#            if  (i+nPos < nParam*nPos):
#                Hess[i+nPos,i,k] = 0.5*(Hess[i+nPos-1,i,k]+Hess[i+nPos,i-1,k])
#                Hess[i,i+nPos,k] = 0.5*(Hess[i-1,i+nPos,k]+Hess[i,i+nPos-1,k])
#            if (i+2*nPos < nParam*nPos):
#                Hess[i+2*nPos,i,k] = 0.5*(Hess[i+2*nPos-1,i,k]+Hess[i+2*nPos,i-1,k])
#                Hess[i,i+2*nPos,k] = 0.5*(Hess[i-1,i+2*nPos,k]+Hess[i,i+2*nPos-1,k])
#

# Calculate H12
Hess[:,:,3] = Hess[:,:,0] + Hess[:,:,1]
# Calculate H123
Hess[:,:,4] = Hess[:,:,2] + Hess[:,:,3]

#for i in range(0,nKern):
#    print [np.max(Hess[:,:,i]), np.min(Hess[:,:,i])]

# Plot
pl.rc('text', usetex=True)

for k in [ 0, 1, 2, 3, 4]:

    maxAbs = np.max(np.abs(Hess[:,:,k]))
    print maxAbs

    Hess[:,:,k] = Hess[:,:,k]/maxAbs

#    for i in range(0,nParam):
#        Hess[i*nLen:(i+1)*nLen,:,k] = sc*Hess[i*nLen:(i+1)*nLen,:,k]/(np.max(np.abs(Hess[i*nLen:(i+1)*nLen,:,k]))+1e-12)

    fig = pl.figure(figsize=(5,5))
    image = pl.imshow(Hess[:,:,k],interpolation='none',vmin=-vLim,vmax=vLim)
    ax = pl.gca()

    ax.plot([-1       ,nParam*nLen  ],[nLen-1   ,nLen-1]       ,'k',linewidth='0.5')
    ax.plot([-1       ,nParam*nLen-1],[2*nLen-1 ,2*nLen-1]     ,'k',linewidth='0.5')

    ax.plot([nLen-.5  ,nLen-.5      ],[-1       ,nParam*nLen+1],'k',linewidth='0.5')
    ax.plot([2*nLen+.5,2*nLen       ],[-1       ,nParam*nLen+1],'k',linewidth='0.5')

    pl.xlim([-.5,nParam*nLen-.5])
    pl.ylim([nParam*nLen-.45,-.5])

    image.set_cmap('RdBu')
#    image.set_clim([-cLim,cLim])
    ax.xaxis.tick_top()
    ax.xaxis.set_label_position('top')
    ax.yaxis.tick_right()
    ax.yaxis.set_label_position('right')

    ax.xaxis.set_ticks([])

    ax.yaxis.set_ticks([])
    #ax.yaxis.set_ticks([-0.5, nLen/2-0.5, nLen-0.5,3*nLen/2-0.5,2*nLen-0.5, 5*nLen/2-0.5, 3*nLen-0.5])
    #ax.yaxis.set_ticklabels([' ','10 m','', '20  m','','30 m',''],rotation=-90,va='center')

    ax.text(-nLen/2, nLen/2+1,   '\Large $\mathbf{H^\\rho}$')
    ax.text(-nLen/2, 3*nLen/2+1, '\Large $\mathbf{H^{v_p}}$')
    ax.text(-nLen/2, 5*nLen/2+1, '\Large $\mathbf{H^{v_s}}$')

    ax.text(nLen/2  -2, -5, '\Large $\mathbf{\\delta\\rho}$')
    ax.text(3*nLen/2-2, -5, '\Large $\mathbf{\\delta  v_p}$')
    ax.text(5*nLen/2-2, -5, '\Large $\mathbf{\\delta  v_s}$')

    ax.text(3*nLen, 1*nLen/2+1, '\Large '+str(int((yStop-yStart)*dx))+' m',rotation=-90,va='center')
    ax.text(3*nLen, 3*nLen/2+1, '\Large '+str(int((yStop-yStart)*dx))+' m',rotation=-90,va='center')
    ax.text(3*nLen, 5*nLen/2+1, '\Large '+str(int((yStop-yStart)*dx))+' m',rotation=-90,va='center')

    # Colorbar
    cb = pl.colorbar(image,
            ax=ax,
            orientation='horizontal',
            ticks=([-vLim, -vLim/2, 0, vLim/2, vLim]),
            #ticks=(),
            shrink=0.825,
            pad=0.02,
            aspect=20.0)
    cb.set_label(title[k])

    pl.savefig(save+kernel[k]+'.pdf',bbox_inches='tight')
    #pl.show()

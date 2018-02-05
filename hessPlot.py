#!/usr/bin/env python
import rsf.api as rsf
import numpy as np
import pylab as pl
import os
#import matplotlib.pyplot as mpl

def normalize(x):
    if x.max() != x.min():
        return (x - x.min())/(x.max()-x.min())
    else:
        return (x)

model = 'barn/homo/'
pDir  = 'z'
sLen  = 'short'


project = '/home/vegahag/msim-ext/AGU2017/'+model

part = 'vp_8Hz-'+pDir+'dir-'+sLen
save = 'AGU2017/'+model
freq = ''


if pDir == 'z':
    zStart =  50 #159
    zStop  =  140 #219
    xStart = 350 #500
    xStop  = xStart
elif pDir == 'x':
    zStart = 190
    zStop  = zStart
    xStart = 469
    xStop  = 529


folder = project+part+freq+'/'
modelFolder = project + 'model/'

if not os.path.exists(save+part+freq+'/'):
    os.makedirs(save+part+freq+'/')

dz = 10
dx = dz

fin   = rsf.Input(modelFolder+'rh.rsf')
[n,m] = fin.shape()
data  = np.zeros([n,m],'f')
fin.read(data)
rho   = np.array(data.transpose())
rhoNorm = normalize(rho)

fin   = rsf.Input(modelFolder+'vp.rsf')
fin.read(data)
vp    = np.array(data.transpose())
vpNorm = normalize(vp)

fin   = rsf.Input(modelFolder+'vs.rsf')
fin.read(data)
vs    = np.array(data.transpose())
vsNorm = normalize(vs)

model = np.zeros([m,n,3],'f')
model[:,:,0] = rho
model[:,:,1] = vp
model[:,:,2] = vs

kernel = ['H1','H2','H3','H12','H123']
param  = ['rh','vp','vs']
perts  = ['rh','vp','vs']
title  = ['$\mathbf{H^1}$','$\mathbf{H^2}$','$\mathbf{H^3}$','$\mathbf{H^1}+\mathbf{H^2}$','$\mathbf{H^1}+\mathbf{H^2}+\mathbf{H^3}$']

# plus one to get the correct indices
zPos = np.arange(zStart,zStop+1)
xPos = np.arange(xStart,xStop+1)
#500-1

nParam = len(param)
nPerts = len(perts)
nPos   = len(zPos)
mPos   = len(xPos)
nKern  = len(kernel)
nLen   = np.max([nPos,mPos])
hessian = np.zeros([nParam*nPos,nParam*nPos],'f')
temp    = np.zeros([n,m],'f')
#Hess    = np.zeros([nParam*nPos,nParam*nPos,nKern],'f')

Hess    = np.zeros([nParam*nLen,nParam*nLen,nKern],'f')

# Read in data
for k in range(0,nKern-2):
    for n in range(0,nPos):
        for m in range(0,mPos):
            for i in range(0,nParam):
                for j in range(0,nPerts):
                    # Read data
                    toRead = 'kernels/{0}-{1}-delta_{2}-z{3:04d}-x{4:04d}.rsf'.format(kernel[k],param[i],perts[j],zPos[n],xPos[m])
                    #print toRead
                    fin = rsf.Input(folder+toRead)
                    fin.read(data)
                    # Use temp array since reading data is fucky
                    temp = np.array(data.transpose())

                    #temp = temp/model[:,:,i]
                    if nPos > mPos:
                        #Hess[j*(nPos):(j+1)*(nPos), i*(nPos)+n,k] = temp[zPos-1,xPos-1]
                        Hess[j*nLen:(j+1)*nLen, i*nLen+n,k] = temp[zPos-1,xPos-1]
                    else:
                        Hess[j*nLen:(j+1)*nLen, i*nLen+m,k] = temp[zPos-1,xPos-1]

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
    # Scale by row
    sc = 10
    print np.max(np.abs(Hess[i*nLen:(i+1)*nLen,:,k]))
    for i in range(0,nParam):
        Hess[i*nLen:(i+1)*nLen,:,k] = sc*Hess[i*nLen:(i+1)*nLen,:,k]/(np.max(np.abs(Hess[i*nLen:(i+1)*nLen,:,k]))+1e-12)

    fig = pl.figure(figsize=(5,5))
    image = pl.imshow(Hess[:,:,k],interpolation='none')
    ax = pl.gca()

    ax.plot([-1,nParam*nLen+1],[nLen-.5,nLen-.5],'k',linewidth='0.5')
    ax.plot([-1,nParam*nLen+1],[2*nLen-.5,2*nLen-.5],'k',linewidth='0.5')
    ax.plot([nLen-.5,nLen-.5],[-1,nParam*nLen+1],'k',linewidth='0.5')
    ax.plot([2*nLen-.5,2*nLen-.5],[-1,nParam*nLen+1],'k',linewidth='0.5')

    pl.xlim([-.5,nParam*nLen-.5])
    pl.ylim([nParam*nLen-.45,-.5])

    image.set_cmap('seismic')
    image.set_clim([-1,1])
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

    # Colorbar
    cb = pl.colorbar(image,
            ax=ax,
            orientation='horizontal',
            ticks=([-1, -0.5, 0, 0.5, 1]),
            #ticks=(),
            shrink=0.825,
            pad=0.02,
            aspect=20.0)
    cb.set_label(title[k])

    pl.savefig(save+part+freq+'/'+kernel[k]+'.pdf',bbox_inches='tight')
    #pl.show()

print part+freq

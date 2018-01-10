#!/usr/bin/env python
import rsf.api as rsf
import numpy as np
import pylab as pl
import sys as sysi
import os
#import matplotlib.pyplot as mpl

# Plot the parameter models including differences

def normalize(x):
    if x.max() != x.min():
        return (x - x.min())/(x.max()-x.min())
    else:
        return (x - x.min())

project = '/home/vegahag/msim-ext/AGU2017/'

part = 'gullfaks/model'

freq = ''

folder = project+part+freq+'/'

if not os.path.exists('AGU2017/'+part+'/'):
    os.makedirs('AGU2017/'+part+'/')

dz = 10
dx = 10

fin   = rsf.Input(folder+'rh.rsf')
[n,m] = fin.shape()
data  = np.zeros([n,m],'f')
fin.read(data)
rho   = np.array(data.transpose())

fin   = rsf.Input(folder+'vp.rsf')
fin.read(data)
vp    = np.array(data.transpose())

fin   = rsf.Input(folder+'vs.rsf')
fin.read(data)
vs    = np.array(data.transpose())

fin   = rsf.Input(folder+'rh-incl.rsf')
fin.read(data)
rho_true   = np.array(data.transpose())

fin   = rsf.Input(folder+'vp-incl.rsf')
fin.read(data)
vp_true    = np.array(data.transpose())
vpNorm = normalize(vp)

fin   = rsf.Input(folder+'vs-incl.rsf')
fin.read(data)
vs_true    = np.array(data.transpose())

fin   = rsf.Input(folder+'incl0.rsf')
fin.read(data)
rho_diff    = np.array(data.transpose())

fin   = rsf.Input(folder+'incl1.rsf')
fin.read(data)
vp_diff    = np.array(data.transpose())

model = np.zeros([m,n,8],'f')
model[:,:,0] = rho
model[:,:,1] = vp
model[:,:,2] = vs
model[:,:,3] = rho_true
model[:,:,4] = vp_true
model[:,:,5] = vs_true
model[:,:,6] = rho_diff
model[:,:,7] = vp_diff

param = ['rho-bg','vp-bg','vs-bg',
        'rho-true','vp-true','vs-true',
        'rho-diff','vp-diff']

title = ['$\mathbf{\\rho}$ background','$\mathbf{v_p}$ background','$\mathbf{v_s}$ background',
         '$\mathbf{\\rho}$ model','$\mathbf{v_p}$ model','$\mathbf{v_s}$ model',
         '$\mathbf{\\rho}$ difference','$\mathbf{v_p}$ difference']


# plus one to get the correct indices

nParam = len(param)

pl.rc('text', usetex=True)
# Plot
for i in range(0,nParam):
    # Axis
    o1 = fin.float("o1")
    o2 = fin.float("o2")
    d1 = fin.float("d1")
    d2 = fin.float("d2")
    n1 = fin.int("n1")
    n2 = fin.int("n2")


    fig = pl.figure(figsize=(5,5))
    image = pl.imshow(model[:,:,i],extent=[o2,o2+d2*n2,3000,0], interpolation='nearest')
    #image = pl.imshow(model[:,:,i], interpolation='nearest')
    ax = pl.gca()

    image.set_cmap('viridis')
    ax.xaxis.tick_top()
    ax.xaxis.set_label_position('top')

    pl.xlabel("$x$ (m)")
    #pl.xticks([3000,4000,5000,6000,7000])
    #pl.xlim(3000,7000)

    pl.ylim(3000,0)
    pl.ylabel("$z$ (m)")
    pl.yticks([0,500,1000,1500,2000,2500,3000])

    # Colorbar
    cb = pl.colorbar(image,
            ax=ax,
            orientation='horizontal',
 #           ticks=([-1, -0.5, 0, 0.5, 1]),
            #ticks=(),
            shrink=1,
            pad=0.01,
            aspect=20.0)
    cb.set_label('Gullfaks '+title[i])
 #ax.set_yticklabels(ax.get_yticks(),rotation=90)
 # Colorbar

# Plot shot positions
 #for i in range(0,101):
 #	pl.scatter(i*10*10, 10.0, marker='.', color='y', s=10, zorder=2)
 	#pl.scatter(500+i*50, 500, marker='x', color='y', s=3)

 #for i in range(0,500):
 #pl.plot([0,10000], [10,10], color='r', linewidth=3, zorder=1)

    pl.savefig('AGU2017/'+part+'/'+param[i]+'.pdf',bbox_inches='tight')
    #pl.show()

fig = pl.figure(figsize=(5,5))
image = pl.imshow(model[:,:,4],extent=[o2,o2+d2*n2,3000,0], interpolation='nearest')
ax = pl.gca()

image.set_cmap('viridis')
ax.xaxis.tick_top()
ax.xaxis.set_label_position('top')

pl.xlabel("$x$ (m)")
#pl.xticks([3000,4000,5000,6000,7000])
#pl.xlim(3000,7000)

pl.ylim(3000,0)
pl.ylabel("$z$ (m)")
pl.yticks([0,500,1000,1500,2000,2500,3000])

# Colorbar
cb = pl.colorbar(image,
        ax=ax,
        orientation='horizontal',
        shrink=1,
        pad=0.01,
        aspect=20.0)
cb.set_label('Gullfaks '+title[4])

# Plot shot positions
 #for i in range(0,101):
 #	pl.scatter(i*10*10, 10.0, marker='.', color='y', s=10, zorder=2)
 	#pl.scatter(500+i*50, 500, marker='x', color='y', s=3)

#for i in range(0,500):
pl.plot([5000,5000], [1590,2190], color='white', alpha=0.8, linewidth=1.5, zorder=1)
pl.plot([4690,5290], [1900,1900], color='black', alpha=0.8, linewidth=1.5, zorder=2)
pl.plot([4999,5001], [1899,1901], color='red'  , alpha=1.0, linewidth=1.5, zorder=3)

pl.savefig('AGU2017/'+part+'/'+param[4]+'_hess.pdf',bbox_inches='tight')



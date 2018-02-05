#!/usr/bin/env python
import rsf.api as rsf
import numpy as np
import pylab as pl
import sys as sysi
import os
#import matplotlib.pyplot as mpl

# Plot the full Hessian from one experiment.
def normalize(x):
    return x/(np.max(np.abs(x)))

model = 'barn/homo/'

project = '/home/vegahag/msim-ext/AGU2017/'+model

#part = 'hessian_'
part = 'vp_8Hz-zdir-short'

freq = ''
#freq = '16Hz'
#freq = '32Hz'

pos  = '70'

hessians = ['H1', 'H2']
params   = ['rh' , 'vp' , 'vs']
paramTit = ['\\rho', 'v_p', 'v_s']
perturbs = ['vp']

folder = project+part+freq+'/kernels/'

if not os.path.exists(part+'/'):
    os.makedirs(part+'/')

dz = 10
dx = 10

nPlots = len(hessians)*len(params)*len(perturbs)

#toRead = folder+hessians[0]+'-'+params[0]+'-delta_'+perturbs[0]+'-z'+pos+'-x500.rsf'
toRead = folder+hessians[0]+'-'+params[0]+'-delta_'+perturbs[0]+'-z'+pos+'-x350.rsf'
print toRead
fin   = rsf.Input(toRead)
[n,m] = fin.shape()
o1 = fin.float("o1")
o2 = fin.float("o2")
d1 = fin.float("d1")
d2 = fin.float("d2")
n1 = fin.int("n1")
n2 = fin.int("n2")

data  = np.zeros([n,m],'f')
model = np.zeros([m,n,nPlots],'f')
title = ''

it  = 0
for h in hessians:
    pit = 0
    for p in params:
        for d in perturbs:
            #fin   = rsf.Input(folder+h+'-'+p+'-delta_'+d+'-z'+pos+'-x500.rsf')
            fin   = rsf.Input(folder+h+'-'+p+'-delta_'+d+'-z'+pos+'-x350.rsf')
            fin.read(data)
            model[:,:,it] = data.transpose()
            title += '$\\mathbf{H}^{'+paramTit[pit]+'}_{'+h[1:]+'}\\delta '+d+'_'+pos+'$'
            it += 1
        pit += 1

for i in range(0,nPlots):
    model[:,:,i] = normalize(model[:,:,i])

param = ['rho-H1','vp-H1','vs-H1',
         'rho-H2','vp-H2','vs-H2',]

title = ['$\mathbf{H}^\\rho_1$','$\mathbf{H}^{v_p}_1$','$\mathbf{H}^{v_s}_1$',
         '$\mathbf{H}^\\rho_2$','$\mathbf{H}^{v_p}_2$','$\mathbf{H}^{v_s}_2$']


# plus one to get the correct indices
nParam = len(param)

pl.rc('text', usetex=True)

# Plot
it = 0
for h in hessians:
    for p in params:
        for d in perturbs:
            # Axis

            fig = pl.figure(figsize=(5,5))
            image = pl.imshow(model[:,:,i],extent=[o2,o2+d2*n2,3000,0], interpolation='nearest')
            ax = pl.gca()

            image.set_cmap('seismic')
            #image.set_clim([-.05,.05])
            image.set_clim([-.5,.5])

            ax.xaxis.tick_top()
            ax.xaxis.set_label_position('top')

            pl.xlabel("$x$ (m)")
            #pl.xticks([0,1000,2000,3000,4000])
            #pl.xlim(0,4000)
            pl.xlim(0,7000)

            #pl.ylim(3000,0)
            pl.ylim(1400,0)
            pl.ylabel("$z$ (m)")
            #pl.yticks([0,500,1000,1500,2000,2500,3000])

            # Colorbar
            cb = pl.colorbar(image,
                    ax=ax,
                    orientation='horizontal',
                    ticks=([ -.05, -.025, 0, .025, .05]),
                    #ticks=(),
                    shrink=1,
                    pad=0.01,
                    aspect=20.0)
            cb.set_label(title[i])

            pl.savefig(part+'/'+h+'-'+p+'-delta'+d+'-'+pos+'.pdf',bbox_inches='tight')
            #pl.show()


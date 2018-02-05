#!/usr/bin/env python
import rsf.api as rsf
import numpy as np
import pylab as pl
import sys as sys
import os
import matplotlib.patches as patches
from matplotlib.colors import LinearSegmentedColormap
from subprocess import call

direction = 'z'
offset    = 'short'
cLim      = 0.05 #0.00005
normalise = 1


baseFolder  = '/home/vegahag/msim-ext/AGU2017/barn/homo/'
simFolder   = 'vp_8Hz-{}dir-{}/'.format(direction,offset)

print 'Direction: {}\tOffset: {}\tcLim: {}'.format(direction,offset,cLim)

modelFolder = baseFolder + 'model/'
dataFolder  = baseFolder + simFolder + 'kernels/'
saveFolder  = baseFolder + simFolder + 'hessAnim/'

model = 'vp'
fileName = 'H1-rh-delta_rh'
name = '$H^1_\\rho\delta\\rho$'

if direction == 'x':
    zPos = range(190,191) #(190,191) (159,220)
    xPos = range(469,530) #(469,530) (500,501)
elif direction == 'z':
    zPos = range( 50,140)#range(159,220)
    xPos = range(350,351) #range(500,501)

if offset == 'short':
    sPos = [349,349] #[499,501]
elif offset == 'long':
    sPos = [1,999]

t = max([len(zPos),len(xPos)])

if not os.path.exists(saveFolder):
    os.makedirs(saveFolder)

fin   = rsf.Input(dataFolder+fileName+'-z100-x350.rsf') #rsf.Input(dataFolder+fileName+'-z190-x500.rsf')
[n,m] = fin.shape()

data  = np.zeros([n,m],'f')
fin.read(data)

data = np.array(data.transpose())
data = data/np.max(np.abs(data)) # Normalise data

pl.rc('text', usetex=True)

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

fNames = ['rh','vp','vs']
lNames = ['\\rho','{v_p}','{v_s}']
Hessian = np.zeros([m,n,t,4,3,3],'f') #[zdim,xdim,pPos,H,param,pert]
print 'Reading data'
for h in [1,2,3]:
    for param in range(3):
        for pert in range(3):
            fileName = 'H{}-{}-delta_{}'.format(h,fNames[param],fNames[pert])
            it = 0
            for i in zPos:
                for j in xPos:
                    #print 'Frame {} of {}'.format(it,t)
                    data  = np.zeros([n,m],'f')
                    fin = rsf.Input(dataFolder+fileName+'-z{:2d}-x{:3d}.rsf'.format(i,j))
                    fin.read(data)
                    #data = np.array(data.transpose())
                    #print [it,h,param,pert]
                    if normalise:
                        data = data/np.max(np.abs(data)+np.finfo(float).eps) # Normalise data
                    Hessian[:,:,it,h,param,pert] = np.array(data.transpose())
                    #print '{}\t{:12.12e}'.format(fileName,np.max(np.abs(data)))
                    it += 1

print 'Summing Hessian'
for param in range(3):
    for pert in range(3):
        for it in range(t):
            Hessian[:,:,it,0,param,pert] = Hessian[:,:,it,1,param,pert] + Hessian[:,:,it,2,param,pert] + Hessian[:,:,it,3,param,pert]

print 'Saving data'
for h in [1,2,3,0]:
    for param in range(3):
        for pert in range(3):
            it = 0
            fileName = 'H{}-{}-delta_{}'.format(h,fNames[param],fNames[pert])
            name = '$H^{}_{}\delta {}$'.format(h,lNames[param],lNames[pert])
            if h == 0:
                name = '$\Sigma H_{}\delta {}$'.format(lNames[param],lNames[pert])
            for i in zPos:
                for j in xPos:

                    data = Hessian[:,:,it,h,param,pert]
                    it += 1

                    fig = pl.figure()
                    #fig.set_size_inches(10,3)
                    fig.set_size_inches(7,1.4)
                    ax  = pl.Axes(fig, [0., 0., 1., 1.])
                    ax.set_axis_off()
                    fig.add_axes(ax)

                    mod = ax.imshow(model,cmap='terrain') #terrain viridis

                    img = ax.imshow(data,interpolation='nearest', cmap=grayAlpha)
                    img.set_clim([-cLim,cLim])

                    ptr = patches.Circle((j-1,i-1),1,linewidth=1,edgecolor='c',facecolor='none',alpha=0.5)
                    ax.add_patch(ptr)
                    #inc = patches.Rectangle((499-5,189-5),10,10,linewidth=1,edgecolor='y',facecolor='none',alpha=0.5)
                    #ax.add_patch(inc)
                    src = ax.plot(sPos,[3,3],'g^',alpha=0.8) #short: 499,501 long: 1,999

                    pPos = 'Perturbation: ({:d} m,{:d} m)'.format(10*j,10*i)
                    #ax.annotate(pPos, xy=(995,293), fontsize=15, color='black',horizontalalignment='right',fontweight='bold')
                    #ax.annotate(name, xy=( 25,293), fontsize=15, color='black',horizontalalignment='left',fontweight='bold')
                    ax.annotate(pPos, xy=(995,133), fontsize=15, color='black',horizontalalignment='right',fontweight='bold')
                    ax.annotate(name, xy=( 25,133), fontsize=15, color='black',horizontalalignment='left',fontweight='bold')

                    pl.savefig(saveFolder+fileName+'-z{:03d}-x{:03d}'.format(i,j)+'.png')
                    pl.close()
                    #pl.show()

# System calls for merging pngs into gif and converting to movie
print 'Creating .mp4'
a = 0
for param in range(3):
    for pert in range(3):
        aList = ''
        it = 0
        for i in zPos:
            for j in xPos:
                inFiles = (saveFolder + 'H1-{}-delta_{}-z{}-x{}.png '.format(fNames[param],fNames[pert],i,j)
                     + saveFolder + 'H2-{}-delta_{}-z{}-x{}.png '.format(fNames[param],fNames[pert],i,j)
                     + saveFolder + 'H3-{}-delta_{}-z{}-x{}.png '.format(fNames[param],fNames[pert],i,j)
                     + saveFolder + 'H0-{}-delta_{}-z{}-x{}.png '.format(fNames[param],fNames[pert],i,j))
                options = '-background black -tile 2x2 -geometry +2+2 -'
                imgSave = saveFolder + 'H-{}-delta_{}-{:04d}.png '.format(fNames[param],fNames[pert],it)
                cmnd = 'montage '+ inFiles + options + '| convert - -bordercolor black -border 276x0 ' + imgSave
                call(cmnd,shell=True)
                call('cp ' + saveFolder + 'H-{}-delta_{}-{:04d}.png '.format(fNames[param],fNames[pert],it) + saveFolder + 'H{:04d}.png'.format(a),shell=True)
                it += 1
                a  += 1
        cmnd = ('ffmpeg '
                + '-y -r 8 -f image2 -i '
                + saveFolder + 'H-{}-delta_{}-%04d.png '.format(fNames[param],fNames[pert])
                + '-vcodec libx264 -crf 10 '
                + saveFolder + 'H_{0}-d{1}.mp4'.format(fNames[param],fNames[pert]))
        call(cmnd,shell=True)
call('ffmpeg -v 16 -y -r 8 -f image2 -i ' + saveFolder + 'H%04d.png -vcodec libx264 -crf 10 ' + saveFolder + 'H.mp4',shell=True)
print 'Total images: {}'.format(a)

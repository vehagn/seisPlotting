#!/usr/bin/env python
import os
from subprocess import call

zFolder = 'vp_8Hz-zdir-short/hessAnim/'
xFolder = 'vp_8Hz-xdir-short/hessAnim/'

saveFolder = 'short/'

if not os.path.exists(saveFolder):
    os.makedirs(saveFolder)

tot = 549
params = {0: 'rho',
          1: 'vp',
          2: 'vs'}

for i in range(0,tot):
    call('montage '
         + xFolder + 'H{:04d}.png '.format(i)
         + zFolder + 'H{:04d}.png '.format(i)
         + '-background black -tile 1x2 -geometry +0+56 '
         + saveFolder + 'H{:04d}.png '.format(i),
         shell=True)
    print '{} {}'.format(i,(i/(tot/9))%3)
    call('convert '
         + '-pointsize 50 -fill white -gravity center '
         + '-annotate +0-690 "Short offset horizontal {} perturbation" '.format(params[(i/(tot/9))%3])
         + '-annotate +0+30  "Short offset vertical {} perturbation" '.format(params[(i/(tot/9))%3])
         + saveFolder + 'H{0:04d}.png '.format(i)
         + saveFolder + 'H{0:04d}.png '.format(i),
         shell=True)


call('ffmpeg '
     + '-v 16 -y -r 8 -f image2 -i '
     + saveFolder + 'H%04d.png '
     + '-vcodec libx264 -crf 10 -pix_fmt yuv420p -video_size 2560x1440 '
     + saveFolder + 'H.mp4',
     shell=True)

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 14 14:02:51 2018

@author: peter
"""

import numpy as np
import glob
import pickle

import matplotlib.pyplot as plt
from matplotlib import rcParams

plt.close('all')

# IMPORT BOUNDS
# Get folder path containing pickle files
file_list = sorted(glob.glob('./bounds/[0-9]_bounds.pkl'))
bounds = []
for file in file_list:
    with open(file, 'rb') as infile:
        param_space, ub, lb, mean = pickle.load(infile)
        bounds.append(mean)

# IMPORT PREDS
# Get folder path containing files
file_list = sorted(glob.glob('./pred/[0-9].csv'))
preds = []
for k,file_path in enumerate(file_list):
    preds.append(np.genfromtxt(file_path, delimiter=','))

isTested = np.zeros(len(param_space))
all_preds = np.zeros((len(param_space),4))

for k, pol in enumerate(param_space):
    for k2,batch in enumerate(preds):
        for row in batch:
            if (pol==row[0:3]).all():
                isTested[k] += 1
                all_preds[k][k2] = row[4]

all_preds[all_preds == 0] = np.nan

means_preds = np.nanmean(all_preds, axis=1)
stdevs_preds = np.nanstd(all_preds, axis=1)

sterr_preds = 1.96*stdevs_preds/np.sqrt(isTested) # 95% CI

ye = [(mean-lb)/(5*0.5**5),(ub-mean)/(5*0.5**5)]

######################## PLOTTING ########################
FS = 12
LW = 3

rcParams['pdf.fonttype'] = 42
rcParams['ps.fonttype'] = 42
rcParams['font.size'] = FS
rcParams['axes.labelsize'] = FS
rcParams['xtick.labelsize'] = FS
rcParams['ytick.labelsize'] = FS
rcParams['font.sans-serif'] = ['Arial']

lower_lim, upper_lim = 500, 1400

plt.figure()
plt.plot((-100,upper_lim+100),(-100,upper_lim+100), ls='--', c='.3',label='_nolegend_')

cm = plt.get_cmap('winter')
colors = [cm(1.*i/5) for i in np.arange(5)]
colors[0] = (0,0,0,1)

leg = [str(i)+" repetitions" for i in np.arange(5)]
leg[1] = "1 repetition"

for k in np.arange(1, 5):
    idx = np.where(isTested==k)
    plt.plot(means_preds[idx],mean[idx],'o',color=colors[k],label=leg[k])

plt.xlabel('Mean of prediction input into CLO (cycles)')
plt.ylabel('CLO-estimated cycle life (cycles)')
plt.gca().set_aspect('equal', adjustable='box')
plt.xlim([lower_lim,upper_lim])
plt.ylim([lower_lim,upper_lim])
plt.legend(frameon=False)

plt.tight_layout()
plt.savefig('moreplots/mean_comparison.png',bbox_inches='tight')
plt.savefig('moreplots/mean_comparison.pdf',bbox_inches='tight',format='pdf')

plt.figure()
plt.plot((-100,upper_lim+100),(0,0), ls='--', c='.3',label='_nolegend_')

## Residual plots
for k in np.arange(5):
    idx = np.where(isTested==k)
    plt.plot(mean[idx],means_preds[idx]-mean[idx],'o',color=colors[k],label=leg[k])

plt.xlabel('CLO-estimated cycle life (cycles)')
plt.ylabel('Difference between CLO-estimated cycle life\nand mean of prediction input into CLO (cycles)')
#plt.gca().set_aspect('equal', adjustable='box')
plt.xlim([lower_lim,upper_lim])
#plt.ylim([lower_lim,upper_lim])
plt.legend(frameon=False)

plt.tight_layout()
plt.savefig('moreplots/mean_residual.png',bbox_inches='tight')
plt.savefig('moreplots/mean_residual.pdf',bbox_inches='tight',format='pdf')

plt.figure()
## St dev plots
for k in np.arange(2,5):
    idx = np.where(isTested==k)
    plt.plot(sterr_preds[idx],ye[0][idx],'o',color=colors[k],label=leg[k])

plt.xlabel('95% CI of prediction input into CLO (cycles)')
plt.ylabel('Standard deviation of cycle life\nafter round 4, $\mathit{σ_{4,i}}$')
plt.gca().set_aspect('equal', adjustable='box')

lower_lim, upper_lim = 0, 400

plt.xlim([lower_lim,upper_lim])
plt.ylim([lower_lim,upper_lim])
plt.legend(frameon=False)

plt.tight_layout()
plt.savefig('moreplots/stdev_comparison.png',bbox_inches='tight')
plt.savefig('moreplots/stdev_comparison.pdf',bbox_inches='tight',format='pdf')
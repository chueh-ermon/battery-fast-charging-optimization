#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 20 07:29:14 2018

@author: peter

"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
import glob
import pickle

plt.close('all')

FS = 14

rcParams['pdf.fonttype'] = 42
rcParams['ps.fonttype'] = 42
rcParams['font.size'] = FS
rcParams['axes.labelsize'] = FS
rcParams['xtick.labelsize'] = FS
rcParams['ytick.labelsize'] = FS
rcParams['font.sans-serif'] = ['Arial']

# IMPORT RESULTS
# Get pickle files of bounds
file_list = sorted(glob.glob('./bounds/[0-9]_bounds.pkl'))
means = []
ubs = []
lbs = []
for file in file_list:
    with open(file, 'rb') as infile:
        param_space, ub, lb, mean = pickle.load(infile)
        means.append(mean)
        ubs.append(ub)
        lbs.append(lb)
    
top_pol_idx = np.argsort(-mean)
param_space = param_space[top_pol_idx]

# Get folder path containing predictions
file_list = sorted(glob.glob('./batch/[0-9].csv'))
batch_data = []
for k, file_path in enumerate(file_list):
    batch = np.genfromtxt(file_path, delimiter=',')
    # convert protocols to index of param_space
    indices_batch = []
    for p in batch:
        idx = np.where((p[0]==param_space[:,0])*(p[1]==param_space[:,1])*(p[2]==param_space[:,2]))[0][0]
        indices_batch.append(idx)
    
    batch_data.append(indices_batch)

## Find number of batches and policies
n_batches  = len(means)
n_policies = len(param_space)

## plot
batches = np.arange(n_batches-1)+1
plt.subplots(3,2,figsize=(9,12))

batches = np.arange(n_batches-1)+1

### CHANGE THIS SETTING FOR TWO VERSIONS OF PLOT
plot_bounds_with_beta = False

## Bounds
for k, mean in enumerate(means):
    mean = mean[top_pol_idx]
    
    # indices of selected protocols
    indices = batch_data[k]
    unselected_indices = np.setdiff1d(np.arange(224),indices)
    
    # y uncertainties for unselected protocols
    ub = ubs[k][top_pol_idx][unselected_indices]
    lb = lbs[k][top_pol_idx][unselected_indices]
    if plot_bounds_with_beta:
        ye = [(mean[unselected_indices]-lb),(ub-mean[unselected_indices])]
    else: # divide by beta
        ye = [(mean[unselected_indices]-lb)/(5*0.5**k),(ub-mean[unselected_indices])/(5*0.5**k)]
    
    
    # y uncertainties for selected protocols
    ub = ubs[k][top_pol_idx][indices]
    lb = lbs[k][top_pol_idx][indices]
    if plot_bounds_with_beta:
        ye2 = [(mean[indices]-lb),(ub-mean[indices])]
    else: # divide by beta
        ye2 = [(mean[indices]-lb)/(5*0.5**k),(ub-mean[indices])/(5*0.5**k)]
    
    ax = plt.subplot2grid((5, 2), (k, 0), colspan=2)
    ax.errorbar(np.arange(224)[unselected_indices],mean[unselected_indices],yerr=ye,fmt='o',color=[0.1,0.4,0.8],capsize=2)
    ax.errorbar(np.arange(224)[indices],mean[indices],yerr=ye2,fmt='o',color=[0.8,0.4,0.1],capsize=2)
    
    ax.set_xlim((-1,225))
    if plot_bounds_with_beta:
        ax.set_ylim((500,1500))
    else:
        ax.set_ylim((647,1247))
    ax.set_xlabel('Protocol rank after round 4')
    if k==0:
        if plot_bounds_with_beta:
            ax.set_ylabel('Est. cycle life before\nround 1, $\mathit{μ_{0,i}±β_{0}σ_{0,i}}$')
        else:
            ax.set_ylabel('Est. cycle life before\nround 1, $\mathit{μ_{0,i}±σ_{0,i}}$')
    else:
        if plot_bounds_with_beta:
            mathstr = '{μ_{'+str(k)+',i}±β_{'+str(k)+'}σ_{'+str(k)+',i}}'
        else:
            mathstr = '{μ_{'+str(k)+',i}±σ_{'+str(k)+',i}}'
        ax.set_ylabel('Est. cycle life after\n round {}, $\mathit'.format(k)+mathstr+'$')
    ax.set_xticks([], [])
    ax.set_title(chr(97+k), loc='left', weight='bold')
    
    if k==4:
        plt.legend(['Unselected protocols','Selected protocols'],loc='upper right',frameon=False)

plt.tight_layout()
if plot_bounds_with_beta:
    plt.savefig('bounds_evolution_withbeta.png', bbox_inches='tight')
    plt.savefig('bounds_evolution_withbeta.pdf', bbox_inches='tight', format='pdf')
else:
    plt.savefig('bounds_evolution_nobeta.png', bbox_inches='tight')
    plt.savefig('bounds_evolution_nobeta.pdf', bbox_inches='tight', format='pdf')
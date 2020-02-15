#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 16 20:12:13 2019

@author: peter

This script plots the results of simulations comparing the closed loop with OED
to random searching. The results were generated on a cluster
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
import pickle

FS = 14
LW = 2

rcParams['pdf.fonttype'] = 42
rcParams['ps.fonttype'] = 42
rcParams['font.size'] = FS
rcParams['axes.labelsize'] = FS
rcParams['xtick.labelsize'] = FS
rcParams['ytick.labelsize'] = FS
rcParams['font.sans-serif'] = ['Arial']

file = 'sim_ablation.pkl'
with open(file, 'rb') as infile:
        random_performance_means, random_performance_stds, \
        oed_performance_means, oed_performance_stds = pickle.load(infile)

num_channels = [1, 8, 16, 24, 48]
num_rounds = [1, 2, 3, 4]

fig, ax = plt.subplots(2,3,figsize=(12,8),sharey=True)

for k, channels in enumerate(num_channels):
    temp_ax = ax[int(k/3)][k%3]
    temp_ax.errorbar(np.array(num_rounds),
                 random_performance_means[k],
                 linewidth = LW,
                 yerr=1.96*np.vstack((random_performance_stds[k], 
                                      random_performance_stds[k])),
                 marker='o',
                 linestyle=':',
                 color=u'#7A68A6',
                 label='CLO w/ random')
    
    temp_ax.errorbar(np.array(num_rounds),
                     oed_performance_means[k][1:],
                     linewidth = LW,
                     yerr=1.96*np.vstack((oed_performance_stds[k][1:], 
                                          oed_performance_stds[k][1:])),
                     marker='o',
                     linestyle=':',
                     color=u'#467821',
                     label='CLO w/ MAB')
    
    temp_ax.set_title(chr(k+97), loc='left', weight='bold')
    #temp_ax.set_xlim((0.5, 4.5))
    temp_ax.set_ylim((700,1200))
    annotation_text = ' channels'
    if k==0:
        annotation_text = ' channel'
    temp_ax.annotate(str(channels) + annotation_text, (4.1, 720),
                     fontsize=FS,horizontalalignment='right')
    if k==0:
        temp_ax.legend(loc='upper left',frameon=False)
        
    """
    if int(k/3)==1 or k==2:
        temp_ax.set_xlabel('Number of rounds of testing',fontsize=FS)
    else:
        plt.setp(temp_ax.get_xticklabels(), visible=False)
    """
    temp_ax.set_xlabel('Number of rounds of testing',fontsize=FS)
        
    if k%3 == 0:
        temp_ax.set_ylabel('True cycle life of\ncurrent best protocol',fontsize=FS)
        
ax[-1, -1].axis('off')

## SAVE
plt.tight_layout()
plt.savefig('sim_ablation.png', bbox_inches = 'tight')
plt.savefig('sim_ablation.pdf', bbox_inches = 'tight', format='pdf')
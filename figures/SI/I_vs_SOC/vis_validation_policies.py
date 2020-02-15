#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 25 08:57:12 2018

@author: peter
"""

from IvSOC_function import plot_policy
import matplotlib.pyplot as plt
from matplotlib import rcParams
import numpy as np
import glob
import pickle

FS = 12

rcParams['pdf.fonttype'] = 42
rcParams['ps.fonttype'] = 42
rcParams['font.size'] = FS
rcParams['axes.labelsize'] = FS
rcParams['xtick.labelsize'] = FS
rcParams['ytick.labelsize'] = FS
rcParams['font.sans-serif'] = ['Arial']

fig, ax = plt.subplots(3,3,figsize=(8,8), sharex=True, sharey=True)

filename = 'validation_pols.csv'
pols = np.genfromtxt(filename, delimiter=',',skip_header=1)

########## LOAD DATA ##########
# Load predictions
filename = 'predictions.csv'
pred_data = np.genfromtxt(filename, delimiter=',',skip_header=1)

validation_policies = pred_data[:,0:3]
predicted_lifetimes = pred_data[:,3:]

# Load final results
filename = 'final_results.csv'
final_data = np.genfromtxt(filename, delimiter=',',skip_header=1)

final_lifetimes = final_data[:,3:]

# Load OED means
oed_bounds_file = glob.glob('4_bounds.pkl')[0]
with open(oed_bounds_file, 'rb') as infile:
        param_space, ub, lb, all_oed_means = pickle.load(infile)

intersect = [i for i, policy in enumerate(param_space) if (policy == validation_policies).all(1).any()]

oed_means = all_oed_means[intersect]
oed_policy_subset = param_space[intersect]

# reorder oed_means by comparing ordering of oed_policy_subset with validation_policies 
idx = np.argwhere(np.all(validation_policies[:, None] == oed_policy_subset, axis=-1))[:, 1]
oed_means = oed_means[idx]

########## CALCULATIONS ##########

# Summary statistics
pred_means = np.round(np.nanmean(predicted_lifetimes,axis=1))
pred_sterr = np.round(1.96*np.nanstd(predicted_lifetimes,axis=1)/np.sqrt(5))

final_means = np.round(np.nanmean(final_lifetimes,axis=1))
final_sterr = np.round(1.96*np.nanstd(final_lifetimes,axis=1)/np.sqrt(5))

########## PLOT ##########
for k, p in enumerate(pols):
    life_dict = {'oed':oed_means[k], 
                 'pred':pred_means[k],
                 'pred_sterr':pred_sterr[k],
                 'final':final_means[k],
                 'final_sterr':final_sterr[k]}
    ax_temp = ax[int(k/3)][k%3]
    plot_policy(p[0],p[1],p[2],ax_temp,life_dict)
    if int(k/3) == 2:
        ax_temp.set_xlabel('State of charge (%)',fontsize=FS)
    if k%3 == 0:
        ax_temp.set_ylabel('Current (C rate)',fontsize=FS)
    ax_temp.set_title(chr(k+97), loc='left', weight='bold')
    

plt.tight_layout()
plt.savefig('val_pols.png',bbox_inches='tight')
plt.savefig('val_pols.pdf',bbox_inches='tight',format='pdf')
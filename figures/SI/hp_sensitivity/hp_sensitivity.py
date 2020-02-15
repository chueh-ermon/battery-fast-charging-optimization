#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 28 15:51:58 2018

@author: peter
"""

import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import rcParams

lifetime_key = 'l4'

#sims = pd.read_csv('log.csv', names=['beta','gamma','epsilon','seed',lifetime_key])
sims = pd.read_csv('log.csv', names=['beta','gamma','epsilon','seed',
                                     'l1','l2','l3','l4','l5','l6','l7','l8','l9','l10'])

# Add column for rank
lifetimes = pd.read_csv('policies_lifetimes_sim.csv', 
                        names=['C1','C2','C3','C4','Lifetime'])
lifetimes_sorted = lifetimes.sort_values('Lifetime',ascending=False)
lifetimes_sorted = lifetimes_sorted.reset_index(drop=True)

lifetime_idx_array = np.zeros((len(sims),1))

for k,val in enumerate(sims[lifetime_key]):
    lifetime_idx_array[k] = lifetimes_sorted.index[lifetimes_sorted['Lifetime']==val].tolist()[0]

sims['mean rank'] = lifetime_idx_array


# Preinitialize
beta_len = len(sims.beta.unique())
gamma_len = len(sims.gamma.unique())
epsilon_len = len(sims.epsilon.unique())

means = np.zeros((beta_len,gamma_len,epsilon_len))
std = np.zeros((beta_len,gamma_len,epsilon_len))
mean_rank = np.zeros((beta_len,gamma_len,epsilon_len))
min_rank = np.zeros((beta_len,gamma_len,epsilon_len))
max_rank = np.zeros((beta_len,gamma_len,epsilon_len))

beta_count = 0
gamma_count = 0
epsilon_count = 0

for beta, subgroup in sims.groupby('beta'):
    for gamma, subgroup2 in subgroup.groupby('gamma'):
        for epsilon, subgroup3 in subgroup2.groupby('epsilon'):
            print('beta = ', beta,', gamma = ', gamma,', epsilon = ', epsilon)
            #print('beta_count = ', beta_count,', gamma_count = ', gamma_count,
            #      ', epsilon_count = ', epsilon_count)

            means[beta_count,gamma_count,epsilon_count] = subgroup3[lifetime_key].mean()
            std[beta_count,gamma_count,epsilon_count] = subgroup3[lifetime_key].std()

            mean_rank[beta_count,gamma_count,epsilon_count] = subgroup3['mean rank'].mean()
            min_rank[beta_count,gamma_count,epsilon_count] = subgroup3['mean rank'].min()
            max_rank[beta_count,gamma_count,epsilon_count] = subgroup3['mean rank'].max()

            epsilon_count += 1

        epsilon_count = 0
        gamma_count += 1

    gamma_count = 0
    beta_count += 1
    
min_lifetime = np.amin(means)
max_lifetime = np.amax(means)


## INITIALIZE PLOT
# SETTINGS
FS = 14
colormap = 'plasma_r'
lower_lifetime_lim = 1000

## PLOT
rcParams['pdf.fonttype'] = 42
rcParams['ps.fonttype'] = 42
rcParams['font.size'] = FS
rcParams['axes.labelsize'] = FS
rcParams['xtick.labelsize'] = FS
rcParams['ytick.labelsize'] = FS
rcParams['font.sans-serif'] = ['Arial']

fig, ax = plt.subplots(2,4,figsize=(12,9),sharey=True)
#plt.style.use('classic')
plt.set_cmap(colormap)
minn, maxx = min_lifetime, max_lifetime

fig.subplots_adjust(right=0.8)
fig.subplots_adjust(top=0.93)

k2 = 0

# FUNCTION FOR LOOPING THROUGH COMBINATIONS
for k, gamma in enumerate(sims.gamma.unique()):
    temp_ax = ax[int(k/4)][k%4]
    plt.axis('square')
    
    ## PLOT COMBINATIONS
    [X,Y] = np.meshgrid(sims.beta.unique(),sims.epsilon.unique())
    temp_ax.scatter(X.ravel(),Y.ravel(),vmin=minn,vmax=maxx,
                c=means[:,k,:].ravel(),zorder=2,s=100)
    
    temp_ax.set_title(chr(k+97),loc='left', weight='bold',fontsize=FS)
    
    temp_ax.annotate('γ=' + str(gamma), (15, 0.95), fontsize=FS,horizontalalignment='right')
    
    # Add xlabel/xticks
    if int(k/4)==1 or k==3:
        temp_ax.set_xlabel(r'$\beta_0$',fontsize=FS)
    #plt.setp(temp_ax.get_xticklabels(), visible=False)
    
    if k%4 == 0:
        temp_ax.set_ylabel('ε',fontsize=FS)
    
    temp_ax.set_xlim((0.1, 20))
    temp_ax.set_ylim((0.4,1.0))
    temp_ax.set_xscale('log')

ax[-1, -1].axis('off')

# ADD COLORBAR
cbar_ax = fig.add_axes([0.85, 0.15, 0.04, 0.72]) # [left, bottom, width, height]
norm = matplotlib.colors.Normalize(minn, maxx)
m = plt.cm.ScalarMappable(norm=norm, cmap=colormap)
m.set_array([])
cbar = fig.colorbar(m, cax=cbar_ax)
cbar.ax.tick_params(labelsize=FS,length=0)
cbar.ax.set_title('Cycle life\nof best protocol\n(simulated)',fontsize=FS)

## SAVE
plt.savefig('hyperparameter_sensitivity.png', bbox_inches = 'tight')
plt.savefig('hyperparameter_sensitivity.pdf', bbox_inches = 'tight',format='pdf')

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 20 07:29:14 2018

@author: peter
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.cm as cm
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import rcParams
import glob
import seaborn as sns

plt.close('all')

FS = 14

rcParams['pdf.fonttype'] = 42
rcParams['ps.fonttype'] = 42
rcParams['font.size'] = FS
rcParams['axes.labelsize'] = FS
rcParams['xtick.labelsize'] = FS
rcParams['ytick.labelsize'] = FS
rcParams['font.sans-serif'] = ['Arial']

##############################################################################
# PLOTTING PARAMETERS
batches_to_plot = [0,1,2,3]
labels = ['a','b','c','d']

colormap = 'plasma_r'
el, az = 30, 240
point_size = 70
num_policies = 224
seed = 0
##############################################################################

# IMPORT RESULTS
# Get folder path containing files
file_list = sorted(glob.glob('./pred/[0-9].csv'))
data = []
min_lifetime = 10000
max_lifetime = -1
for k,file_path in enumerate(file_list):
    data.append(np.genfromtxt(file_path, delimiter=','))
    min_lifetime = min(np.min(data[k][:,4]),min_lifetime)
    max_lifetime = max(np.max(data[k][:,4]),max_lifetime)

fig = plt.figure(figsize=(20,5))

## MAKE SUBPLOTS
for k, batch_idx in enumerate(batches_to_plot):
    with sns.axes_style('white'):
        ax = fig.add_subplot(1, len(batches_to_plot), k+1, projection='3d')
        #ax = plt.subplot(2, len(batches_to_plot), k+1, projection='3d')
    ax.set_aspect('equal')
    
    ## PLOT POLICIES
    policy_subset = data[k][:,0:4]
    lifetime_subset = data[k][:,4]
    if np.size(lifetime_subset):
        with plt.style.context(('classic')):
            plt.set_cmap(colormap)
            ax.scatter(policy_subset[:,0],policy_subset[:,1],\
                       policy_subset[:,2],vmin=min_lifetime,vmax=max_lifetime,\
                       c=lifetime_subset.ravel(),zorder=2,s=100)
    
    ax.set_xlabel('CC1',fontsize=FS), ax.set_xlim([3, 8])
    ax.set_ylabel('CC2',fontsize=FS), ax.set_ylim([3, 8])
    ax.set_zlabel('CC3',fontsize=FS), ax.set_zlim([3, 8])
    #ax.set_title('Before batch '+str(batch_idx))
    ax.set_title(labels[k], loc='left', weight='bold')
    
    ax.view_init(elev=el, azim=az)

# ADD COLORBAR
plt.tight_layout()
plt.subplots_adjust(left=0.02,right=0.92)

cbar_ax = fig.add_axes([0.93, 0.2, 0.02, 0.6]) # [left, bottom, width, height]
norm = matplotlib.colors.Normalize(min_lifetime, max_lifetime)
m = plt.cm.ScalarMappable(norm=norm, cmap=colormap)
m.set_array([])

cbar = plt.colorbar(m, cax=cbar_ax)
cbar.ax.tick_params(labelsize=FS,length=0)
cbar.ax.set_title('Predicted\ncycle life',fontsize=FS)

plt.savefig('pred_evolution.png')
plt.savefig('pred_evolution.pdf',format='pdf')
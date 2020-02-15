#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 14 14:02:51 2018

@author: peter

For manual axis sizing: See lines:
    - ax = fig.add_axes([0.27+0.3*(k-1),0.55,0.4,0.4],projection='3d')
    - 
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams

import matplotlib
import matplotlib.cm as cm
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.axes_grid1 import make_axes_locatable
import seaborn as sns

import glob
import pickle
from scipy.stats import pearsonr, linregress

fig, axes = plt.subplots(3,3,figsize=(16,16))
axes[0,0].set_axis_off()
axes[0,1].set_axis_off()
axes[0,2].set_axis_off()
axes[1,0].set_axis_off()
axes[1,1].set_axis_off()
axes[1,2].set_axis_off()
axes[2,0].set_axis_off()
axes[2,1].set_axis_off()
axes[2,2].set_axis_off()


fig.set_size_inches(w=15,h=11)

FS = 14
LW = 3

rcParams['pdf.fonttype'] = 42
rcParams['ps.fonttype'] = 42
rcParams['font.size'] = FS
rcParams['axes.labelsize'] = FS
rcParams['xtick.labelsize'] = FS
rcParams['ytick.labelsize'] = FS
rcParams['font.sans-serif'] = ['Arial']


def text(x1,y1,x2,y2,k):
    ax.annotate("Round "+str(k+1), xy=(x2, y1), xycoords='figure fraction',
                xytext=(x1, y1), textcoords='figure fraction',
                size=20, va="center", ha="center",
                bbox=dict(boxstyle="round", fc="w"))

def arrow(x1,y1,x2,y2):
    ax.annotate("", xy=(x2, y1), xycoords='figure fraction',
                xytext=(x1, y1), textcoords='figure fraction',
                size=20, va="center", ha="center",
                bbox=dict(boxstyle="round4", fc="w"),
                arrowprops=dict(arrowstyle="-|>",
                                connectionstyle="arc3,rad=-1",
                                relpos=(1., 0.),fc="k"))

##############################################################################
# PLOTTING PARAMETERS
batches_to_plot = [0,1,2,3]

colormap = 'winter_r'
el, az = 30, 240
point_size = 50
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

## MAKE SUBPLOTS
for k, batch_idx in enumerate(batches_to_plot):
    with sns.axes_style('white'):
        ax = fig.add_axes([0.26+0.21*(k-1),0.82,0.29,0.29],projection='3d')
        #ax = fig.add_subplot(1, len(batches_to_plot), k+1, projection='3d')
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
                       c=lifetime_subset.ravel(),zorder=2,s=point_size)
    
    ax.set_xlim([3, 8]), ax.set_ylim([3, 8]), ax.set_zlim([3, 8])
    ax.set_xticks([4,6,8]), ax.set_xticklabels([4,6,8])
    ax.set_yticks([4,6,8]), ax.set_yticklabels([4,6,8])
    ax.set_zticks([4,6,8]), ax.set_zticklabels([4,6,8])
    
    if k==0:
        ax.set_xlabel('CC1',fontsize=FS) 
        ax.set_ylabel('CC2',fontsize=FS)
        ax.set_zlabel('CC3',fontsize=FS,rotation=90)
        ax.set_title('a', loc='left', weight='bold')
    #ax.set_title('Before batch '+str(batch_idx))
    
    ax.view_init(elev=el, azim=az)


# ADD COLORBAR
#plt.tight_layout()
#plt.subplots_adjust(left=0.02,right=0.92)

cbar_ax = fig.add_axes([0.92, 0.82, 0.02, 0.3]) # [left, bottom, width, height]
norm = matplotlib.colors.Normalize(min_lifetime, max_lifetime)
m = plt.cm.ScalarMappable(norm=norm, cmap=colormap)
m.set_array([])

cbar = plt.colorbar(m, cax=cbar_ax)
cbar.ax.tick_params(labelsize=FS,length=0)
cbar.ax.set_title('Predicted\ncycle life',fontsize=FS)


########## 3b ##########
##############################################################################
# PLOTTING PARAMETERS
batches_to_plot = [0,1,2,3,4]

colormap = 'plasma_r'
el, az = 30, 240
point_size = 50
seed = 0
##############################################################################

# IMPORT RESULTS
# Get folder path containing pickle files
file_list = sorted(glob.glob('./bounds/[0-9]_bounds.pkl'))
data = []
min_lifetime = 10000
max_lifetime = -1
for file in file_list:
    with open(file, 'rb') as infile:
        param_space, ub, lb, mean = pickle.load(infile)
        data.append(mean)
        min_lifetime = min(np.min(mean),min_lifetime)
        max_lifetime = max(np.max(mean),max_lifetime)
        
        
## MAKE SUBPLOTS
for k, batch_idx in enumerate(batches_to_plot):
    with sns.axes_style('white'):
        if k==0:
            ax = fig.add_axes([0.05,0.48,0.24,0.24],projection='3d')
            ax.set_title('b', loc='left', weight='bold')
        else:
            ax = fig.add_axes([0.05+0.165*k,0.48,0.24,0.24],projection='3d')
        #ax = plt.subplot(2, len(batches_to_plot), k+1, projection='3d')
    ax.set_aspect('equal')
    
    ## PLOT POLICIES
    CC1 = param_space[:,0]
    CC2 = param_space[:,1]
    CC3 = param_space[:,2]
    lifetime = data[batch_idx][:]
    with plt.style.context(('classic')):
        plt.set_cmap(colormap)
        ax.scatter(CC1,CC2,CC3, s=point_size, c=lifetime.ravel(),
               vmin=min_lifetime, vmax=max_lifetime)
    
    ax.set_xlim([3, 8]), ax.set_ylim([3, 8]), ax.set_zlim([3, 8])
   
    if k == 0:
        ax.set_xlabel('CC1',fontsize=FS)
        ax.set_ylabel('CC2',fontsize=FS)
        ax.set_zlabel('CC3',fontsize=FS,rotation=90)
    #ax.set_title('Before batch '+str(batch_idx))
    
    
    ax.view_init(elev=el, azim=az)

# ADD COLORBAR
cbar_ax = fig.add_axes([0.92, 0.45, 0.02, 0.3]) # [left, bottom, width, height]
norm = matplotlib.colors.Normalize(min_lifetime, max_lifetime)
m = plt.cm.ScalarMappable(norm=norm, cmap=colormap)
m.set_array([])

cbar = plt.colorbar(m, cax=cbar_ax)
cbar.ax.tick_params(labelsize=FS,length=0)
cbar.ax.set_title('CLO-estimated\ncycle life',fontsize=FS)


## ADD ARROWS
margin = 0.18
arrow(0.15,0.61,0.3,0.5)
arrow(0.33,0.61,0.48,0.5)
arrow(0.51,0.61,0.66,0.5)
arrow(0.69,0.61,0.84,0.5)

#arrow(0.15+0.18,0.61,0.25+0.18,0.61)
#arrow(0.15+2*0.18,0.61,0.25+2*0.18,0.61)
#arrow(0.15+3*0.18,0.61,0.25+3*0.18,0.61)
for k in np.arange(4):
    text(0.16+0.22*k,0.67,0.18+0.22*k,0.67,k)


########## 3c ##########
num_batches=5 # 4 batches, plus one

ax7 = fig.add_axes([0.1,0.05,0.35,0.35])
ax8 = fig.add_axes([0.6,0.05,0.35,0.35])

data = []
file_list = sorted(glob.glob('./pred/[0-9].csv'))
for k,file_path in enumerate(file_list):
    data.append(np.genfromtxt(file_path, delimiter=','))

policies = np.genfromtxt('policies_all.csv', delimiter=',')

isTested = np.zeros(len(policies))

for k, pol in enumerate(policies):
    for batch in data:
        for row in batch:
            if (pol==row[0:4]).all():
                isTested[k] += 1

pol_reps = np.zeros(num_batches)

for k in np.arange(num_batches):
    pol_reps[k] = sum(isTested==k)

ax7.bar(np.arange(num_batches),pol_reps,tick_label = ['0','1','2','3','4'],\
        align='center',color=[0.1,0.4,0.8])

# Add labels to bar plot
all_black_labels = True
if all_black_labels:
    for k, pol_rep in enumerate(pol_reps):
        ax7.text(k, pol_rep+2, str(int(pol_rep)), horizontalalignment='center', fontsize=FS)
        
    ax7.set_ylim([0,130])
else:
    for k, pol_rep in enumerate(pol_reps[:-1]):
        #ax7.text(k, pol_rep+2, str(int(pol_rep)), horizontalalignment='center', fontsize=FS)
        ax7.text(k, pol_rep-7, str(int(pol_rep)), 
                 color='white',horizontalalignment='center', fontsize=FS)
    ax7.text(4, pol_reps[k+1]+2, str(int(pol_reps[k+1])), horizontalalignment='center', fontsize=FS)

ax7.set_xlabel('Repetitions per protocol',fontsize=FS)
ax7.set_ylabel('Count',fontsize=FS)
ax7.set_title('c', loc='left', weight='bold')

########## 3d ##########
# IMPORT RESULTS
# Get folder path containing text files
file = glob.glob('./bounds/4_bounds.pkl')[0]
with open(file, 'rb') as infile:
    policies_temp, ub, lb, mean = pickle.load(infile)

# add cc4 to policies
policies = []
for k, pol in enumerate(policies_temp):
    cc4 = 0.2/(1/6 - (0.2/pol[0] + 0.2/pol[1] + 0.2/pol[2])) # analytical expression for cc4
    policies.append([pol[0],pol[1],pol[2],cc4])
    
policies = np.asarray(policies) # cast to numpy array

## MODELS
slope, intercept, r_value, p_value, std_err = linregress(np.log(np.sum(policies,axis=1)),np.log(mean))
values = np.sum(policies**2,axis=1)
    
xlabel_mod = r'$\mathdefault{sum(I^2)=\Sigma_{i=1}^{4}CC_i}$'
leglabel = 'ρ = {:.2}'.format(pearsonr(values,mean)[0])

ax8.plot(values,mean,'o',label=leglabel,color=[0.1,0.4,0.8])
ax8.set_xlabel(xlabel_mod,fontsize=FS)
ax8.set_ylabel('CLO-estimated cycle life after round 4',fontsize=FS)
ax8.legend(loc='best',markerscale=0,frameon=False)
ax8.set_title('d', loc='left', weight='bold')

#plt.tight_layout()
plt.savefig('fig3_v4.png',bbox_inches='tight')
plt.savefig('fig3_v4.pdf',bbox_inches='tight',format='pdf')
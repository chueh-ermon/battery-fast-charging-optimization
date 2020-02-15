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
ax7 = plt.subplot(337)
ax8 = plt.subplot(338)
ax9 = plt.subplot(339)
ax7.set_title('h', loc='left', weight='bold')
ax8.set_title('i', loc='left', weight='bold')
ax9.set_title('j', loc='left', weight='bold')

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




##############################################################################
# PLOTTING PARAMETERS
batches_to_plot = [0,1,2,3]
labels = ['a','b','c','d']

colormap = 'plasma_r'
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
        ax = fig.add_axes([0.22+0.22*(k-1),0.9,0.3,0.3],projection='3d')
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
    
    ax.set_xlabel('CC1',fontsize=FS), ax.set_xlim([3, 8])
    ax.set_ylabel('CC2',fontsize=FS), ax.set_ylim([3, 8])
    ax.set_zlabel('CC3',fontsize=FS), ax.set_zlim([3, 8])
    #ax.set_title('Before batch '+str(batch_idx))
    ax.set_title(labels[k], loc='left', weight='bold')
    
    ax.view_init(elev=el, azim=az)

# ADD COLORBAR
#plt.tight_layout()
#plt.subplots_adjust(left=0.02,right=0.92)

cbar_ax = fig.add_axes([0.92, 0.9, 0.02, 0.3]) # [left, bottom, width, height]
norm = matplotlib.colors.Normalize(min_lifetime, max_lifetime)
m = plt.cm.ScalarMappable(norm=norm, cmap=colormap)
m.set_array([])

cbar = plt.colorbar(m, cax=cbar_ax)
cbar.ax.tick_params(labelsize=FS,length=0)
cbar.ax.set_title('Predicted\ncycle life',fontsize=FS)





########## 3e-g ##########
##############################################################################
# PLOTTING PARAMETERS
batches_to_plot = [0,2,4]
labels = ['e','f','g']

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
        ax = fig.add_axes([0.31+0.31*(k-1),0.45,0.3,0.3],projection='3d')
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
    
    ax.set_xlabel('CC1',fontsize=FS), ax.set_xlim([3, 8])
    ax.set_ylabel('CC2',fontsize=FS), ax.set_ylim([3, 8])
    ax.set_zlabel('CC3',fontsize=FS), ax.set_zlim([3, 8])
    #ax.set_title('Before batch '+str(batch_idx))
    ax.set_title(labels[k], loc='left', weight='bold')
    
    ax.view_init(elev=el, azim=az)

# ADD COLORBAR
cbar_ax = fig.add_axes([0.92, 0.45, 0.02, 0.3]) # [left, bottom, width, height]
norm = matplotlib.colors.Normalize(min_lifetime, max_lifetime)
m = plt.cm.ScalarMappable(norm=norm, cmap=colormap)
m.set_array([])

cbar = plt.colorbar(m, cax=cbar_ax)
cbar.ax.tick_params(labelsize=FS,length=0)
cbar.ax.set_title('Estimated\ncycle life',fontsize=FS)

########## 3g ##########
num_batches=5 # 4 batches, plus one

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
ax7.set_xlabel('n',fontsize=FS)
ax7.set_ylabel('Number of policies tested n times',fontsize=FS)

########## 3h ##########
# IMPORT RESULTS
# Get pickle files of bounds
file_list = sorted(glob.glob('./bounds/[0-9]_bounds.pkl'))
data = []
for file in file_list:
    with open(file, 'rb') as infile:
        param_space, ub, lb, mean = pickle.load(infile)
        data.append(mean)

## Find number of batches and policies
n_batches  = len(data)
n_policies = len(param_space)

change = np.zeros((n_policies, n_batches-1))
for k, batch in enumerate(data[:-1]):
    change[:,k] = data[k+1] - data[k]

mean_change = np.mean(np.abs(change),axis=0)

## find mean change as a percent
per_change = np.zeros((n_policies, n_batches-1))
for k, batch in enumerate(data[:-1]):
    per_change[:,k] = 100*(data[k+1] - data[k])/data[k+1]

mean_per_change = np.mean(np.abs(per_change),axis=0)

## find mean change for top K policies
top_K_pols_list = [5,10,25,50,224]
mean_change_topk = np.zeros((len(top_K_pols_list),n_batches-1))
mean_per_change_topk = np.zeros((len(top_K_pols_list),n_batches-1))

for k, n in enumerate(top_K_pols_list):
    top_pol_idx = np.argsort(-mean)[0:n]
    mean_change_topk[k,:] = np.mean(np.abs(change[top_pol_idx]),axis=0)
    mean_per_change_topk[k,:] = np.mean(np.abs(per_change[top_pol_idx]),axis=0)

## plot
batches = np.arange(n_batches-1)+1

cm = plt.get_cmap('winter')

legend = ['K = ' + str(k) for k in top_K_pols_list]

ax8.set_color_cycle([cm(1.*i/len(top_K_pols_list)) for i in range(len(top_K_pols_list))])
for i in range(len(top_K_pols_list)-1):
    ax8.plot(batches,mean_per_change_topk[i])
ax8.plot(batches,mean_per_change_topk[i+1],'k')
ax8.set_xlabel('Batch index (change from round i-1 to i)')
ax8.set_ylabel('Mean abs. change in estimated\ncycle life for top K policies (%)')
ax8.set_ylim((0,14))
ax8.set_xticks(np.arange(1, 5, step=1))
ax8.legend(legend,frameon=False)

########## 3i ##########
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

ax9.plot(values,mean,'o',label=leglabel,color=[0.1,0.4,0.8])
ax9.set_xlabel(xlabel_mod,fontsize=FS)
ax9.set_ylabel('OED-estimated cycle life',fontsize=FS)
ax9.legend(loc='best',markerscale=0,frameon=False)


plt.tight_layout()
plt.savefig('fig3_v2.png',bbox_inches='tight')
plt.savefig('fig3_v2.pdf',bbox_inches='tight',format='pdf')
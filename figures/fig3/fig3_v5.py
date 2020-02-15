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
import matplotlib.patheffects as pe

import glob
import pickle

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

FS = 16
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
filename = 'predictions.csv'
pred_data = np.genfromtxt(filename, delimiter=',',skip_header=1)

validation_policies = pred_data[:,0:3]

### I vs SOC

LW = 3
tol = 0.5

# Initialize axis limits
ax8.set_xlim([0,100])
ax8.set_ylim([0,10])

ax8.set_xlabel('State of charge (%)',fontsize=FS)
ax8.set_ylabel('Current (C rate)',fontsize=FS)

ax8.set_xticks(np.arange(0,101,20))

"""
# Add grey lines
C1list = [3.6, 4.0, 4.4, 4.8, 5.2, 5.6, 6.0, 7.0, 8.0]
C2list = [3.6, 4.0, 4.4, 4.8, 5.2, 5.6, 6.0, 7.0]
C3list = [3.6, 4.0, 4.4, 4.8, 5.2, 5.6]

for c1 in C1list:
    ax1.plot([0,20-tol], [c1,c1], linewidth=2, color='grey')
for c2 in C2list:
    ax1.plot([20+tol,40-tol],[c2,c2], linewidth=2, color='grey')
for c3 in C3list:
    ax1.plot([40+tol,60-tol],[c3,c3], linewidth=2, color='grey')
"""
    
# Add bands
ax8.axvspan(0,      20-tol/2, ymin=0.36, ymax=0.8,  facecolor='red', alpha=0.25, edgecolor=None)
ax8.axvspan(20+tol/2, 40-tol/2, ymin=0.36, ymax=0.7,  facecolor='red', alpha=0.25, edgecolor=None)
ax8.axvspan(40+tol/2, 60-tol/2, ymin=0.36, ymax=0.56, facecolor='red', alpha=0.25, edgecolor=None)
ax8.axvspan(60+tol/2, 80-tol/2, ymin=0,    ymax=0.48, facecolor='blue', alpha=0.25, edgecolor=None)

# Add 1C charging
ax8.plot([80,89],[1,1], linewidth=LW, color='black')
x = np.linspace(89,100,100)
y = np.exp(-0.5*(x-89))
ax8.plot(x,y, linewidth=LW, color='black')

# Plot protocols
idx_subset = [2,3,1]
valpol_subset = validation_policies[idx_subset,:]
color_subset  = ['rebeccapurple','darkviolet','violet']
styles        = ['-','--','-.']
paths         = [4,4,4]
labels        = ['CLO 1: 4.8C-5.2C-5.2C-4.160C',
                 'CLO 2: 5.2C-5.2C-4.8C-4.160C',
                 'CLO 3: 4.4C-5.6C-5.2C-4.252C']

for k, pol in enumerate(valpol_subset):
    CC1 = pol[0]
    CC2 = pol[1]
    CC3 = pol[2]
    CC4 = 0.2/(1/6 - (0.2/CC1 + 0.2/CC2 + 0.2/CC3))
    
    c = color_subset[k]
    s = styles[k]
    l = labels[k]
    
    CC = [CC2,CC3,CC4]
    
    # Plot
    ax8.plot([0,20-tol], [CC1,CC1], linewidth=LW, color=c, ls=s,label=l,
             path_effects=[pe.Stroke(linewidth=paths[k], foreground='k'), pe.Normal()])
    for k2, CCstep in enumerate(CC):
        ax8.plot([20*(k2+1)+tol,20*(k2+2)-tol],[CCstep,CCstep], linewidth=LW, color=c, ls=s,
                 path_effects=[pe.Stroke(linewidth=paths[k], foreground='k'), pe.Normal()])
        
        """
        ax1.plot([0,20-tol], [CC1,CC1], linewidth=LW, color=c, ls=s,label=l,
                 path_effects=[pe.Stroke(linewidth=4, foreground='k'), pe.Normal()])
        ax1.plot([20+tol,40-tol],[CC2,CC2], linewidth=LW, color=c, ls=s,
                 path_effects=[pe.Stroke(linewidth=4, foreground='k'), pe.Normal()])
        ax1.plot([40+tol,60-tol],[CC3,CC3], linewidth=LW, color=c, ls=s,
                 path_effects=[pe.Stroke(linewidth=4, foreground='k'), pe.Normal()])
        ax1.plot([60+tol,80-tol],[CC4,CC4], linewidth=LW, color=c, ls=s,
                 path_effects=[pe.Stroke(linewidth=4, foreground='k'), pe.Normal()])
        """

ax8.legend(frameon=False)
ax8.set_title('d', loc='left', weight='bold')

#plt.tight_layout()
plt.savefig('fig3_v5.png',bbox_inches='tight')
plt.savefig('fig3_v5.pdf',bbox_inches='tight',format='pdf')
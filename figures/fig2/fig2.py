#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 14 14:02:51 2018

@author: peter
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams

import matplotlib
import matplotlib.cm as cm
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.axes_grid1 import make_axes_locatable
import seaborn as sns

fig = plt.subplots(1,2,figsize=(16,6))
ax1 = plt.subplot(121)
with sns.axes_style('white'):
    ax2 = plt.subplot(122, projection='3d')

FS = 14
LW = 3

rcParams['pdf.fonttype'] = 42
rcParams['ps.fonttype'] = 42
rcParams['font.size'] = FS
rcParams['axes.labelsize'] = FS
rcParams['xtick.labelsize'] = FS
rcParams['ytick.labelsize'] = FS
rcParams['font.sans-serif'] = ['Arial']


########## 2a ##########

# Initialize axis limits
ax1.set_xlabel('State of charge (%)')
ax1.set_ylabel('Current (C rate)')
ax1.set_xlim([0,100])
ax1.set_ylim([0,10])

# Add grey lines
C1list = [3.6, 4.0, 4.4, 4.8, 5.2, 5.6, 6.0, 7.0, 8.0]
C2list = [3.6, 4.0, 4.4, 4.8, 5.2, 5.6, 6.0, 7.0]
C3list = [3.6, 4.0, 4.4, 4.8, 5.2, 5.6]

for c1 in C1list:
    ax1.plot([0,20], [c1,c1], linewidth=2, color='grey')
for c2 in C2list:
    ax1.plot([20,40],[c2,c2], linewidth=2, color='grey')
for c3 in C3list:
    ax1.plot([40,60],[c3,c3], linewidth=2, color='grey')
    
# Add example policy
c1, c2, c3, c4 = 7.0, 4.8, 5.2, 3.45
ax1.plot([0,20], [c1,c1], linewidth=LW, color='red')
ax1.plot([20,40],[c2,c2], linewidth=LW, color='red')
ax1.plot([40,60],[c3,c3], linewidth=LW, color='red')
ax1.plot([60,80],[c4,c4], linewidth=LW, color='blue')

# Add bands
ax1.axvspan(0,  20, ymin=0.36, ymax=0.8,  color='red', alpha=0.25)
ax1.axvspan(20, 40, ymin=0.36, ymax=0.7,  color='red', alpha=0.25)
ax1.axvspan(40, 60, ymin=0.36, ymax=0.56, color='red', alpha=0.25)
ax1.axvspan(60, 80, ymin=0,    ymax=0.48, color='blue', alpha=0.25)

# Dotted lines for SOC bands
for k in [2,4,6,8]:
    ax1.plot([k*10,k*10],[0,10], linewidth=2, color='grey', linestyle=':')
    
# CC labels
label_height = 9.2
for k in np.arange(4):
    ax1.text(10+20*k,label_height,'CC'+str(k+1), horizontalalignment='center')
ax1.text(90,label_height,'CC5-CV1', horizontalalignment='center')

# Add 1C charging
ax1.plot([80,89],[1,1], linewidth=LW, color='black')
x = np.linspace(89,100,100)
y = np.exp(-0.5*(x-89))
ax1.plot(x,y, linewidth=LW, color='black')

# Charging time text box
ct_label_height = 0.5
ax1.plot([0.1,0.1],[ct_label_height-0.25,ct_label_height+0.25], linewidth=3, color='grey')
ax1.plot([80,80],[ct_label_height-0.25,ct_label_height+0.25], linewidth=2, color='grey')
ax1.plot([0,80],[ct_label_height,ct_label_height], linewidth=2, color='grey')

textstr = 'Charging time to 80% SOC = 10 minutes'
props = dict(boxstyle='round', facecolor='white', edgecolor='grey',alpha=1,linewidth=2)
ax1.text(0.4, ct_label_height/10, textstr,transform=ax1.transAxes, fontsize=FS,
        verticalalignment='center', horizontalalignment='center',bbox=props)

# Voltage label text box
v_label_height = 8.4
v_label_lines = False
if v_label_lines:
    ax1.plot([0.1,0.1],[v_label_height-0.25,v_label_height+0.25], linewidth=3, color='grey')
    ax1.plot([99.9,99.9],[v_label_height-0.25,v_label_height+0.25], linewidth=3, color='grey')
    ax1.plot([0,100],[v_label_height,v_label_height], linewidth=2, color='grey')

textstr = 'Max voltage = 3.6 V'
props = dict(boxstyle='round', facecolor='white', edgecolor='grey',alpha=1,linewidth=2)
ax1.text(0.5, v_label_height/10, textstr,transform=ax1.transAxes, fontsize=FS,
        verticalalignment='center', horizontalalignment='center',bbox=props)

ax1.set_title('a', loc='left', weight='bold')

########## 2b ##########
##############################################################################
# PLOTTING PARAMETERS
colormap = 'viridis'
el, az = 30, 240
point_size = 70
seed = 0
##############################################################################

# IMPORT DATA
param_space = np.genfromtxt('policies_all.csv', delimiter=',')

CC1 = param_space[:,0]
CC2 = param_space[:,1]
CC3 = param_space[:,2]
CC4 = param_space[:,3]
min_CC4 = np.min(CC4)
max_CC4 = np.max(CC4)

## INITIALIZE PLOT
# SETTINGS
ax2.set_aspect('equal')
ax2.set_title('b',loc='left', weight='bold')

## PLOT POLICIES
with plt.style.context(('classic')):
    plt.set_cmap(colormap)
    ax2.scatter(CC1,CC2,CC3, s=point_size, c=CC4,
               vmin=min_CC4, vmax=max_CC4)
    
    ax2.set_xlabel('CC1',fontsize=FS), ax2.set_xlim([3, 8])
    ax2.set_ylabel('CC2',fontsize=FS), ax2.set_ylim([3, 8])
    ax2.set_zlabel('CC3',fontsize=FS), ax2.set_zlim([3, 8])
    
    ax2.view_init(elev=el, azim=az)
    
    # ADD COLORBAR
    #ax2.subplots_adjust(left=0.01,right=0.85,bottom=0.02,top=0.98,wspace=0.000001)
    cbar_ax = fig[0].add_axes([0.9, 0.15, 0.02, 0.7]) # [left, bottom, width, height]
    norm = matplotlib.colors.Normalize(min_CC4, max_CC4)
    m = plt.cm.ScalarMappable(norm=norm, cmap=colormap)
    m.set_array([])
    
    cbar = plt.colorbar(m, cax=cbar_ax)
    cbar.ax.tick_params(labelsize=FS,length=0)
    cbar.ax.set_title('CC4',fontsize=FS)

#plt.tight_layout()
plt.savefig('fig2.png',bbox_inches='tight')
plt.savefig('fig2.pdf',bbox_inches='tight',format='pdf')
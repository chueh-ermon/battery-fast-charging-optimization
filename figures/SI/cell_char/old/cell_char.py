#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 16 19:42:58 2019

@author: peter
"""

import numpy as np
import glob
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib import rcParams

FS = 10
LW = 3

rcParams['pdf.fonttype'] = 42
rcParams['ps.fonttype'] = 42
rcParams['font.size'] = FS
rcParams['axes.labelsize'] = FS
rcParams['xtick.labelsize'] = FS
rcParams['ytick.labelsize'] = FS
rcParams['font.sans-serif'] = ['Arial']

f, ax = plt.subplots(3, 2, figsize=(6.5,9))

for k in range(6):
    k1 = int(k/2)
    k2 = k%2
    ax[k1][k2].set_title(chr(97+k1*2+k2), loc='left', weight='bold')

########## a,c ##########
file = sorted(glob.glob('2018*.csv'))[1]

# Extract data
data = np.genfromtxt(file,skip_header=True,delimiter=',')
test_time = data[:,1]/3600
step_time = data[:,3]/3600
step_idx = data[:,4]
cycle_idx = data[:,5]
I = data[:,6]/1.1 # C rate
V = data[:,7]
Qc = data[:,8]
Qd = data[:,9]
T = data[:,14]

# Pre-initialize lists
step_idx_list = np.arange(2,40,4)
n_cycles = len(step_idx_list)
t_cycle = []
V_cycle = []
I_cycle = []
Qc_cycle = []
T_cycle = []
I_leg = []

# Extract charge cycles
for k in range(n_cycles):
    step_indices = np.where(step_idx == step_idx_list[k])[0]
    t_cycle.append(test_time[step_indices] - test_time[step_indices[0]])
    V_cycle.append(V[step_indices])
    I_cycle.append(I[step_indices])
    Qc_cycle.append(Qc[step_indices] - Qc[step_indices[0]])
    T_cycle.append(T[step_indices])

    I_leg.append(str(int(np.mean(I_cycle[k]))) + 'C') # find charge C rate

cmap = plt.get_cmap('Reds')
ax[0][0].set_color_cycle([cmap(1.*i/n_cycles) for i in range(n_cycles)])
ax[1][0].set_color_cycle([cmap(1.*i/n_cycles) for i in range(n_cycles)])
    
for k in range(n_cycles):
    ax[0][0].plot(Qc_cycle[k], V_cycle[k],'-')

ax[0][0].legend(I_leg,ncol=2,frameon=False)

ax[0][0].set_yticks(np.arange(2,3.51,0.5))
ax[1][0].set_ylim([30,40])

for k in range(n_cycles):
    ax[1][0].plot(Qc_cycle[k], T_cycle[k],'-')
    
ax[0][0].set_xlabel('Capacity (Ah)')
ax[0][0].set_ylabel('Voltage (V)')
ax[1][0].set_xlabel('Capacity (Ah)')
ax[1][0].set_ylabel('Can temperature (°C)')

########## b,d ##########
file = sorted(glob.glob('2018*.csv'))[0]

# Extract data
data = np.genfromtxt(file,skip_header=True,delimiter=',')
test_time = data[:,1]/3600
step_time = data[:,3]/3600
step_idx = data[:,4]
cycle_idx = data[:,5]
I = data[:,6]/1.1 # C rate
V = data[:,7]
Qc = data[:,8]
Qd = data[:,9]
T = data[:,14]

# Pre-initialize lists
step_idx_list = np.arange(3,40,4)
n_cycles = len(step_idx_list)
t_cycle = []
V_cycle = []
I_cycle = []
Qd_cycle = []
T_cycle = []
I_leg = []

# Extract charge cycles
for k in range(n_cycles):
    step_indices = np.where(step_idx == step_idx_list[k])[0]
    t_cycle.append(test_time[step_indices] - test_time[step_indices[0]])
    V_cycle.append(V[step_indices])
    I_cycle.append(I[step_indices])
    Qd_cycle.append(Qd[step_indices] - Qd[step_indices[0]])
    T_cycle.append(T[step_indices])

for k in range(n_cycles):
    I_leg.append(str(-int(np.round(np.mean(I_cycle[k])))) + 'C') # find discharge C rate
    
cmap = plt.get_cmap('Blues')
ax[0][1].set_color_cycle([cmap(1.*i/n_cycles) for i in range(n_cycles)])
ax[1][1].set_color_cycle([cmap(1.*i/n_cycles) for i in range(n_cycles)])

for k in range(n_cycles):
    ax[0][1].plot(Qd_cycle[k], V_cycle[k],'-')

for k in range(n_cycles):
    ax[1][1].plot(Qd_cycle[k], T_cycle[k],'-')

ax[1][1].legend(I_leg,ncol=2,frameon=False)
ax[0][1].set_yticks(np.arange(2,3.51,0.5))
ax[1][1].set_ylim([30,80])

ax[0][1].set_xlabel('Capacity (Ah)')
ax[0][1].set_ylabel('Voltage (V)')
ax[1][1].set_xlabel('Capacity (Ah)')
ax[1][1].set_ylabel('Can temperature (°C)')

########## e-f ##########
colors = []
colors = cm.viridis(np.linspace(0, 1, 4))
colors = colors[:,0:3]

file_list = sorted(glob.glob('2019*.csv'))

Crates = np.asarray([3.6,4,4.4,4.8,5.2,5.6,6,7,8])
currents = Crates * 1.1
SOCs = ['20% SOC','40% SOC','60% SOC','80% SOC']
SOCs2 = [20,40,60,80]

overpotential = np.zeros((len(file_list),4,9))
R = np.zeros((2,4,2))

for k, file in enumerate(file_list):
    # Extract data
    data = np.genfromtxt(file,skip_header=True,delimiter=',')
    test_time = data[:,1]/3600
    step_time = data[:,3]/3600
    step_idx = data[:,4]
    cycle_idx = data[:,5]
    I = data[:,6]/1.1 # C rate
    V = data[:,7]
    Qc = data[:,8]
    Qd = data[:,9]

    for k2, idx_c in enumerate(np.arange(4)):
        for k3, idx_p in enumerate(np.linspace(4,36,9)):
            idx = np.intersect1d(np.where(step_idx==idx_p),
                                 np.where(cycle_idx==idx_c))
            if idx.size > 0:
                idx = np.insert(idx, 0, idx[0]-1)
                # Calculate potential change during rest period
                overpotential[k][k2][k3] = V[idx[0]] - V[idx[-1]]

    # Plotting
    for k2 in np.arange(4):
        ax[2][k].plot(Crates,overpotential[k][k2],'.-',c=colors[k2,:])
    ax[2][k].set_xlabel('C rate')
    ax[2][k].set_ylabel('Overpotential (V)')
    ax[2][k].legend(SOCs,loc='upper left',frameon=False)
    ax[2][k].set_xlim((3.5,8.1))
    ax[2][k].set_ylim((0.1,0.4))
    for k2 in np.arange(4):
        # V = I*R + V0
        if k2<2:
            R[k][k2][:] = np.polyfit(currents,overpotential[k][k2],1)
        else:
            R[k][k2][:] = np.polyfit(currents[1:],overpotential[k][k2][1:],1)
        ax[2][k].plot(Crates,R[k][k2][0]*currents + R[k][k2][1],'--',c=colors[k2,:])

        ## annotate
        SOC_str = '{}% SOC: η = {:0.3f}I+{:0.3f}'.format(20*(k2+1),R[k][k2][0], R[k][k2][1])
        ax[2][k].annotate(SOC_str,(8,0.185-0.025*k2),ha='right')


# Resistance vs soc
Rmean = np.mean(R, axis=0)
Rstd  =  np.std(R, axis=0)

# Overall average resistance
R_tot = np.mean(Rmean[:,0],axis=0)

plt.tight_layout()
plt.savefig('cell_char.png',bbox_inches='tight')
plt.savefig('cell_char.pdf',bbox_inches='tight',format='pdf')

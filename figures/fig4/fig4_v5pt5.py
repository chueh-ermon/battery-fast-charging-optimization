#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 29 22:09:03 2019

@author: peter
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
from matplotlib.legend_handler import HandlerLine2D, HandlerTuple
import matplotlib.patheffects as pe
import glob
import pickle
from cycler import cycler
from scipy.stats import kendalltau
from scipy.stats import pearsonr
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

plt.close('all')

FS = 12
LW = 3
upper_lim = 1400

rcParams['pdf.fonttype'] = 42
rcParams['ps.fonttype'] = 42
rcParams['font.size'] = FS
rcParams['axes.labelsize'] = FS
rcParams['xtick.labelsize'] = FS
rcParams['ytick.labelsize'] = FS
rcParams['font.sans-serif'] = ['Arial']

########## LOAD DATA ##########
# Load echem
Qn = np.genfromtxt('Qn.csv', delimiter=',',skip_header=0)

# Load predictions
filename = 'predictions.csv'
pred_data = np.genfromtxt(filename, delimiter=',',skip_header=1)

validation_policies = pred_data[:,0:3]
predicted_lifetimes = pred_data[:,3:]

# Load prediction intervals
filename = 'PI_lo.csv'
PI_lo = np.genfromtxt(filename, delimiter=',',skip_header=1)[:,3:]

filename = 'PI_hi.csv'
PI_hi = np.genfromtxt(filename, delimiter=',',skip_header=1)[:,3:]

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

# Rankings calculations
oed_ranks = np.empty_like(oed_means.argsort())
oed_ranks[oed_means.argsort()] = np.arange(len(oed_means))
oed_ranks = np.max(oed_ranks) - oed_ranks + 1 # swap order and use 1-indexing

pred_ranks = np.empty_like(pred_means.argsort())
pred_ranks[pred_means.argsort()] = np.arange(len(pred_means))
pred_ranks = np.max(pred_ranks) - pred_ranks + 1 # swap order and use 1-indexing

final_ranks = np.empty_like(final_means.argsort())
final_ranks[final_means.argsort()] = np.arange(len(final_means))
final_ranks = np.max(final_ranks) - final_ranks + 1 # swap order and use 1-indexing

########## PLOTS ##########

fig = plt.subplots(2,3,figsize=(12,8))
ax1 = plt.subplot2grid((2, 3), (0, 0))
ax2 = plt.subplot2grid((2, 3), (0, 1))
ax3 = plt.subplot2grid((2, 3), (0, 2))
ax4 = plt.subplot2grid((2, 3), (1, 0), colspan=2)
ax5 = plt.subplot2grid((2, 3), (1, 2))

ax1.set_title('a',loc='left', weight='bold')
ax2.set_title('b',loc='left', weight='bold')
ax3.set_title('c',loc='left', weight='bold')
ax4.set_title('d',loc='left', weight='bold')
ax5.set_title('e',loc='left', weight='bold')

default_colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
c1 = default_colors[0]
c2 = default_colors[1]
c3 = default_colors[2]
c4 = default_colors[3]
#custom_cycler = (cycler(color=    [c1 , c2, c2, c2, c3, c1, c1, c1, c4]) +
#                 cycler(marker=   ['o','o','s','v','o','s','v','^','o']) +
custom_cycler = (cycler(color=    [c1, c2, c2, c2, c3, c1, c1, c1, c3]) +
                 cycler(marker=   ['v','^','<','>','o','p','h','8','s']) +
                 cycler(linestyle=['' , '', '', '', '', '', '', '', '']))

#### Q(n)

default_colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
c1 = default_colors[0]
c2 = default_colors[1]
c3 = default_colors[2]
c4 = default_colors[3]

custom_cycler1 = (cycler(color=   [c1, c1, c1, c1, c1,
                                   c2, c2, c2, c2, c2,
                                   c2, c2, c2, c2, c2,
                                   c2, c2, c2, c2, c2,
                                   c3, c3, c3, c3, c3,
                                   c1, c1, c1, c1, c1,
                                   c1, c1, c1, c1, c1,
                                   c1, c1, c1, c1, c1,
                                   c3, c3, c3, c3, c3]) +
                 cycler(marker=   ['v', 'v', 'v', 'v', 'v',
                                   '^', '^', '^', '^', '^',
                                   '<', '<', '<', '<', '<',
                                   '>', '>', '>', '>', '>',
                                   'o', 'o', 'o', 'o', 'o',
                                   'p', 'p', 'p', 'p', 'p',
                                   'h', 'h', 'h', 'h', 'h',
                                   '8', '8', '8', '8', '8', 
                                   's', 's', 's', 's', 's']) +
                 cycler(linestyle=['', '', '', '', '',
                                   '', '', '', '', '',
                                   '', '', '', '', '',
                                   '', '', '', '', '',
                                   '', '', '', '', '',
                                   '', '', '', '', '',
                                   '', '', '', '', '',
                                   '', '', '', '', '',
                                   '', '', '', '', '']))

zorders = [2,2,2,2,2,
           3,3,3,3,3,
           3,3,3,3,3,
           3,3,3,3,3,
           1,1,1,1,1,
           2,2,2,2,2,
           2,2,2,2,2,
           2,2,2,2,2,
           1,1,1,1,1]

## a. Q(n)
ax1.set_prop_cycle(custom_cycler1)
CL = []
for k,row in enumerate(Qn):
    idx = np.where(row>0.88)[0]
    CL.append(len(idx)) # sanity check
    ax1.plot(idx,row[idx],markersize=1.5,zorder=zorders[k])
#ax1.legend(validation_policies)
ax1.set_xlim([0,1500])
ax1.set_ylim([0.88,1.1])
ax1.set_xlabel('Cycle number')
ax1.set_ylabel('Discharge capacity (Ah)')
ax1.set_yticks([0.9,0.95,1.0,1.05,1.1])
ax1.set_yticklabels(['0.90','0.95','1.00','1.05','1.10'])

## b. 
ax2.plot((-100,upper_lim+100),(-100,upper_lim+100), ls='--', c='.3',label='_nolegend_')
ax2.set_prop_cycle(custom_cycler)
p = [None]*9
for k in range(len(pred_means)):
    p[k] = ax2.errorbar(pred_means[k],oed_means[k],xerr=pred_sterr[k])
ax2.set_xlabel('Mean early-predicted cycle life\n(validation)',fontsize=FS)
ax2.set_ylabel('CLO-estimated cycle life',fontsize=FS)
r = pearsonr(oed_means,pred_means)[0]
ax2.set_xlim([0,upper_lim])
ax2.set_ylim([0,upper_lim])
ax2.set_aspect('equal', 'box')
ax2.set_xticks(np.arange(0,1501,250))
ax2.set_yticks(np.arange(0,1501,250)) # consistent with x
ax2.annotate('r = {:.2}'.format(r),(1450,75),horizontalalignment='right')


with_errorbars = False

## c. Lifetimes plot - raw
ax3.plot((-100,upper_lim+100),(-100,upper_lim+100), ls='--', c='.3',label='_nolegend_')
ax3.set_prop_cycle(custom_cycler)
if with_errorbars:
    # 95% prediction interval
    pred_err_lo = predicted_lifetimes - PI_lo
    pred_err_hi = PI_hi - predicted_lifetimes
    
    for protocol in zip(predicted_lifetimes, final_lifetimes, pred_err_lo, pred_err_hi):
        ax3.errorbar(protocol[0],protocol[1],
                 xerr=[protocol[2], protocol[3]], markersize=8)
else:
    ax3.plot(np.transpose(predicted_lifetimes), np.transpose(final_lifetimes),markersize=8)
ax3.set_xlim([0,upper_lim])
ax3.set_ylim([0,upper_lim])
ax3.set_aspect('equal', 'box')
ax3.set_xticks(np.arange(0,1501,250))
ax3.set_yticks(np.arange(0,1501,250)) # consistent with x
ax3.set_xlabel('Early-predicted cycle life (validation)')
ax3.set_ylabel('Final cycle life (validation)')
idx = ~np.isnan(predicted_lifetimes.ravel())
r = pearsonr(predicted_lifetimes.ravel()[idx],final_lifetimes.ravel()[idx])[0]
ax3.annotate('r = {:.2}'.format(r),(1450,75),horizontalalignment='right')


## d. bar plot
final_idx = oed_ranks - 1
final_means_sorted = final_means[final_idx]
final_sterr_sorted = final_sterr[final_idx]
final_lifetimes_sorted = final_lifetimes[final_idx]

colors = [c1, c2, c2, c2, c3, c1, c1, c1, c3]

for k,pol in enumerate(validation_policies):
    if k in [1,5,8]: # legend hack
        ax4.barh(9-final_idx[k],final_means[k],xerr=final_sterr[k], 
                 color=colors[k])
    else:
        ax4.barh(9-final_idx[k],final_means[k],xerr=final_sterr[k], 
                 color=colors[k],label='_nolegend_')
    
    CC1, CC2, CC3 = pol
    CC4 = 0.2/(1/6 - (0.2/CC1 + 0.2/CC2 + 0.2/CC3))
    protocol_life_str = '{0}C-{1}C-{2}C-{3:.3f}C: {4:.0f}'.format(CC1, CC2, CC3, CC4,
           final_means[k])
    if final_idx[k]==0:
        ax4.annotate(protocol_life_str+" cycles",(10,9-final_idx[k]),
                     verticalalignment='center')
    else:
        ax4.annotate(protocol_life_str,(10,9-final_idx[k]),
                     verticalalignment='center')

ax4.set_prop_cycle(custom_cycler)
for k, row in enumerate(final_lifetimes):
    print(k,row)
    ax4.plot(row,(9-final_idx[k])*np.ones((5,1)), markersize=8,
             markeredgecolor='k',label='_nolegend_',)

ax4.set_xlabel('Final cycle life (validation)',fontsize=FS)
ax4.set_ylabel('Validation protocol\n(sorted by CLO ranking)',fontsize=FS)
ax4.set_xlim([0,1200])
ax4.get_yaxis().set_ticks([])
ax4.legend(['CLO top 3', 'Lit-inspired', 'Other'],frameon=False)

def make_legend(ax):
    handles = [(p[1][0], p[2][0], p[3][0]),(p[0][0],p[5][0],p[6][0],p[7][0]),
               (p[4][0],p[8][0])]
    ax.legend(handles,['CLO top 3', 'Lit-inspired', 'Other'],
              numpoints=1, handlelength=2.5, frameon=False,
              handler_map={tuple: HandlerTuple(ndivide=None)})

make_legend(ax1)
make_legend(ax2)
make_legend(ax3)

# Ablation plot
with open('fig4_plot_data.pkl', 'rb') as infile:
        data_dict = pickle.load(infile)

def plot_4c(ax):
    ax.set_prop_cycle(plt.style.library['bmh']['axes.prop_cycle'])
    
    # Dotted line for best protocol
    #ax4.plot([-1300,30000],[np.max(final_means),np.max(final_means)],color='k',linestyle='--',linewidth=3,
    #         label='Cycle life of best protocol')
    
    ## NOTE: We swtiched the x and y axes
    
    ax.errorbar(data_dict['no_oed_no_ep_y'],data_dict['no_oed_no_ep_x'], 
                 yerr=data_dict['no_oed_no_ep_xerr'],xerr=data_dict['no_oed_no_ep_yerr'],
    	alpha=0.8, 
    	linewidth=2, 
    	marker='o', 
    	linestyle=':',
        capsize=4,
    	#color=[0,112/256,184/256], 
    	label='CLO w/o early pred\n + random')
    ax.errorbar(data_dict['oed_no_ep_y'],data_dict['oed_no_ep_x'],
                 yerr=data_dict['oed_no_ep_xerr'],xerr=data_dict['oed_no_ep_yerr'],
        alpha=0.8,
    	linewidth=2,
    	marker='o', 
    	linestyle=':', 
        capsize=4,
        #color=[227/256,86/256,0], 
    	label='CLO w/o early pred\n + MAB')
    ax.errorbar(data_dict['no_oed_ep_y'],data_dict['no_oed_ep_x'], 
                 xerr=data_dict['no_oed_ep_yerr'],
    	alpha=0.8, 
    	linewidth=2, 
    	marker='o', 
    	linestyle=':',
        capsize=4,
        #color=[0,167/256,119/256], 
    	label='CLO w/ early pred\n + random')
    ax.errorbar(data_dict['oed_ep_y'],data_dict['oed_ep_x'],xerr=data_dict['oed_ep_yerr'],
    	alpha=0.8, 
        linewidth=2, 
    	marker='o', 
    	linestyle=':',
        capsize=4,
        #color=[227/256,86/256,0], 
    	label='CLO w/ early pred\n + MAB')
    # plt.xticks(np.arange(max_budget+1))

plot_4c(ax5)
ax5.set_xlim((ax5.get_xlim()[0],ax5.get_xlim()[1]-3))
ax5.set_ylim((-1100, 23500))
xrange = ax5.get_xlim()[1] - ax5.get_xlim()[0]
yrange = ax5.get_ylim()[1] - ax5.get_ylim()[0]
ax5.set_aspect(aspect=xrange/yrange)

ax5.legend(frameon=False)
ax5.set_ylabel('Experimental time (hours)')
ax5.set_xlabel('True cycle life of current best protocol')

# Log inset
add_inset = False
if add_inset:
    ax_ins = inset_axes(ax5, width='100%', height='100%', loc='upper left',
                        bbox_to_anchor=(0.12,0.28,0.5,0.38), bbox_transform=ax5.transAxes)
    plot_4c(ax_ins)
    ax_ins.set_xlim((825,900))
    ax_ins.set_ylim((100,30000))
    ax_ins.set_yscale('symlog')
                   
# tight layout and save
plt.tight_layout()
plt.savefig('fig4_v5pt5.png',bbox_inches='tight')
plt.savefig('fig4_v5pt5.pdf',bbox_inches='tight',format='pdf')

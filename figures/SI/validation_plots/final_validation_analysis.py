#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 29 22:09:03 2019

@author: peter
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
import glob
import pickle
from cycler import cycler
from scipy.stats import kendalltau
from scipy.stats import pearsonr

plt.close('all')

FS = 14
LW = 3

rcParams['pdf.fonttype'] = 42
rcParams['ps.fonttype'] = 42
rcParams['font.size'] = FS
rcParams['axes.labelsize'] = FS
rcParams['xtick.labelsize'] = FS
rcParams['ytick.labelsize'] = FS
rcParams['font.sans-serif'] = ['Arial']
upper_lim = 1500

########## LOAD DATA ##########
# Load predictions
filename = 'predictions.csv'
pred_data = np.genfromtxt(filename, delimiter=',',skip_header=1)

validation_policies = pred_data[:,0:3]
predicted_lifetimes = pred_data[:,3:]

validation_pol_leg = []
for p in validation_policies:
    c4 = 0.2/(1/6 - (0.2/p[0] + 0.2/p[1] + 0.2/p[2]))
    validation_pol_leg.append(str(p[0])+'C-'+str(p[1])+'C-'+str(p[2])+'C-'+'{0:.3f}C'.format(c4))

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

# Standardization
std_mean = np.mean(final_lifetimes)
std_stdev = np.std(final_lifetimes)
std_pred = np.nanstd(final_lifetimes - predicted_lifetimes)
std_sampling = np.mean(np.std(final_lifetimes,axis=1))
eta = np.sqrt(std_pred**2 + std_sampling**2)
print('Standardization mean = ' + str(std_mean))
print('Standardization stdev = '  + str(std_stdev))
print('sigma_pred = '  + str(std_pred))
print('sigma_sampling = '  + str(std_sampling))
print('eta = '  + str(eta))
print()

# calculate early prediction error
rmse = np.sqrt(np.nanmean(((predicted_lifetimes - final_lifetimes) ** 2)))
print('RMSE = ' + str(rmse) + ' cycles')

mape = np.nanmean( np.abs(predicted_lifetimes - final_lifetimes) /final_lifetimes) * 100
print('MAPE = ' + str(mape) + '%')

# calculate bias-corrected early prediction error
predicted_lifetimes_bias_corrected = predicted_lifetimes \
  - np.nanmean(predicted_lifetimes) + np.nanmean(final_lifetimes)
print('Bias correction: ' + str(np.nanmean(final_lifetimes) - \
                                np.nanmean(predicted_lifetimes)) + ' cycles')
rmse = np.sqrt(np.nanmean(((predicted_lifetimes_bias_corrected - final_lifetimes) ** 2)))
print('RMSE = ' + str(rmse) + ' cycles (bias-corrected)')

mape = np.nanmean( np.abs(predicted_lifetimes_bias_corrected - final_lifetimes) /final_lifetimes) * 100
print('MAPE = ' + str(mape) + '% (bias-corrected)')

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

fig = plt.figure(figsize=(11,10))

default_colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
c1 = default_colors[0]
c2 = default_colors[1]
c3 = default_colors[2]
c4 = default_colors[3]
custom_cycler = (cycler(color=    [c1 , c2, c2, c2, c3, c1, c1, c1, c3]) +
                 cycler(marker=   ['v','^','<','>','o','p','h','8','s']) +
                 cycler(linestyle=['' , '', '', '', '', '', '', '', '']))

def init_plot(ax):
    ax.plot((-100,upper_lim+100),(-100,upper_lim+100), ls='--', c='.3',label='_nolegend_')
    ax.set_prop_cycle(custom_cycler)

def format_lifetimes_plot(r):
    plt.xlim([0,upper_lim])
    plt.ylim([0,upper_lim])
    ax0.set_aspect('equal', 'box')
    ax0.set_xticks([0,750,1500])
    ax0.set_yticks([0,750,1500])
    #plt.legend(validation_pol_leg, bbox_to_anchor=(1.01, 0.85))
    plt.annotate('r = {:.2}'.format(r),(75,1250))

def format_rankings_plot(tau):
    plt.xlim([0,10])
    plt.ylim([0,10])
    ax0.set_yticks([0,5,10]) # consistent with x
    ax0.set_aspect('equal', 'box')
    #plt.legend(validation_pol_leg, bbox_to_anchor=(1.01, 0.85))
    plt.annotate('τ = {:.2}'.format(tau),(0.5,8.33))

ax0 = plt.subplot(3,3,1)
#### PRED vs OED
## Lifetimes plot
init_plot(ax0)
for k in range(len(pred_means)):
    ax0.errorbar(pred_means[k],oed_means[k],xerr=pred_sterr[k])
plt.xlabel('Mean early-predicted cycle life\n(validation)',fontsize=FS)
plt.ylabel('CLO-estimated cycle life',fontsize=FS)
plt.title(chr(97),loc='left', weight='bold',fontsize=FS)
r = pearsonr(pred_means,oed_means)[0]
format_lifetimes_plot(r)

## Individual lifetimes plot
ax0 = plt.subplot(3,3,2)
init_plot(ax0)
for k in range(len(predicted_lifetimes)):
    predicted_lifetimes_vector = predicted_lifetimes[k][~np.isnan(predicted_lifetimes[k])]
    oed_value_vector = oed_means[k]*np.ones(np.count_nonzero(~np.isnan(predicted_lifetimes[k])))
    ax0.plot(predicted_lifetimes_vector, oed_value_vector)
plt.xlabel('Early-predicted cycle life\n(validation)',fontsize=FS)
plt.ylabel('CLO-estimated cycle life',fontsize=FS)
plt.title(chr(98),loc='left', weight='bold',fontsize=FS)
idx = ~np.isnan(predicted_lifetimes.ravel())
r = pearsonr(predicted_lifetimes.ravel()[idx],np.repeat(oed_means,5)[idx])[0]
format_lifetimes_plot(r)

## Rankings plot
ax0 = plt.subplot(3,3,3)
init_plot(ax0)
for k in range(len(pred_ranks)):
    plt.plot(pred_ranks[k],oed_ranks[k])
plt.xlabel('Mean early-predicted ranking\n(validation)',fontsize=FS)
plt.ylabel('CLO-estimated ranking',fontsize=FS)
plt.title(chr(99),loc='left', weight='bold',fontsize=FS)
tau = kendalltau(pred_ranks,oed_ranks)[0]
format_rankings_plot(tau)

#### PRED vs FINAL
## Lifetimes plot
ax0 = plt.subplot(3,3,4)
init_plot(ax0)
for k in range(len(pred_means)):
    plt.errorbar(pred_means[k],final_means[k],xerr=pred_sterr[k],yerr=final_sterr[k])
plt.xlabel('Mean early-predicted cycle life\n(validation)',fontsize=FS)
plt.ylabel('Mean final cycle life\n(validation)',fontsize=FS)
plt.title(chr(100),loc='left', weight='bold',fontsize=FS)
r = pearsonr(pred_means,final_means)[0]
format_lifetimes_plot(r)

## Lifetimes plot - raw
ax0 = plt.subplot(3,3,5)
init_plot(ax0)
ax0.plot(np.transpose(predicted_lifetimes), np.transpose(final_lifetimes))
plt.xlabel('Early-predicted cycle life\n(validation)',fontsize=FS)
plt.ylabel('Final cycle life validation)',fontsize=FS)
plt.title(chr(101),loc='left', weight='bold',fontsize=FS)
r = pearsonr(predicted_lifetimes.ravel()[idx],final_lifetimes.ravel()[idx])[0]
format_lifetimes_plot(r)

## Rankings plot
ax0 = plt.subplot(3,3,6)
init_plot(ax0)
for k in range(len(pred_ranks)):
    plt.plot(pred_ranks[k],final_ranks[k])
plt.xlabel('Mean early-predicted ranking\n(validation)',fontsize=FS)
plt.ylabel('Final ranking (validation)',fontsize=FS)
plt.title(chr(102),loc='left', weight='bold',fontsize=FS)
tau = kendalltau(pred_ranks,final_ranks)[0]
format_rankings_plot(tau)

### Lifetimes plot - raw, bias-corrected
#fig, ax0 = plt.subplots()
#ax0.set_prop_cycle(custom_cycler)
#ax0.plot(np.transpose(predicted_lifetimes_bias_corrected), \
#                np.transpose(final_lifetimes))
#plt.legend(validation_policies)
#plt.xlim([0,upper_lim])
#plt.ylim([0,upper_lim])
#ax0.set_aspect('equal', 'box')
#ax0.plot(ax0.get_xlim(), ax0.get_ylim(), ls='--', c='.3')
##ax0.plot([100,100], ax0.get_ylim(), ls="--", c='r')
#plt.xlabel('Predicted cycle life')
#plt.ylabel('Observed cycle life')

#### OED vs FINAL
## Lifetimes plot
ax0 = plt.subplot(3,3,7)
init_plot(ax0)
for k in range(len(oed_means)):
    plt.errorbar(oed_means[k],final_means[k],yerr=final_sterr[k])
plt.xlabel('CLO-estimated cycle life',fontsize=FS)
plt.ylabel('Mean final cycle life\n(validation)',fontsize=FS)
plt.title(chr(103),loc='left', weight='bold',fontsize=FS)
r = pearsonr(oed_means,final_means)[0]
format_lifetimes_plot(r)

## Individual lifetimes plot
ax0 = plt.subplot(3,3,8)
init_plot(ax0)
for k in range(len(predicted_lifetimes)):
    final_lifetimes_vector = final_lifetimes[k][~np.isnan(final_lifetimes[k])]
    oed_value_vector = oed_means[k]*np.ones(np.count_nonzero(~np.isnan(final_lifetimes[k])))
    ax0.plot(oed_value_vector, final_lifetimes_vector)
plt.xlabel('CLO-estimated cycle life',fontsize=FS)
plt.ylabel('Final cycle life (validation)',fontsize=FS)
plt.title(chr(104),loc='left', weight='bold',fontsize=FS)
r = pearsonr(np.repeat(oed_means,5),final_lifetimes.ravel())[0]
format_lifetimes_plot(r)

## Rankings plot
ax0 = plt.subplot(3,3,9)
init_plot(ax0)
for k in range(len(pred_ranks)):
    plt.plot(oed_ranks[k],final_ranks[k])
plt.xlabel('CLO-estimated ranking',fontsize=FS)
plt.ylabel('Final ranking (validation)',fontsize=FS)
plt.title(chr(105),loc='left', weight='bold',fontsize=FS)
tau = kendalltau(oed_ranks,final_ranks)[0]
format_rankings_plot(tau)

# Adjust and legend
plt.tight_layout()
ax0 = plt.subplot(3,3,5)
#plt.legend(validation_pol_leg,loc=9, bbox_to_anchor=(3.8, 1.2),frameon=True)
plt.legend(validation_pol_leg,loc=9, bbox_to_anchor=(0.5, -1.9),frameon=True,ncol=3)
plt.savefig('validation_ablation.png',bbox_inches='tight')
plt.savefig('validation_ablation.pdf',bbox_inches='tight',format='pdf')
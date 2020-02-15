#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May  2 10:04:25 2019

@author: peter
"""

import pandas as pd
import numpy as np
from sklearn import linear_model
import matplotlib.pyplot as plt

## Load fitted data
# Data was manually extracted from Smith, Dahn Fig 7
# using via https://automeris.io/WebPlotDigitizer/

xlsx = pd.ExcelFile('DahnFig7.xlsx')
T30 = pd.read_excel(xlsx, '30 C')
T40 = pd.read_excel(xlsx, '40 C')
T50 = pd.read_excel(xlsx, '50 C')

## Generate linear fits of QSEI = k*t^0.5 + b
T = np.asarray([30,40,50])+273.15
invT = np.transpose(1/T)
k = 8.617e-5
m = np.zeros((3,1))
b = np.zeros((3,1))

reg = linear_model.LinearRegression()
reg.fit(T30[['t^0.5']], T30['QSEI'])
m[0], b[0] = reg.coef_, reg.intercept_

reg.fit(T40[['t^0.5']], T40['QSEI'])
m[1], b[1] = reg.coef_, reg.intercept_

reg.fit(T50[['t^0.5']], T50['QSEI'])
m[2], b[2] = reg.coef_, reg.intercept_

# Calculate activation energy from log(k) vs 1/T
logk = np.log(m)
reg.fit(invT.reshape(-1,1),logk)
slope = reg.coef_
Ea = -slope[0][0]*k

# Plot/display results
plt.plot(invT,logk,'o')
plt.xlabel('1/T (1/K)')
plt.ylabel('log(k)')
plt.title('Ea = {0:.3f} eV'.format(Ea))
plt.savefig('Dahn_Ea.png')
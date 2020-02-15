#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
NEW 4-STEP STRUCTURE
policies.py is a script that generates all policies within a
4-step policy space based on your specifications.
Each step has a width of 20% SOC. C1, C2, and C3 are free,
and C4 is constrained.
It saves a file called policies.csv to the current directory,
where each row is a charging policy
"""

import os
import sys
import numpy as np
import surface_points

##############################################################################

# PARAMETERS TO CREATE POLICY SPACE
C1 = [3.6, 4.0, 4.4, 4.8, 5.2, 5.6, 6.0, 7.0, 8.0]
C2 = [3.6, 4.0, 4.4, 4.8, 5.2, 5.6, 6.0, 7.0]
C3 = [3.6, 4.0, 4.4, 4.8, 5.2, 5.6]

C4_LIMITS = [0.1, 4.81] # Lower and upper limits specifying valid C4s

DIR = sys.argv[1]
FILENAME = sys.argv[2]

##############################################################################

# Pre-initialize arrays and counters
policies = -1.*np.ones((1000,4));
valid_policies = -1.*np.ones((1000,4));

count = valid_count = 0

# Generate policies
for c1, c2, c3 in [(c1,c2,c3) for c1 in C1 for c2 in C2 for c3 in C3]:
    c4 = 0.2/(1/6 - (0.2/c1 + 0.2/c2 + 0.2/c3))
    policies[count,:] = [c1, c2, c3, c4]
    count += 1
    
    if c4 >= C4_LIMITS[0] and c4 <= C4_LIMITS[1]:
        if c1 == 4.8 and c2 == 4.8 and c3 == 4.8:
            print('baseline') # Exclude baseline
        else:
            valid_policies[valid_count,:] = [c1, c2, c3, c4]
            valid_count += 1

# Remove trailing zeros
policies = policies[0:count]
valid_policies = valid_policies[0:valid_count]

# Update user
print('Count = ' + str(count))
print('Valid count = ' + str(valid_count))

## Save policies
np.savetxt(os.path.join(DIR, 'policies_' + FILENAME + '.csv'), valid_policies, delimiter=',', fmt='%1.3f')

surface_points.plot_surface(C1, C2, C3, C4_LIMITS, DIR, FILENAME)

# -*- coding: utf-8 -*-
"""
Created on Thu Jul  1 21:04:03 2021

@author: jackm
"""

import numpy as np
a = np.array([0,1,2,3,2,2,2,0])
uni_a = np.unique(a)
#for i, u in enumerate(uni_a):
#    print('\n\n{}:\n'.format(u, np.where(a==uni_a[i])))
print(np.where(a==uni_a[2]))
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  7 14:55:51 2020

@author: sony
"""

import pandas as pd
import numpy as np 

from scipy import stats

distribution = np.random.randint(5,size=100)
stats.kurtosis(distribution)
stats.skew(distribution)


df = pd.read_csv('grades.csv')
early = df[df['assignment1_submission'] <= '2015-12-31']
late = df[df['assignment1_submission'] > '2015-12-31']

early.mean()
late.mean()

from scipy import stats
stats.ttest_ind?

stats.ttest_ind(early['assignment1_grade'],late['assignment1_grade'],nan_policy= 'omit')
'''
pvalue = 0.16148
So we can't reject the null hypothesis
'''






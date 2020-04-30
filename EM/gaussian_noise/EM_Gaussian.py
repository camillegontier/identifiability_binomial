# -*- coding: utf-8 -*-
"""
Created on Tue Apr 16 13:54:30 2019

@author: gontier
"""

# Computes the mean and the std of the data
# INPUT : vector of EPSCs
# OUTPUT : its mean and std

import numpy as np

def EM_Gaussian(EPSC):
    
    mu_hat = np.mean(EPSC)
    sigma_hat = np.std(EPSC)
    
    return mu_hat,sigma_hat


# -*- coding: utf-8 -*-
"""
Created on Tue Apr 16 11:59:52 2019

@author: gontier
"""

# Generates T data points from a binomial model

import numpy as np

def generate_data_binomial(N,p,q,sigma,T):
        
    # Measurement noise
    epsilon = np.random.normal(0,sigma,T)

    # Values initilization
    EPSC = np.zeros(T)    # EPSC
    k = np.zeros(T) # Number of released vesicles

    # Loop over time
    for i in range(T):
        k[i] = np.random.binomial(N,p)
        EPSC[i] = q*k[i] + epsilon[i]
        
    return EPSC


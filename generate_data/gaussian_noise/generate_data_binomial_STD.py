# -*- coding: utf-8 -*-
"""
Created on Sun Nov  3 18:00:19 2019

@author: gontier
"""

# Generates EPSCs from a given set of parameters and ISI
# INPUT : values for N, p, q, sigma, tauD, and vector of ISI
# (convention : delta_t[0] = 0, delta_t[i] = t_i - t_{i-1})
# OUTPUT : vector of EPSCs

import numpy as np

def generate_data_binomial_tau_D(N,p,q,sigma,tau_D,delta_t):
    
    T = delta_t.size
    # Measurement noise
    epsilon = np.random.normal(0,sigma,T)

    # Values initilization
    EPSC = np.zeros(T)    # EPSC
    k = np.zeros(T) # Number of released vesicles
    n = np.zeros(T) # Number of available vesicles
    I = np.zeros(T) # Refill probability
    n[0] = N
    
    # Loop over time
    for i in range(0,T-1):
        k[i]    = np.random.binomial(n[i],p)
        EPSC[i]    = q*k[i] + epsilon[i]
        I[i+1] = 1-np.exp(-delta_t[i+1]/tau_D)
        n[i+1]  = n[i] - k[i] + np.random.binomial(N - (n[i]-k[i]),I[i+1])
        
    k[-1] = np.random.binomial(n[-1],p)
    EPSC[-1] = q*k[-1] + epsilon[-1]
        
    return EPSC
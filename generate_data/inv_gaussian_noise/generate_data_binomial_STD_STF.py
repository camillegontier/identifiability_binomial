# -*- coding: utf-8 -*-
"""
Created on Tue Aug 27 09:03:13 2019

@author: gontier
"""

# Generates EPSCs from a given set of parameters and ISI
# INPUT : values for N, p, q, sigma, tauD, and tauF, and vector of ISI
# (convention : delta_t[0] = 0, delta_t[i] = t_i - t_{i-1})
# OUTPUT : vector of EPSCs

import numpy as np

def generate_data_binomial_tau_D_tau_F(N,p,q,sigma,tau_D,tau_F,delta_t):
    
    T = delta_t.size

    # Values initilization
    EPSC = np.zeros(T)    # EPSC
    k = np.zeros(T) # Number of released vesicles
    n = np.zeros(T) # Number of available vesicles
    u = np.zeros(T) # Release probability
    I = np.zeros(T) # Refill probability
    n[0] = N
    u[0] = p
    
    # Loop over time
    for i in range(0,T-1):
        k[i]    = np.random.binomial(n[i],u[i])
        EPSC[i] = 0
        if k[i] > 0:
            EPSC[i]    = np.random.wald(k[i]*q,k[i]**2*q**3/sigma**2)
        I[i+1] = 1-np.exp(-delta_t[i+1]/tau_D)
        n[i+1]  = n[i] - k[i] + np.random.binomial(N - (n[i]-k[i]),I[i+1])
        u[i+1] = p + u[i]*(1-p)*np.exp(-delta_t[i+1]/tau_F)
        
    k[-1] = np.random.binomial(n[-1],u[-1])
    EPSC[-1] = 0
    if k[-1] > 0:
        EPSC[-1] = np.random.wald(k[-1]*q,k[-1]**2*q**3/sigma**2)
        
    return EPSC
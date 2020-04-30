# -*- coding: utf-8 -*-
"""
Created on Tue Apr 16 14:30:21 2019

@author: gontier
"""
# Computes the log likelihood of data under model M_1
# INPUT : vector of EPSCs, values for N, p, q, sigma
# OUTPUT : log p(EPSC|theta)

import scipy.stats
import numpy as np

def ll_binomial(EPSP,N_hat,p_hat,q_hat,sigma_hat):
    T = EPSP.size
    
    res = 0
    for i in range(T):
        prob = 0
        for k in range(N_hat + 1):
            if k ==0:
                prob = prob + float(EPSP[i]==0)*scipy.stats.binom.pmf(k,N_hat,p_hat)
            elif EPSP[i]!=0:
                prob = prob + (q_hat**(3/2)*k/np.sqrt(2*np.pi*sigma_hat**2*EPSP[i]**3))*np.exp(-(q_hat*(EPSP[i]-q_hat*k)**2)/(2*sigma_hat**2*EPSP[i]))*scipy.stats.binom.pmf(k,N_hat,p_hat)
        res = res + np.log(prob)
    
    return res


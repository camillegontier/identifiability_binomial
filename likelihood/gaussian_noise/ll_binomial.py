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
        
    return sum(np.log(sum(scipy.stats.norm.pdf(EPSP[i],loc=q_hat*k,scale=sigma_hat)*scipy.stats.binom.pmf(k,N_hat,p_hat) for k in range(N_hat+1))) for i in range(T))


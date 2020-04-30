# -*- coding: utf-8 -*-
"""
Created on Mon Jul 29 14:57:13 2019

@author: gontier
"""

# Computes the log-likelihood of data under a Gaussian distribution
# INPUT : vector of EPSCs, values for theta = (mu, sigma)
# OUTPUT : log p(EPSC|theta)

import scipy.stats
import numpy as np

def ll_gaussian(EPSP,mu_hat,sigma_hat):
    T = EPSP.size       
    return sum(np.log(scipy.stats.norm.pdf(EPSP[i],loc=mu_hat,scale=sigma_hat)) for i in range(T))
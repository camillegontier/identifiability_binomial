# -*- coding: utf-8 -*-
"""
Created on Sun Nov  3 18:38:12 2019

@author: gontier
"""

# Computes the log likelihood of data under model M_2
# INPUT : vector of EPSCs, values for N, p, q, sigma, and tauD, and vector of ISI
# (convention : delta_t[0] = 0, delta_t[i] = t_i - t_{i-1})
# OUTPUT : log p(EPSC|theta)

import numpy as np
import scipy.stats

def ll_binomial_tau_D(EPSP,N,p,q,sigma,tau_D,delta_t):
    T = EPSP.size
    
    # Uses the Baum-Welch algorithm (https://en.wikipedia.org/wiki/Baum%E2%80%93Welch_algorithm)
    A = np.zeros([N+1,N+1,N+1,N+1,T])
        #A[n_i,k_i,n_i+1,k_i+1,i+1] : probability to transit from i to i+1
    for n1 in range(N+1):
        for k1 in range(n1+1):
            for n2 in range(N+1):
                for k2 in range(n2+1):
                    for i in range(T-1):
                        A[n1,k1,n2,k2,i+1] = scipy.stats.binom.pmf(k2,n2,p)*scipy.stats.binom.pmf(n2-n1+k1,N-n1+k1,1-np.exp(-delta_t[i+1]/tau_D))
                             
                            
    B = np.zeros([N+1,N+1,T])
    for n in range(N+1):
        for k in range(n+1):
            for i in range(T):
                if k == 0:
                    B[n,k,i] = float(EPSP[i] == 0)
                elif EPSP[i] != 0:
                    B[n,k,i] = (q**(3/2)*k/np.sqrt(2*np.pi*sigma**2*EPSP[i]**3))*np.exp(-(q*(EPSP[i]-q*k)**2)/(2*sigma**2*EPSP[i]))

    alpha = np.zeros([N+1,N+1,T]) #n,k,i

    for k in range(N+1):    
        if k == 0:
            alpha[N,k,0] = scipy.stats.binom.pmf(k,N,p)*float(EPSP[0] == 0)
        elif EPSP[0] != 0:
            alpha[N,k,0] = scipy.stats.binom.pmf(k,N,p)*(q**(3/2)*k/np.sqrt(2*np.pi*sigma**2*EPSP[0]**3))*np.exp(-(q*(EPSP[0]-q*k)**2)/(2*sigma**2*EPSP[0]))

                
    for i in range(T-1):
        for n in range(N+1):
            for k in range(n+1):
                alpha[n,k,i+1] = B[n,k,i+1]*sum(alpha[nn,kk,i]*A[nn,kk,n,k,i+1] for kk in range(N+1) for nn in range (N+1))
         
    return np.log(np.sum(alpha,(0,1))[-1])
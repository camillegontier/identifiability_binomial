# -*- coding: utf-8 -*-
"""
Created on Tue Apr 16 13:54:30 2019

@author: gontier
"""
# Computes inferred values for p, q, and sigma
# INPUT : vector of EPSCs, and initial values for N, p, q, and sigma
# OUTPUT : MLE for p, q, and sigma given N

import numpy as np
import scipy.stats
import warnings
import scipy.integrate

def EM_Binomial(e,N,p_init,q_init,sigma_init):
    
    p_new = p_init
    sigma_new = sigma_init
    q_new = q_init

    tol = 1e-4
    T = e.size # Number of data points
    max_it = 1000 # Maximum number of iterations
    
    # Updates theta until convergence or max_it is reached
    for it in range(max_it):
     
        # Computes theta_{t+1}
        p_old = p_new
        sigma_old = sigma_new 
        q_old = q_new
        
        # E step
        # Uses the Baum-Welch algorithm (https://en.wikipedia.org/wiki/Baum%E2%80%93Welch_algorithm)
    
        p_k_e = np.zeros([N+1,T])
        p_e = np.zeros(T)
      
        for kk in range(N+1):
            for ii in range(T):
                p_k_e[kk,ii] = scipy.stats.norm.pdf(e[ii],loc=q_old*kk,scale=sigma_old)*scipy.stats.binom.pmf(kk,N,p_old)
        
        p_e = np.sum(p_k_e,axis=0)
        
        # M step
    
        # Computation of q_new
        num = np.sum(np.sum(e[i]*ki*p_k_e[ki,i]/p_e[i] for ki in range(N+1)) for i in range(T))
        den = np.sum(np.sum(ki*ki*p_k_e[ki,i]/p_e[i] for ki in range(N+1)) for i in range(T))   
        q_new = num/den
         
        # Computation of sigma_new
        num = np.sum(np.sum((e[i]-q_new*ki)**2*p_k_e[ki,i]/p_e[i] for ki in range(N+1)) for i in range(T))        
        sigma_new = np.sqrt(num/T)
        
        # Computation of p_new
        num = np.sum(np.sum(ki*p_k_e[ki,i]/p_e[i] for ki in range(N+1)) for i in range(T))
        p_new = num/(T*N)
        
        if abs(sigma_new-sigma_old)/sigma_old < tol and abs(p_new-p_old)/p_old < tol and abs(q_new-q_old)/q_old < tol:
            print('Convergence reached in ' + str(it) + ' iterations')
            print('Sigma_hat = ' + str(sigma_new))
            print('p_hat = ' + str(p_new))
            print('q_hat = ' + str(q_new))
            print('N_hat = ' + str(N))
            
            break
        
        if it == max_it:
            warnings.warn('Maximum number of iterations reached')
            
        if np.isnan(sigma_new) or np.isnan(p_new) or np.isnan(q_new):
            warnings.warn('NaN value')
            break
        
    sigma = sigma_new
    p = p_new
    q = q_new
    
    return p,q,sigma

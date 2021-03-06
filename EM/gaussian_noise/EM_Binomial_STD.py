# -*- coding: utf-8 -*-
"""
Created on Sat Dec 14 19:11:14 2019

@author: gontier
"""

# Computes inferred values for p, q, sigma, and tau_D
# INPUT : vector of EPSCs, initial values for N, p, q, sigma, and tauD, and vector of ISI
# (convention : delta_t[0] = 0, delta_t[i] = t_i - t_{i-1})
# OUTPUT : MLE for p, q, sigma, and tauD given N

import numpy as np
import scipy.stats
import warnings
import scipy.integrate
from scipy import optimize

def EM_Binomial_STD(e,N,p_init,q_init,sigma_init,tauD_init,delta_t):
    
    p_new = p_init
    sigma_new = sigma_init
    q_new = q_init
    tauD_new = tauD_init

    tol = 1e-4
    T = e.size # Number of data points
    max_it = 1000 # Maximum number of iterations
    
    # Updates theta until convergence or max_it is reached
    for it in range(max_it):
     
        # Computes theta_{t+1}
        p_old = p_new
        sigma_old = sigma_new 
        q_old = q_new
        tauD_old = tauD_new
        
        # E step
        # Uses the Baum-Welch algorithm (https://en.wikipedia.org/wiki/Baum%E2%80%93Welch_algorithm)
        
        # Probability of hidden variables conditionned on observation
        p_n_k_e = np.zeros([N+1,N+1,T]) # Probability of having n_i and k_i knowing e
        p_n_n_1_k_1_e = np.zeros([N+1,N+1,N+1,T]) # Probability of having n_i n_i-1 and k_i-1 knowing e
    
        A = np.zeros([N+1,N+1,N+1,N+1,T])
        #A[n_i,k_i,n_i+1,k_i+1,i+1] : probability to transit from i to i+1
        for n1 in range(N+1):
            for k1 in range(n1+1):
                for n2 in range(N+1):
                    for k2 in range(n2+1):
                        for i in range(T-1):
                            A[n1,k1,n2,k2,i+1] = scipy.stats.binom.pmf(k2,n2,p_old)*scipy.stats.binom.pmf(n2-n1+k1,N-n1+k1,1-np.exp(-delta_t[i+1]/tauD_old))
                                                         
        B = np.zeros([N+1,N+1,T])
        for n in range(N+1):
            for k in range(n+1):
                for i in range(T):
                    B[n,k,i] = scipy.stats.norm.pdf(e[i],loc=q_old*k,scale=sigma_old)

        alpha = np.zeros([N+1,N+1,T]) #n,k,i
        beta = np.zeros([N+1,N+1,T])
        
        beta[:,:,-1] = 1
        for k in range(N+1):            
            alpha[N,k,0] = scipy.stats.binom.pmf(k,N,p_old)*scipy.stats.norm.pdf(e[0],loc=q_old*k,scale=sigma_old)
                
        for i in range(T-1):
            for n in range(N+1):
                for k in range(n+1):
                    alpha[n,k,i+1] = B[n,k,i+1]*sum(alpha[nn,kk,i]*A[nn,kk,n,k,i+1] for kk in range(N+1) for nn in range (N+1))

        for i in range(T-2,-1,-1):
            for n in range(N+1):
                for k in range(n+1): 
                    beta[n,k,i] = sum(beta[nn,kk,i+1]*A[n,k,nn,kk,i+1]*B[nn,kk,i+1] for kk in range(N+1) for nn in range(N+1))
              
        for n in range(N+1):
            for k in range(n+1):  
                for i in range(T):                    
                    p_n_k_e[n,k,i] = alpha[n,k,i]*beta[n,k,i]/sum(alpha[nn,kk,i]*beta[nn,kk,i] for kk in range(N+1) for nn in range(N+1))
        
        for n in range(N+1):
            for n_1 in range(N+1):
                for k_1 in range(n_1+1):
                    for i in range(1,T):
                        p_n_n_1_k_1_e[n,n_1,k_1,i] = sum(alpha[n_1,k_1,i-1]*A[n_1,k_1,n,k,i]*B[n,k,i]*beta[n,k,i]/sum(alpha[nn,kk,i]*beta[nn,kk,i] for nn in range(N+1) for kk in range(N+1)) for k in range(n+1))
               
        # M step        
        
        # Computation of q_new
        num = np.sum(np.sum(e[i]*ki*p_n_k_e[ni,ki,i] for ni in range(N+1) for ki in range(N+1)) for i in range(T))
        den = np.sum(np.sum(ki*ki*p_n_k_e[ni,ki,i] for ni in range(N+1) for ki in range(N+1)) for i in range(T))   
        q_new = num/den
         
        # Computation of sigma_new
        num = np.sum(np.sum((e[i]-q_new*ki)**2*p_n_k_e[ni,ki,i] for ni in range(N+1) for ki in range(N+1)) for i in range(T))        
        sigma_new = np.sqrt(num/T)
        
        # Computation of p_new
        num = np.sum(np.sum(ki*p_n_k_e[ni,ki,i] for ni in range(N+1) for ki in range(N+1)) for i in range(T))
        den = np.sum(np.sum(ni*p_n_k_e[ni,ki,i] for ni in range(N+1) for ki in range(N+1)) for i in range(T))
        p_new = num/den
        
        # Numerical computation of tauD_new
        def g(x):
            num = 0
            I = np.zeros(T)
            dI_tauD = np.zeros(T)
            for i in range(1,T):
                I[i] = 1-np.exp(-delta_t[i]/x)
                dI_tauD[i] = -np.exp(-delta_t[i]/x)*delta_t[i]/x**2
               
                for n in range(N+1):
                    for n_1 in range(N+1):
                        for k_1 in range(n_1+1):
                            num = num + dI_tauD[i]*((n - n_1 + k_1)/I[i] - (N-n)/(1-I[i]))*p_n_n_1_k_1_e[n,n_1,k_1,i]
            return num
        sol = optimize.root(g,tauD_old)
        tauD_new = sol.x        
        
        if abs(sigma_new-sigma_old)/sigma_old < tol and abs(p_new-p_old)/p_old < tol and abs(q_new-q_old)/q_old < tol and abs(tauD_new-tauD_old)/tauD_old < tol:
            print('Convergence reached in ' + str(it) + ' iterations')
            print('Sigma_hat = ' + str(sigma_new))
            print('p_hat = ' + str(p_new))
            print('q_hat = ' + str(q_new))
            print('tau_D_hat = ' + str(tauD_new))
            print('N_hat = ' + str(N))
            break
        
        if it == max_it:
            warnings.warn('Maximum number of iteration reached')
            
        if np.isnan(sigma_new) or np.isnan(p_new) or np.isnan(q_new) or np.isnan(tauD_new):
            warnings.warn('NaN value')
            break
        
    sigma = sigma_new
    p = p_new
    q = q_new
    tau_D = tauD_new
    
    return p,q,sigma,tau_D
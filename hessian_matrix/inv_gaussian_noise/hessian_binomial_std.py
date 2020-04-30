# -*- coding: utf-8 -*-
"""
Created on Fri Feb  7 10:31:05 2020

@author: gontier
"""

# Computes the Hessian of log p(D|M,theta)
# INPUT : vector of EPSCs, values for N, p, q, sigma, and tauD, and vector of ISI
# OUTPUT : Hessian matrix H
# Convention : 
# theta_1 = p
# theta_2 = q
# theta_3 = sigma
# theta_4 = tauD
# theta_5 = N

import numpy as np
import scipy.stats

def hessian(e,N,p,q,sigma,tau,delta_t):
    
    T = e.size
    # Uses the Baum-Welch algorithm (https://en.wikipedia.org/wiki/Baum%E2%80%93Welch_algorithm)
    # Probability of hidden variables conditionned on observation
    
    A = np.zeros([N+1,N+1,N+1,N+1,T])
    #A[n_i,k_i,n_i+1,k_i+1,i+1] : probability to transit from i to i+1
    dA_dp = np.zeros([N+1,N+1,N+1,N+1,T])
    dA_dtau = np.zeros([N+1,N+1,N+1,N+1,T])
    dA_dq = np.zeros([N+1,N+1,N+1,N+1,T])
    dA_dsigma = np.zeros([N+1,N+1,N+1,N+1,T])
    dA_dN = np.zeros([N+1,N+1,N+1,N+1,T])
    
    dA_dpdp = np.zeros([N+1,N+1,N+1,N+1,T])
    dA_dtaudtau = np.zeros([N+1,N+1,N+1,N+1,T])
    dA_dqdq = np.zeros([N+1,N+1,N+1,N+1,T])
    dA_dsigmadsigma = np.zeros([N+1,N+1,N+1,N+1,T])
    dA_dNdN = np.zeros([N+1,N+1,N+1,N+1,T])
    
    dA_dNdp = np.zeros([N+1,N+1,N+1,N+1,T])
    dA_dNdq = np.zeros([N+1,N+1,N+1,N+1,T])
    dA_dNdsigma = np.zeros([N+1,N+1,N+1,N+1,T])
    dA_dNdtau = np.zeros([N+1,N+1,N+1,N+1,T])
    
    dA_dpdtau = np.zeros([N+1,N+1,N+1,N+1,T])
    dA_dpdq = np.zeros([N+1,N+1,N+1,N+1,T])
    dA_dpdsigma = np.zeros([N+1,N+1,N+1,N+1,T])
    
    dA_dqdsigma = np.zeros([N+1,N+1,N+1,N+1,T])
    dA_dsigmadtau = np.zeros([N+1,N+1,N+1,N+1,T])
    
    dA_dqdtau = np.zeros([N+1,N+1,N+1,N+1,T])
    
    for n1 in range(N+1):
        for k1 in range(n1+1):
            for n2 in range(N+1):
                for k2 in range(n2+1):
                    for i in range(T-1):
                        A[n1,k1,n2,k2,i+1] = scipy.stats.binom.pmf(k2,n2,p)*scipy.stats.binom.pmf(n2-n1+k1,N-n1+k1,1-np.exp(-delta_t[i+1]/tau))
                        
                        dA_dp[n1,k1,n2,k2,i+1] = A[n1,k1,n2,k2,i+1]*(k2/p - (n2-k2)/(1-p))
                        
                        p2 = 1-np.exp(-delta_t[i+1]/tau)
                        dp2 = -np.exp(-delta_t[i+1]/tau)*delta_t[i+1]/tau**2
                        d2p2 = -(-np.exp(-delta_t[i+1]/tau)*2*delta_t[i+1]/tau**3 + np.exp(-delta_t[i+1]/tau)*(delta_t[i+1]/tau**2)**2)
                        
                        dA_dtau[n1,k1,n2,k2,i+1] = A[n1,k1,n2,k2,i+1]*dp2*((n2-n1+k1)/(p2)-(N-n2)/(1-p2))
                        
                        dA_dpdp[n1,k1,n2,k2,i+1] = dA_dp[n1,k1,n2,k2,i+1]*(k2/p - (n2-k2)/(1-p)) + \
                        A[n1,k1,n2,k2,i+1]*(-k2/p**2-(n2-k2)/(1-p)**2)
                        
                        dA_dtaudtau[n1,k1,n2,k2,i+1] = dA_dtau[n1,k1,n2,k2,i+1]*dp2*((n2-n1+k1)/(p2)-(N-n2)/(1-p2)) + \
                        A[n1,k1,n2,k2,i+1]*d2p2*((n2-n1+k1)/(p2)-(N-n2)/(1-p2)) + \
                        A[n1,k1,n2,k2,i+1]*dp2*dp2*(-(n2-n1+k1)/(p2**2) - (N-n2)/(1-p2)**2)
                        
                        dA_dpdtau[n1,k1,n2,k2,i+1] = dA_dtau[n1,k1,n2,k2,i+1]*(k2/p - (n2-k2)/(1-p))
                        
                        if n1 == N and k1 == 0:
                            dA_dN[n1,k1,n2,k2,i+1] = A[n1,k1,n2,k2,i+1]*(-1/(2*N) - (2*(N*p-k2)*p*N - (N*p-k2)**2)/(2*p*(1-p)*N**2))
                        else:
                            dA_dN[n1,k1,n2,k2,i+1] = A[n1,k1,n2,k2,i+1]*(-1/(2*(N-n1+k1)) + ((n2-n1+k1-p2*(N-n1+k1))*(n2-n1+k1+p2*(N-n1+k1)))/(2*p2*(1-p2)*(N-n1+k1)**2))
                            
                        
                        if n1 == N and k1 == 0:
                            dA_dNdN[n1,k1,n2,k2,i+1] = dA_dN[n1,k1,n2,k2,i+1]*(-1/(2*N) - (2*(N*p-k2)*p*N - (N*p-k2)**2)/(2*p*(1-p)*N**2)) + \
                            A[n1,k1,n2,k2,i+1]*((N + (2*(k2 - N*p)*(k2 - N*p + 2*N*p))/((-1 + p)*p))/(2*N**3))
                        else:
                            dA_dNdN[n1,k1,n2,k2,i+1] = dA_dN[n1,k1,n2,k2,i+1]*(-1/(2*(N-n1+k1)) + ((n2-n1+k1-p2*(N-n1+k1))*(n2-n1+k1+p2*(N-n1+k1)))/(2*p2*(1-p2)*(N-n1+k1)**2)) + \
                            A[n1,k1,n2,k2,i+1]*((2*k1**2 + 2*n1**2 + 2*n2**2 - N*p2 + N*p2**2 + \
                            k1*(-4*n1 + 4*n2 + (-1 + p2)*p2) + \
                            n1*(-4*n2 + p2 - p2**2))/(2*(k1 + N - n1)**3*(-1 + p2)*p2))

                        if n1 == N and k1 == 0:
                            dA_dNdtau[n1,k1,n2,k2,i+1] = dA_dtau[n1,k1,n2,k2,i+1]*(-1/(2*N) - (2*(N*p-k2)*p*N - (N*p-k2)**2)/(2*p*(1-p)*N**2))
                        else:                        
                            dA_dNdtau[n1,k1,n2,k2,i+1] = dA_dtau[n1,k1,n2,k2,i+1]*(-1/(2*(N-n1+k1)) + ((n2-n1+k1-p2*(N-n1+k1))*(n2-n1+k1+p2*(N-n1+k1)))/(2*p2*(1-p2)*(N-n1+k1)**2)) + \
                            A[n1,k1,n2,k2,i+1]*dp2*(-((n2**2 + k1**2*(-1 + p2)**2 + n1**2*(-1 + p2)**2 - 2*n2**2*p2 + N**2*p2**2 - \
                            2*k1*(-n2 + n1*(-1 + p2)**2 + 2*n2*p2 - N*p2**2) - \
                            2*n1*(n2 - 2*n2*p2 + N*p2**2))/(2*(k1 + N - n1)**2*(-1 + p2)**2* \
                            p2**2)))

                        if n1 == N and k1 == 0:
                            dA_dNdp[n1,k1,n2,k2,i+1] = dA_dp[n1,k1,n2,k2,i+1]*(-1/(2*N) - (2*(N*p-k2)*p*N - (N*p-k2)**2)/(2*p*(1-p)*N**2)) + \
                            A[n1,k1,n2,k2,i+1]*(((k2 - N*p)*(-1 + 2*p)*(k2 - N*p + 2*N*p))/(2*N**2*(-1 + p)**2*p**2))
                        else:                        
                            dA_dNdp[n1,k1,n2,k2,i+1] = dA_dp[n1,k1,n2,k2,i+1]*(-1/(2*(N-n1+k1)) + ((n2-n1+k1-p2*(N-n1+k1))*(n2-n1+k1+p2*(N-n1+k1)))/(2*p2*(1-p2)*(N-n1+k1)**2))
                        
                        
                        
    B = np.zeros([N+1,N+1,T])
    dB_dp = np.zeros([N+1,N+1,T])
    dB_dtau = np.zeros([N+1,N+1,T])
    dB_dq = np.zeros([N+1,N+1,T])
    dB_dsigma = np.zeros([N+1,N+1,T])
    dB_dN = np.zeros([N+1,N+1,T])
    
    dB_dpdp = np.zeros([N+1,N+1,T])
    dB_dtaudtau = np.zeros([N+1,N+1,T])
    dB_dqdq = np.zeros([N+1,N+1,T])
    dB_dsigmadsigma = np.zeros([N+1,N+1,T])
    dB_dNdN = np.zeros([N+1,N+1,T])
    
    dB_dNdp = np.zeros([N+1,N+1,T])
    dB_dNdq = np.zeros([N+1,N+1,T])
    dB_dNdsigma = np.zeros([N+1,N+1,T])
    dB_dNdtau = np.zeros([N+1,N+1,T])
    
    dB_dpdtau = np.zeros([N+1,N+1,T])
    dB_dpdq = np.zeros([N+1,N+1,T])
    dB_dpdsigma = np.zeros([N+1,N+1,T])
    
    dB_dqdsigma = np.zeros([N+1,N+1,T])
    dB_dsigmadtau = np.zeros([N+1,N+1,T])
    
    dB_dqdtau = np.zeros([N+1,N+1,T])

    
    
    for n in range(N+1):
        for k in range(n+1):
            for i in range(T):
                if k == 0:
                    B[n,k,i] = float(e[i] == 0)
                    dB_dsigma[n,k,i] = 0
                    dB_dq[n,k,i] = 0
                    dB_dsigmadsigma[n,k,i] = 0
                    dB_dqdq[n,k,i] = 0
                    dB_dqdsigma[n,k,i] = 0
                    
                elif e[i] != 0:
                    B[n,k,i] = (q**(3/2)*k/np.sqrt(2*np.pi*sigma**2*e[i]**3))*np.exp(-(q*(e[i]-q*k)**2)/(2*sigma**2*e[i]))

                    dB_dsigma[n,k,i] = B[n,k,i]*((e[i]**2*q + k**2*q**3 - e[i]*(2*k*q**2 + sigma**2))/(e[i]*sigma**3))
                 
                    dB_dq[n,k,i] = B[n,k,i]*(-((e[i]**2*q - 4*e[i]*k*q**2 + 3*k**2*q**3 - 3*e[i]*sigma**2)/(2*e[i]*q*sigma**2)))
                    
                    dB_dsigmadsigma[n,k,i] = B[n,k,i]*((1/(e[i]**2*sigma**6))*(e[i]**4*q**2 + k**4*q**6 - e[i]*k**2*q**3*(4*k*q**2 + 5*sigma**2) - \
                                   e[i]**3*(4*k*q**3 + 5*q*sigma**2) + 2*e[i]**2*(3*k**2*q**4 + 5*k*q**2*sigma**2 + sigma**4)))
                    
                    dB_dqdq[n,k,i] = B[n,k,i]*((1/(4*e[i]**2*q**2*sigma**4))*(e[i]**4*q**2 + 9*k**4*q**6 - 6*e[i]*k**2*q**3*(4*k*q**2 + 5*sigma**2) - \
                           2*e[i]**3*(4*k*q**3 + 3*q*sigma**2) + e[i]**2*(22*k**2*q**4 + 32*k*q**2*sigma**2 + 3*sigma**4)))
                                    
                    dB_dqdsigma[n,k,i] = B[n,k,i]*(-((1/(2*e[i]**2*q*sigma**5))*(e[i]**4*q**2 + 3*k**4*q**6 - 6*e[i]**3*q*(k*q**2 + sigma**2) - \
                               2*e[i]*k**2*q**3*(5*k*q**2 + 6*sigma**2) + 3*e[i]**2*(4*k**2*q**4 + 6*k*q**2*sigma**2 + sigma**4))))


    alpha = np.zeros([N+1,N+1,T]) #n,k,i
    dalpha_dsigma = np.zeros([N+1,N+1,T])
    dalpha_dq = np.zeros([N+1,N+1,T])
    dalpha_dp = np.zeros([N+1,N+1,T])
    dalpha_dtau = np.zeros([N+1,N+1,T])
    dalpha_dN = np.zeros([N+1,N+1,T])
    
    dalpha_dsigmadsigma = np.zeros([N+1,N+1,T])
    dalpha_dqdq = np.zeros([N+1,N+1,T])
    dalpha_dpdp = np.zeros([N+1,N+1,T])
    dalpha_dtaudtau = np.zeros([N+1,N+1,T])
    dalpha_dNdN = np.zeros([N+1,N+1,T])
    
    dalpha_dNdp = np.zeros([N+1,N+1,T])
    dalpha_dNdq = np.zeros([N+1,N+1,T])
    dalpha_dNdsigma = np.zeros([N+1,N+1,T])
    dalpha_dNdtau = np.zeros([N+1,N+1,T])
    
    dalpha_dpdsigma = np.zeros([N+1,N+1,T])
    dalpha_dqdsigma = np.zeros([N+1,N+1,T])
    dalpha_dsigmadtau = np.zeros([N+1,N+1,T])
    
    dalpha_dpdq = np.zeros([N+1,N+1,T])
    dalpha_dqdtau = np.zeros([N+1,N+1,T])
    
    dalpha_dpdtau = np.zeros([N+1,N+1,T])
    
    for k in range(N+1):            
        alpha[N,k,0] = scipy.stats.binom.pmf(k,N,p)*scipy.stats.norm.pdf(e[0],loc=q*k,scale=sigma)
        dalpha_dsigma[N,k,0] = alpha[N,k,0]*((e[0]-k*q)**2-sigma**2)/(sigma**3)
        dalpha_dq[N,k,0] = alpha[N,k,0]*k*(e[0]-k*q)/(sigma**2)
        dalpha_dp[N,k,0] = alpha[N,k,0]*(k/p - (N-k)/(1-p))
        dalpha_dN[N,k,0] = alpha[N,k,0]*(-1/(2*N) - (2*(N*p-k)*p*N-(N*p-k)**2)/(2*p*(1-p)*N**2))
        
        dalpha_dsigmadsigma[N,k,0] = dalpha_dsigma[N,k,0]*((e[0]-k*q)**2-sigma**2)/(sigma**3) + \
        alpha[N,k,0]*(1/sigma**2 - 3*(e[0]-q*k)**2/sigma**4)
        
        dalpha_dqdq[N,k,0] = dalpha_dq[N,k,0]*k*(e[0]-k*q)/(sigma**2) + \
        alpha[N,k,0]*-(k/sigma)**2
        
        dalpha_dpdp[N,k,0] = dalpha_dp[N,k,0]*(k/p - (N-k)/(1-p)) + \
        alpha[N,k,0]*(-k2/p**2-(n2-k2)/(1-p)**2)
        
        dalpha_dNdN[N,k,0] = dalpha_dN[N,k,0]*(-1/(2*N) - (2*(N*p-k)*p*N-(N*p-k)**2)/(2*p*(1-p)*N**2)) + \
        alpha[N,k,0]*((N + (2*(k - N*p)*(k - N*p + 2*p*N))/((-1 + p)*p))/(2*N**3))
        
        dalpha_dNdp[N,k,0] = dalpha_dp[N,k,0]*(-1/(2*N) - (2*(N*p-k)*p*N-(N*p-k)**2)/(2*p*(1-p)*N**2)) + \
        alpha[N,k,0]*(((k - N*p)*(-1 + 2*p)*(k - N*p + 2*N*p))/(2*N**2*(-1 + p)**2*p**2))
        
        dalpha_dNdq[N,k,0] = dalpha_dq[N,k,0]*(-1/(2*N) - (2*(N*p-k)*p*N-(N*p-k)**2)/(2*p*(1-p)*N**2))
        
        dalpha_dNdsigma[N,k,0] = dalpha_dsigma[N,k,0]*(-1/(2*N) - (2*(N*p-k)*p*N-(N*p-k)**2)/(2*p*(1-p)*N**2))
        
        dalpha_dpdq[N,k,0] = dalpha_dp[N,k,0]*k*(e[0]-k*q)/(sigma**2)
        
        dalpha_dpdsigma[N,k,0] = dalpha_dsigma[N,k,0]*(k/p - (N-k)/(1-p))
        
        dalpha_dqdsigma[N,k,0] = dalpha_dsigma[N,k,0]*(k*(e[0]-k*q)/(sigma**2)) + \
        alpha[N,k,0]*(-2*k*(e[i]-q*k)/sigma**3)
        
    for i in range(T-1):
        for n in range(N+1):
            for k in range(n+1):
                alpha[n,k,i+1] = B[n,k,i+1]*sum(alpha[nn,kk,i]*A[nn,kk,n,k,i+1] for kk in range(N+1) for nn in range (N+1))

                dalpha_dN[n,k,i+1] = dB_dN[n,k,i+1]*sum(alpha[nn,kk,i]*A[nn,kk,n,k,i+1] for kk in range(N+1) for nn in range (N+1)) + \
                B[n,k,i+1]*sum(dalpha_dN[nn,kk,i]*A[nn,kk,n,k,i+1] + alpha[nn,kk,i]*dA_dN[nn,kk,n,k,i+1] for kk in range(N+1) for nn in range (N+1))    

                dalpha_dsigma[n,k,i+1] = dB_dsigma[n,k,i+1]*sum(alpha[nn,kk,i]*A[nn,kk,n,k,i+1] for kk in range(N+1) for nn in range (N+1)) + \
                B[n,k,i+1]*sum(dalpha_dsigma[nn,kk,i]*A[nn,kk,n,k,i+1] + alpha[nn,kk,i]*dA_dsigma[nn,kk,n,k,i+1] for kk in range(N+1) for nn in range (N+1))
    
                dalpha_dq[n,k,i+1] = dB_dq[n,k,i+1]*sum(alpha[nn,kk,i]*A[nn,kk,n,k,i+1] for kk in range(N+1) for nn in range (N+1)) + \
                B[n,k,i+1]*sum(dalpha_dq[nn,kk,i]*A[nn,kk,n,k,i+1] + alpha[nn,kk,i]*dA_dq[nn,kk,n,k,i+1] for kk in range(N+1) for nn in range (N+1))
                
                dalpha_dp[n,k,i+1] = dB_dp[n,k,i+1]*sum(alpha[nn,kk,i]*A[nn,kk,n,k,i+1] for kk in range(N+1) for nn in range (N+1)) + \
                B[n,k,i+1]*sum(dalpha_dp[nn,kk,i]*A[nn,kk,n,k,i+1] + alpha[nn,kk,i]*dA_dp[nn,kk,n,k,i+1] for kk in range(N+1) for nn in range (N+1))
                            
                dalpha_dtau[n,k,i+1] = dB_dtau[n,k,i+1]*sum(alpha[nn,kk,i]*A[nn,kk,n,k,i+1] for kk in range(N+1) for nn in range (N+1)) + \
                B[n,k,i+1]*sum(dalpha_dtau[nn,kk,i]*A[nn,kk,n,k,i+1] + alpha[nn,kk,i]*dA_dtau[nn,kk,n,k,i+1] for kk in range(N+1) for nn in range (N+1))
                        
                dalpha_dsigmadsigma[n,k,i+1] = dB_dsigmadsigma[n,k,i+1]*sum(alpha[nn,kk,i]*A[nn,kk,n,k,i+1] for kk in range(N+1) for nn in range (N+1)) + \
                dB_dsigma[n,k,i+1]*sum(dalpha_dsigma[nn,kk,i]*A[nn,kk,n,k,i+1] + alpha[nn,kk,i]*dA_dsigma[nn,kk,n,k,i+1] for kk in range(N+1) for nn in range (N+1)) + \
                dB_dsigma[n,k,i+1]*sum(dalpha_dsigma[nn,kk,i]*A[nn,kk,n,k,i+1] + alpha[nn,kk,i]*dA_dsigma[nn,kk,n,k,i+1] for kk in range(N+1) for nn in range (N+1)) + \
                B[n,k,i+1]*sum(dalpha_dsigmadsigma[nn,kk,i]*A[nn,kk,n,k,i+1] + \
                dalpha_dsigma[nn,kk,i]*dA_dsigma[nn,kk,n,k,i+1] + \
                dalpha_dsigma[nn,kk,i]*dA_dsigma[nn,kk,n,k,i+1] + \
                alpha[nn,kk,i]*dA_dsigmadsigma[nn,kk,n,k,i+1] for kk in range(N+1) for nn in range (N+1))
                
                dalpha_dpdp[n,k,i+1] = dB_dpdp[n,k,i+1]*sum(alpha[nn,kk,i]*A[nn,kk,n,k,i+1] for kk in range(N+1) for nn in range (N+1)) + \
                dB_dp[n,k,i+1]*sum(dalpha_dp[nn,kk,i]*A[nn,kk,n,k,i+1] + alpha[nn,kk,i]*dA_dp[nn,kk,n,k,i+1] for kk in range(N+1) for nn in range (N+1)) + \
                dB_dp[n,k,i+1]*sum(dalpha_dp[nn,kk,i]*A[nn,kk,n,k,i+1] + alpha[nn,kk,i]*dA_dp[nn,kk,n,k,i+1] for kk in range(N+1) for nn in range (N+1)) + \
                B[n,k,i+1]*sum(dalpha_dpdp[nn,kk,i]*A[nn,kk,n,k,i+1] + \
                dalpha_dp[nn,kk,i]*dA_dp[nn,kk,n,k,i+1] + \
                dalpha_dp[nn,kk,i]*dA_dp[nn,kk,n,k,i+1] + \
                alpha[nn,kk,i]*dA_dpdp[nn,kk,n,k,i+1] for kk in range(N+1) for nn in range (N+1))
                
                dalpha_dqdq[n,k,i+1] = dB_dqdq[n,k,i+1]*sum(alpha[nn,kk,i]*A[nn,kk,n,k,i+1] for kk in range(N+1) for nn in range (N+1)) + \
                dB_dq[n,k,i+1]*sum(dalpha_dq[nn,kk,i]*A[nn,kk,n,k,i+1] + alpha[nn,kk,i]*dA_dq[nn,kk,n,k,i+1] for kk in range(N+1) for nn in range (N+1)) + \
                dB_dq[n,k,i+1]*sum(dalpha_dq[nn,kk,i]*A[nn,kk,n,k,i+1] + alpha[nn,kk,i]*dA_dq[nn,kk,n,k,i+1] for kk in range(N+1) for nn in range (N+1)) + \
                B[n,k,i+1]*sum(dalpha_dqdq[nn,kk,i]*A[nn,kk,n,k,i+1] + \
                dalpha_dq[nn,kk,i]*dA_dq[nn,kk,n,k,i+1] + \
                dalpha_dq[nn,kk,i]*dA_dq[nn,kk,n,k,i+1] + \
                alpha[nn,kk,i]*dA_dqdq[nn,kk,n,k,i+1] for kk in range(N+1) for nn in range (N+1))
                
                dalpha_dtaudtau[n,k,i+1] = dB_dtaudtau[n,k,i+1]*sum(alpha[nn,kk,i]*A[nn,kk,n,k,i+1] for kk in range(N+1) for nn in range (N+1)) + \
                dB_dtau[n,k,i+1]*sum(dalpha_dtau[nn,kk,i]*A[nn,kk,n,k,i+1] + alpha[nn,kk,i]*dA_dtau[nn,kk,n,k,i+1] for kk in range(N+1) for nn in range (N+1)) + \
                dB_dtau[n,k,i+1]*sum(dalpha_dtau[nn,kk,i]*A[nn,kk,n,k,i+1] + alpha[nn,kk,i]*dA_dtau[nn,kk,n,k,i+1] for kk in range(N+1) for nn in range (N+1)) + \
                B[n,k,i+1]*sum(dalpha_dtaudtau[nn,kk,i]*A[nn,kk,n,k,i+1] + \
                dalpha_dtau[nn,kk,i]*dA_dtau[nn,kk,n,k,i+1] + \
                dalpha_dtau[nn,kk,i]*dA_dtau[nn,kk,n,k,i+1] + \
                alpha[nn,kk,i]*dA_dtaudtau[nn,kk,n,k,i+1] for kk in range(N+1) for nn in range (N+1))    
                
                dalpha_dpdsigma[n,k,i+1] = dB_dpdsigma[n,k,i+1]*sum(alpha[nn,kk,i]*A[nn,kk,n,k,i+1] for kk in range(N+1) for nn in range (N+1)) + \
                dB_dp[n,k,i+1]*sum(dalpha_dsigma[nn,kk,i]*A[nn,kk,n,k,i+1] + alpha[nn,kk,i]*dA_dsigma[nn,kk,n,k,i+1] for kk in range(N+1) for nn in range (N+1)) + \
                dB_dsigma[n,k,i+1]*sum(dalpha_dp[nn,kk,i]*A[nn,kk,n,k,i+1] + alpha[nn,kk,i]*dA_dp[nn,kk,n,k,i+1] for kk in range(N+1) for nn in range (N+1)) + \
                B[n,k,i+1]*sum(dalpha_dpdsigma[nn,kk,i]*A[nn,kk,n,k,i+1] + \
                dalpha_dsigma[nn,kk,i]*dA_dp[nn,kk,n,k,i+1] + \
                dalpha_dp[nn,kk,i]*dA_dsigma[nn,kk,n,k,i+1] + \
                alpha[nn,kk,i]*dA_dpdsigma[nn,kk,n,k,i+1] for kk in range(N+1) for nn in range (N+1))
                
                dalpha_dpdq[n,k,i+1] = dB_dpdq[n,k,i+1]*sum(alpha[nn,kk,i]*A[nn,kk,n,k,i+1] for kk in range(N+1) for nn in range (N+1)) + \
                dB_dp[n,k,i+1]*sum(dalpha_dq[nn,kk,i]*A[nn,kk,n,k,i+1] + alpha[nn,kk,i]*dA_dq[nn,kk,n,k,i+1] for kk in range(N+1) for nn in range (N+1)) + \
                dB_dq[n,k,i+1]*sum(dalpha_dp[nn,kk,i]*A[nn,kk,n,k,i+1] + alpha[nn,kk,i]*dA_dp[nn,kk,n,k,i+1] for kk in range(N+1) for nn in range (N+1)) + \
                B[n,k,i+1]*sum(dalpha_dpdq[nn,kk,i]*A[nn,kk,n,k,i+1] + \
                dalpha_dq[nn,kk,i]*dA_dp[nn,kk,n,k,i+1] + \
                dalpha_dp[nn,kk,i]*dA_dq[nn,kk,n,k,i+1] + \
                alpha[nn,kk,i]*dA_dpdq[nn,kk,n,k,i+1] for kk in range(N+1) for nn in range (N+1))
                
                dalpha_dpdtau[n,k,i+1] = dB_dpdtau[n,k,i+1]*sum(alpha[nn,kk,i]*A[nn,kk,n,k,i+1] for kk in range(N+1) for nn in range (N+1)) + \
                dB_dp[n,k,i+1]*sum(dalpha_dtau[nn,kk,i]*A[nn,kk,n,k,i+1] + alpha[nn,kk,i]*dA_dtau[nn,kk,n,k,i+1] for kk in range(N+1) for nn in range (N+1)) + \
                dB_dtau[n,k,i+1]*sum(dalpha_dp[nn,kk,i]*A[nn,kk,n,k,i+1] + alpha[nn,kk,i]*dA_dp[nn,kk,n,k,i+1] for kk in range(N+1) for nn in range (N+1)) + \
                B[n,k,i+1]*sum(dalpha_dpdtau[nn,kk,i]*A[nn,kk,n,k,i+1] + \
                dalpha_dtau[nn,kk,i]*dA_dp[nn,kk,n,k,i+1] + \
                dalpha_dp[nn,kk,i]*dA_dtau[nn,kk,n,k,i+1] + \
                alpha[nn,kk,i]*dA_dpdtau[nn,kk,n,k,i+1] for kk in range(N+1) for nn in range (N+1))
                
                dalpha_dqdsigma[n,k,i+1] = dB_dqdsigma[n,k,i+1]*sum(alpha[nn,kk,i]*A[nn,kk,n,k,i+1] for kk in range(N+1) for nn in range (N+1)) + \
                dB_dq[n,k,i+1]*sum(dalpha_dsigma[nn,kk,i]*A[nn,kk,n,k,i+1] + alpha[nn,kk,i]*dA_dsigma[nn,kk,n,k,i+1] for kk in range(N+1) for nn in range (N+1)) + \
                dB_dsigma[n,k,i+1]*sum(dalpha_dq[nn,kk,i]*A[nn,kk,n,k,i+1] + alpha[nn,kk,i]*dA_dq[nn,kk,n,k,i+1] for kk in range(N+1) for nn in range (N+1)) + \
                B[n,k,i+1]*sum(dalpha_dqdsigma[nn,kk,i]*A[nn,kk,n,k,i+1] + \
                dalpha_dsigma[nn,kk,i]*dA_dq[nn,kk,n,k,i+1] + \
                dalpha_dq[nn,kk,i]*dA_dsigma[nn,kk,n,k,i+1] + \
                alpha[nn,kk,i]*dA_dqdsigma[nn,kk,n,k,i+1] for kk in range(N+1) for nn in range (N+1))
                
                dalpha_dqdtau[n,k,i+1] = dB_dqdtau[n,k,i+1]*sum(alpha[nn,kk,i]*A[nn,kk,n,k,i+1] for kk in range(N+1) for nn in range (N+1)) + \
                dB_dq[n,k,i+1]*sum(dalpha_dtau[nn,kk,i]*A[nn,kk,n,k,i+1] + alpha[nn,kk,i]*dA_dtau[nn,kk,n,k,i+1] for kk in range(N+1) for nn in range (N+1)) + \
                dB_dtau[n,k,i+1]*sum(dalpha_dq[nn,kk,i]*A[nn,kk,n,k,i+1] + alpha[nn,kk,i]*dA_dq[nn,kk,n,k,i+1] for kk in range(N+1) for nn in range (N+1)) + \
                B[n,k,i+1]*sum(dalpha_dqdtau[nn,kk,i]*A[nn,kk,n,k,i+1] + \
                dalpha_dtau[nn,kk,i]*dA_dq[nn,kk,n,k,i+1] + \
                dalpha_dq[nn,kk,i]*dA_dtau[nn,kk,n,k,i+1] + \
                alpha[nn,kk,i]*dA_dqdtau[nn,kk,n,k,i+1] for kk in range(N+1) for nn in range (N+1))
                
                dalpha_dsigmadtau[n,k,i+1] = dB_dsigmadtau[n,k,i+1]*sum(alpha[nn,kk,i]*A[nn,kk,n,k,i+1] for kk in range(N+1) for nn in range (N+1)) + \
                dB_dsigma[n,k,i+1]*sum(dalpha_dtau[nn,kk,i]*A[nn,kk,n,k,i+1] + alpha[nn,kk,i]*dA_dtau[nn,kk,n,k,i+1] for kk in range(N+1) for nn in range (N+1)) + \
                dB_dtau[n,k,i+1]*sum(dalpha_dsigma[nn,kk,i]*A[nn,kk,n,k,i+1] + alpha[nn,kk,i]*dA_dsigma[nn,kk,n,k,i+1] for kk in range(N+1) for nn in range (N+1)) + \
                B[n,k,i+1]*sum(dalpha_dsigmadtau[nn,kk,i]*A[nn,kk,n,k,i+1] + \
                dalpha_dtau[nn,kk,i]*dA_dsigma[nn,kk,n,k,i+1] + \
                dalpha_dsigma[nn,kk,i]*dA_dtau[nn,kk,n,k,i+1] + \
                alpha[nn,kk,i]*dA_dsigmadtau[nn,kk,n,k,i+1] for kk in range(N+1) for nn in range (N+1))
                
                dalpha_dNdN[n,k,i+1] = dB_dNdN[n,k,i+1]*sum(alpha[nn,kk,i]*A[nn,kk,n,k,i+1] for kk in range(N+1) for nn in range (N+1)) + \
                dB_dN[n,k,i+1]*sum(dalpha_dN[nn,kk,i]*A[nn,kk,n,k,i+1] + alpha[nn,kk,i]*dA_dN[nn,kk,n,k,i+1] for kk in range(N+1) for nn in range (N+1)) + \
                dB_dN[n,k,i+1]*sum(dalpha_dN[nn,kk,i]*A[nn,kk,n,k,i+1] + alpha[nn,kk,i]*dA_dN[nn,kk,n,k,i+1] for kk in range(N+1) for nn in range (N+1)) + \
                B[n,k,i+1]*sum(dalpha_dNdN[nn,kk,i]*A[nn,kk,n,k,i+1] + \
                dalpha_dN[nn,kk,i]*dA_dN[nn,kk,n,k,i+1] + \
                dalpha_dN[nn,kk,i]*dA_dN[nn,kk,n,k,i+1] + \
                alpha[nn,kk,i]*dA_dNdN[nn,kk,n,k,i+1] for kk in range(N+1) for nn in range (N+1))

                dalpha_dNdp[n,k,i+1] = dB_dNdp[n,k,i+1]*sum(alpha[nn,kk,i]*A[nn,kk,n,k,i+1] for kk in range(N+1) for nn in range (N+1)) + \
                dB_dN[n,k,i+1]*sum(dalpha_dp[nn,kk,i]*A[nn,kk,n,k,i+1] + alpha[nn,kk,i]*dA_dp[nn,kk,n,k,i+1] for kk in range(N+1) for nn in range (N+1)) + \
                dB_dp[n,k,i+1]*sum(dalpha_dN[nn,kk,i]*A[nn,kk,n,k,i+1] + alpha[nn,kk,i]*dA_dN[nn,kk,n,k,i+1] for kk in range(N+1) for nn in range (N+1)) + \
                B[n,k,i+1]*sum(dalpha_dNdp[nn,kk,i]*A[nn,kk,n,k,i+1] + \
                dalpha_dp[nn,kk,i]*dA_dN[nn,kk,n,k,i+1] + \
                dalpha_dN[nn,kk,i]*dA_dp[nn,kk,n,k,i+1] + \
                alpha[nn,kk,i]*dA_dNdp[nn,kk,n,k,i+1] for kk in range(N+1) for nn in range (N+1))
                
                dalpha_dNdq[n,k,i+1] = dB_dNdq[n,k,i+1]*sum(alpha[nn,kk,i]*A[nn,kk,n,k,i+1] for kk in range(N+1) for nn in range (N+1)) + \
                dB_dN[n,k,i+1]*sum(dalpha_dq[nn,kk,i]*A[nn,kk,n,k,i+1] + alpha[nn,kk,i]*dA_dq[nn,kk,n,k,i+1] for kk in range(N+1) for nn in range (N+1)) + \
                dB_dq[n,k,i+1]*sum(dalpha_dN[nn,kk,i]*A[nn,kk,n,k,i+1] + alpha[nn,kk,i]*dA_dN[nn,kk,n,k,i+1] for kk in range(N+1) for nn in range (N+1)) + \
                B[n,k,i+1]*sum(dalpha_dNdq[nn,kk,i]*A[nn,kk,n,k,i+1] + \
                dalpha_dq[nn,kk,i]*dA_dN[nn,kk,n,k,i+1] + \
                dalpha_dN[nn,kk,i]*dA_dq[nn,kk,n,k,i+1] + \
                alpha[nn,kk,i]*dA_dNdq[nn,kk,n,k,i+1] for kk in range(N+1) for nn in range (N+1))                

                dalpha_dNdsigma[n,k,i+1] = dB_dNdsigma[n,k,i+1]*sum(alpha[nn,kk,i]*A[nn,kk,n,k,i+1] for kk in range(N+1) for nn in range (N+1)) + \
                dB_dN[n,k,i+1]*sum(dalpha_dsigma[nn,kk,i]*A[nn,kk,n,k,i+1] + alpha[nn,kk,i]*dA_dsigma[nn,kk,n,k,i+1] for kk in range(N+1) for nn in range (N+1)) + \
                dB_dsigma[n,k,i+1]*sum(dalpha_dN[nn,kk,i]*A[nn,kk,n,k,i+1] + alpha[nn,kk,i]*dA_dN[nn,kk,n,k,i+1] for kk in range(N+1) for nn in range (N+1)) + \
                B[n,k,i+1]*sum(dalpha_dNdsigma[nn,kk,i]*A[nn,kk,n,k,i+1] + \
                dalpha_dsigma[nn,kk,i]*dA_dN[nn,kk,n,k,i+1] + \
                dalpha_dN[nn,kk,i]*dA_dsigma[nn,kk,n,k,i+1] + \
                alpha[nn,kk,i]*dA_dNdsigma[nn,kk,n,k,i+1] for kk in range(N+1) for nn in range (N+1))  
                
                dalpha_dNdtau[n,k,i+1] = dB_dNdtau[n,k,i+1]*sum(alpha[nn,kk,i]*A[nn,kk,n,k,i+1] for kk in range(N+1) for nn in range (N+1)) + \
                dB_dN[n,k,i+1]*sum(dalpha_dtau[nn,kk,i]*A[nn,kk,n,k,i+1] + alpha[nn,kk,i]*dA_dtau[nn,kk,n,k,i+1] for kk in range(N+1) for nn in range (N+1)) + \
                dB_dtau[n,k,i+1]*sum(dalpha_dN[nn,kk,i]*A[nn,kk,n,k,i+1] + alpha[nn,kk,i]*dA_dN[nn,kk,n,k,i+1] for kk in range(N+1) for nn in range (N+1)) + \
                B[n,k,i+1]*sum(dalpha_dNdtau[nn,kk,i]*A[nn,kk,n,k,i+1] + \
                dalpha_dtau[nn,kk,i]*dA_dN[nn,kk,n,k,i+1] + \
                dalpha_dN[nn,kk,i]*dA_dtau[nn,kk,n,k,i+1] + \
                alpha[nn,kk,i]*dA_dNdtau[nn,kk,n,k,i+1] for kk in range(N+1) for nn in range (N+1))

    p_e = np.sum(alpha,(0,1))[-1]
    dp_dp = np.sum(dalpha_dp,(0,1))[-1]
    dp_dq = np.sum(dalpha_dq,(0,1))[-1]
    dp_dsigma = np.sum(dalpha_dsigma,(0,1))[-1]
    dp_dtau  = np.sum(dalpha_dtau,(0,1))[-1]
    dp_dN  = np.sum(dalpha_dN,(0,1))[-1]
    
    H = np.zeros([5,5])
    # p,q,sigma,tauD,N
    
    H[0,0] = - dp_dp*dp_dp/p_e**2 + np.sum(dalpha_dpdp,(0,1))[-1]/p_e
    H[1,1] = - dp_dq*dp_dq/p_e**2 + np.sum(dalpha_dqdq,(0,1))[-1]/p_e
    H[2,2] = - dp_dsigma*dp_dsigma/p_e**2 + np.sum(dalpha_dsigmadsigma,(0,1))[-1]/p_e       
    H[3,3] = - dp_dtau*dp_dtau/p_e**2 + np.sum(dalpha_dtaudtau,(0,1))[-1]/p_e
    H[4,4] = - dp_dN*dp_dN/p_e**2 + np.sum(dalpha_dNdN,(0,1))[-1]/p_e
    
    H[0,1] = - dp_dp*dp_dq/p_e**2 + np.sum(dalpha_dpdq,(0,1))[-1]/p_e
    H[0,2] = - dp_dp*dp_dsigma/p_e**2 + np.sum(dalpha_dpdsigma,(0,1))[-1]/p_e
    H[0,3] = - dp_dp*dp_dtau/p_e**2 + np.sum(dalpha_dpdtau,(0,1))[-1]/p_e
    H[0,4] = - dp_dp*dp_dN/p_e**2 + np.sum(dalpha_dNdp,(0,1))[-1]/p_e
    
    H[1,2] = - dp_dq*dp_dsigma/p_e**2 + np.sum(dalpha_dqdsigma,(0,1))[-1]/p_e
    H[1,3] = - dp_dq*dp_dtau/p_e**2 + np.sum(dalpha_dqdtau,(0,1))[-1]/p_e 
    H[1,4] = - dp_dq*dp_dN/p_e**2 + np.sum(dalpha_dNdq,(0,1))[-1]/p_e
    
    H[2,3] = - dp_dsigma*dp_dtau/p_e**2 + np.sum(dalpha_dsigmadtau,(0,1))[-1]/p_e
    H[2,4] = - dp_dsigma*dp_dN/p_e**2 + np.sum(dalpha_dNdsigma,(0,1))[-1]/p_e
    
    H[3,4] = - dp_dN*dp_dtau/p_e**2 + np.sum(dalpha_dNdtau,(0,1))[-1]/p_e
    
    for i in range(1,5):
        for j in range(i):
            H[i,j] = H[j,i]
                    
            
    return H
                


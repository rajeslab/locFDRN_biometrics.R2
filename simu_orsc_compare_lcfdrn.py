##### Ordered Sun Cai Threshold Determination ##########
import numpy as np
import math 
import itertools as itertools
from itertools import product
import statsmodels.stats.multitest
from scipy.stats import norm
from scipy.sparse import lil_matrix
import time
import time
import concurrent.futures
from functools import partial
import multiprocessing
import time
import concurrent.futures
from functools import partial
def locFDRS_GWAS_thread(Z,scov,B,b,pi,tau):
    K = np.size(Z);
    locFDR = np.zeros(K);
    for i in range(K):
        Z_sub = Z[max(i-B,0):(min(i+B,K-1)+1)];
        cov_sub = scov[max(i-B,0):(min(i+B,K-1)+1),max(i-B,0):(min(i+B,K-1)+1)].toarray();
        l = min(i+B,K-1) - max(i-B,0) + 1;
        H = product([0,1], repeat=l);                                           
        sum1 = sum2 = 0;
        for s in H:
            if(s[min(B,i)] == 0): 
                t = ((np.exp(-(1/2)*np.dot(np.inner((Z_sub - np.multiply(s,b)), 
                                         np.linalg.inv(cov_sub+np.multiply(np.diag(s),pow(tau,2)))),
                                  (Z_sub - np.multiply(s,b))))* 
                pow(pi,sum(s)) * pow(1-pi,l-sum(s)))/np.sqrt(np.linalg.det(cov_sub+np.multiply(np.diag(s),pow(tau,2)))));
                sum1 = sum1 + t;
                sum2 = sum2 + t;
            else:
                sum2 = sum2 + ((np.exp(-(1/2)*np.inner(np.dot(Z_sub - np.multiply(s,b), 
                                         np.linalg.inv(cov_sub+np.multiply(np.diag(s),pow(tau,2)))),
                                  (Z_sub - np.multiply(s,b))))* 
               pow(pi,sum(s)) * pow(1-pi,l-sum(s)))/np.sqrt(np.linalg.det(cov_sub+np.multiply(np.diag(s),pow(tau,2)))));
        locFDR[i]= sum1/sum2;
  
  
    return(locFDR);

def bootsdmfdr(V,R,rep,boot):
    np.random.seed(64)
    stat = np.zeros(boot);
    for b in range(boot):
        boot_sample = np.random.choice(range(rep),size=rep,replace=True);
        V_boot = V[boot_sample];
        R_boot = R[boot_sample];
        stat[b]=np.mean(V_boot)/np.mean(R_boot);
    return(np.std(stat));


def rejected(p_val,level):
    oo = np.argsort(p_val);
    ss = np.sort(p_val);
    stat = np.divide(np.cumsum(ss),np.arange(1,np.size(ss)+1,1));
    collection = np.where(stat <= level);
    return(oo[collection]);

def parallel_EM_rev(zip2, zip1):
    print(zip2);
    np.random.seed(zip2)
    pi = zip1[0]
    cov = zip1[1]
    b = zip1[2]
    tau = zip1[3]
    N = zip1[4]
    scov = zip1[5]
    
    H = np.random.binomial(1, pi, size=K)  # Adjusted size based on K
    Z = np.multiply(H, b) + np.matmul(np.linalg.cholesky(cov + np.multiply(np.diag(H), pow(tau, 2))), np.random.standard_normal(K))
    del(cov);
    VR = []

    for n in range(N + 1):
        # Start the timer for the current value of n
        start_time = time.perf_counter()
        
        Hr = np.zeros(K)
        Hr[rejected(locFDRS_GWAS_thread(Z, scov, n, b, pi, tau), alpha)] = 1
        Vr = np.size(np.intersect1d(np.where(H == 0), np.where(Hr == 1)))
        Rr = np.sum(Hr)
        
        # End the timer and record the runtime for the current value of n
        end_time = time.perf_counter()
        VR.append([Vr, Rr, end_time - start_time])

    # Return VR and runtimes together
    return VR

## Rho = np.array([0.3,0.5,0.7]);
Rho = np.array([0.3,0.5,0.8]); # For AR(1) and Equicorrelated
RC = [];
for rho in Rho:
    bandwidth = 7;
    # K=700;
    blockcov = np.repeat(rho,bandwidth*bandwidth).reshape(bandwidth,bandwidth)+(1-rho)*np.identity(bandwidth);
    # cov = np.kron(np.eye(int(K/bandwidth),dtype=int),blockcov);
    # Equicorrelated Covariance
    #K = 1000
    #cov = np.repeat(rho,K*K).reshape(K,K);
    #np.fill_diagonal(cov, 1);
    #AR(1) covariance
    K = 1000
    cov = np.zeros((K, K))
    for i in range(K):
        for j in range(K):
            cov[i, j] = (rho**abs(i - j)) 

    pi = 0.3;
    tau = np.sqrt(4);
    b = -0.2;
    N=5; alpha = 0.05;   
    scov = lil_matrix((K, K))
    for n in range((2*N)+1):
        scov.setdiag(cov.diagonal(n),n);
        scov.setdiag(cov.diagonal(n),-n);
    start = time.perf_counter()
    rep = 1000;     ##### Number of replicates for determining estimates(1000 for K=1000)
    zip1 = [pi,cov,b,tau,N,scov,bandwidth];
    if __name__ == "__main__":
            num_processes = 25 # multiprocessing.cpu_count()
            with multiprocessing.Pool(processes=num_processes) as pool:
                results = pool.map(partial(parallel_EM_rev, zip1=zip1), np.arange(rep))
            results_array = np.array(results)
            del(results)


    finish = time.perf_counter()
    print(f'Finished in {round(finish-start, 2)} second(s)')

    Rn = [];
    for n in range(N+1):
        Vr = [];Rr = [];Tr = [];
        for r in range(rep):
            Vr.append(results_array[r][n,0]);
            Rr.append(results_array[r][n,1]);
            Tr.append(results_array[r][n,2]);
        Vr = np.array(Vr);Rr = np.array(Rr);Tr = np.array(Tr);
        Rn.append([np.mean(Vr)/np.mean(Rr),bootsdmfdr(Vr,Rr,rep,rep),
                   np.mean(np.divide(Vr,np.maximum(Rr,np.ones(rep)))),
                   np.std(np.divide(Vr,np.maximum(Rr,np.ones(rep))))/np.sqrt(rep),
                  np.mean(Rr - Vr),np.std(Rr - Vr)/np.sqrt(rep),np.mean(Tr),np.std(Tr)/np.sqrt(rep)])
    RC.append(Rn);


print(RC)
np.save('RC_orsc_AR1_03_1K.npy',np.array(RC));

import numpy as np
import math 
import torch
import itertools as itertools
from itertools import product
import statsmodels.stats.multitest
from scipy.stats import norm
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
# Using R inside python
import rpy2
import rpy2.robjects as robjects
#from rpy2.robjects import pandas2ri
from rpy2.robjects.vectors import FloatVector
# Defining the R script and loading the instance in Python
r = robjects.r
r['source']('sun_cai(2007)_est.R')

# Loading the functions for sun and cai method defined in R.
est_sun_cai_r = robjects.globalenv['epsest.func']   #### function for estimating $\pi$

rej_sun_cai_r = robjects.globalenv['adaptZ.funcnull']

############################################################

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

def rejected_cutoff(lcfdr,cutoff):
    return(np.where(lcfdr<cutoff));


def parallel_EM_SC(zip2,zip1):
        print(zip2)
        np.random.seed(zip2);
        pi = zip1[0]; cov = zip1[1]; b=zip1[2]; tau = zip1[3]; N = zip1[4]; scov = zip1[5];
        H = np.random.binomial(1,pi,size=K);
        Z = np.multiply(H,b) + np.matmul(np.linalg.cholesky(cov+np.multiply(np.diag(H),pow(tau,2))),np.random.standard_normal(K));
        VR = [];
        est_vec = EM_est(Z,scov)
        # print(est_vec);
        for n in range(N+1):
            Hr = np.zeros(K);
            Hr[rejected(locFDRS_GWAS_thread(Z,scov,n,est_vec[0],est_vec[1],np.sqrt(est_vec[2])),alpha)]=1;
            Vr = np.size(np.intersect1d(np.where(H==0),np.where(Hr==1))); Rr = np.sum(Hr);
            VR.append([Vr,Rr])
        HSC = np.zeros(K);
        HSC[np.array(rej_sun_cai_r(FloatVector(Z),alpha))-1]=1;
        VSC = np.size(np.intersect1d(np.where(H==0),np.where(HSC==1))); RSC = np.sum(HSC);
        VR.append([VSC,RSC])
        HB = np.zeros(K);
        HB[np.where(statsmodels.stats.multitest.multipletests(2*(1-norm.cdf(abs(Z))), alpha=alpha, method='fdr_bh', is_sorted=False, returnsorted=False)[0]==True)[0]]=1;
        VB = np.size(np.intersect1d(np.where(H==0),np.where(HB==1))); RB = np.sum(HB);
        VR.append([VB,RB])
        HAB = np.zeros(K);
        HAB[np.where(statsmodels.stats.multitest.multipletests(2*(1-norm.cdf(abs(Z))), alpha=(alpha/(1-est_vec[1])), method='fdr_bh', is_sorted=False, returnsorted=False)[0]==True)[0]]=1;
        VAB = np.size(np.intersect1d(np.where(H==0),np.where(HAB==1))); RAB = np.sum(HAB);
        VR.append([VAB,RAB])
        return(VR)

def EM_est(Z,scov):
      maxiter = 5000;
      ##est_mat = np.zeros(3*maxiter).reshape(maxiter ,3)
      est_mat = np.zeros(2*maxiter).reshape(maxiter ,2)
      #z = Z[((4*(1+np.arange(1000)))-1)];
      #z = Z[((3*(1+np.arange(1150)))-1)];
      z = Z#[((2*(1+np.arange(1000)))-1)]
      ##pi_ini = max(est_sun_cai_r(FloatVector(z),0.,1.)+0.);
      pi_est = max(est_sun_cai_r(FloatVector(Z),0.,1.)+0.001);
      ##b_ini = np.mean(z)/pi_ini; tausq_ini = np.amax([0.,(np.var(z)-1-(pow(b_ini,2)*(1-pow(pi_ini,2))))/pi_ini]);
      b_ini = np.mean(z)/pi_est; tausq_ini = np.amax([0.,(np.var(z)-1-(pow(b_ini,2)*(1-pow(pi_est,2))))/pi_est]); 
      ##est_mat[0,0] = pi_ini; est_mat[0,0] = b_ini; est_mat[0,1] = tausq_ini;
      est_mat[0,0] = b_ini; est_mat[0,1] = tausq_ini;
      for iter in 1+np.arange(maxiter-1):
        p_tau = 1-locFDRS_GWAS_thread(z,scov,0,b_ini,pi_est,np.sqrt(tausq_ini));
        ##p_tau = 1-locFDRS_GWAS_thread(z,cov,0,b_ini,pi_ini,np.sqrt(tausq_ini));
        ##pi_next = np.mean(p_tau);
        b_next = (np.sum((p_tau)*z)/max(np.sum(p_tau),0.001));
        tausq_next = np.amax([0,(np.sum((p_tau)*pow(z-b_next,2)) /max(np.sum(p_tau),0.001))-1]);
        b_ini = b_next; tausq_ini = tausq_next;
        ##pi_ini = pi_next; b_ini = b_next; tausq_ini = tausq_next;
        ##est_mat[iter,0] = pi_ini; est_mat[iter,1] = b_ini; est_mat[iter,2] = tausq_ini;
        est_mat[iter,0] = b_ini; est_mat[iter,1] = tausq_ini;
        if np.amax(abs(est_mat[iter,:]-est_mat[iter-1,:])) < pow(10,-15):
          break;
      return(np.array([b_ini,pi_est,tausq_ini]));



#################################################################################################################

import multiprocessing
########## Set the parameters of choice ##########
Cov = [];
### AR(1) covariance
rho = 0.8;
#B = 1;
K = 15000;
diagonal = np.concatenate((1/(1-pow(rho,2)),np.repeat(((1+pow(rho,2))/(1-pow(rho,2))),K-2),1/(1-pow(rho,2))),axis=None);
off_diag = -rho/(1-pow(rho,2));
prec = np.diag(diagonal);
for d in range(K-1):
    prec[d,d+1]=prec[d+1,d] = off_diag;
#cov = np.linalg.inv(prec);
cov = np.zeros((K, K))
for i in range(K):
    for j in range(K):
        cov[i, j] = (rho**abs(i - j)) 

Cov.append(cov);

### Banded Covariance with bandwidth 1
rho = 0.5
#K = 1000
cov = np.diag(np.ones(K));
for i in range(K-1):
    cov[i,i+1] = cov[i+1,i] = rho;

Cov.append(cov);
### Equicorrelated Covariance
#rho = 0.8
#K = 1000
#cov = np.repeat(rho,K*K).reshape(K,K);
#for i in range(K):
#    cov[i,i] = 1;

#Cov.append(cov);
### FGN Covariance
#import numpy as np
#H = 0.9;
#K = 1000;
#cov = np.repeat(0.1,(K*K)).reshape(K,K)
#for i in range(K):
#    for j in range(K):
#        cov[i,j] = (0.5)*(pow(abs(abs(i-j)+1),(2*H)) - (2*pow(abs(i-j),(2*H))) + pow(abs(abs(i-j)-1),(2*H)));

#Cov.append(cov);
del(cov)
########## Set the parameters of choice ##########




RC = [];
# warnings.filterwarnings("ignore")
for cov in Cov:
    pi = 0.3;
    tau = np.sqrt(2);
    b = 0.;
    N=2;  alpha = 0.05;  
    scov = lil_matrix((K, K))
    for n in range((2*N)+1):
        scov.setdiag(cov.diagonal(n),n);
        scov.setdiag(cov.diagonal(n),-n);
    # zipp = [pi,cov,b,tau,N,scov];
    start = time.perf_counter()



    import time
    import concurrent.futures
    from functools import partial
    rep = 500;     ##### Number of replicates for determining estimates(500 for K=15000 for data driven)
    zip1 = [pi,cov,b,tau,N,scov];
    if __name__ == "__main__":
            num_processes = multiprocessing.cpu_count()
            with multiprocessing.Pool(processes=num_processes) as pool:
                results = pool.map(partial(parallel_EM_SC, zip1=zip1), np.arange(rep))
            results_array = np.array(results)
            del(results)


    finish = time.perf_counter()
    print(f'Finished in {round(finish-start, 2)} second(s)')

    Rn = [];
    for n in range(N+1):
        Vr = [];Rr = [];
        for r in range(rep):
            Vr.append(results_array[r][n,0]);
            Rr.append(results_array[r][n,1]);
        Vr = np.array(Vr);Rr = np.array(Rr);
        Rn.append([np.mean(Vr)/np.mean(Rr),bootsdmfdr(Vr,Rr,rep,rep),
                   np.mean(np.divide(Vr,np.maximum(Rr,np.ones(rep)))),
                   np.std(np.divide(Vr,np.maximum(Rr,np.ones(rep))))/np.sqrt(rep),
                  np.mean(Rr - Vr),np.std(Rr - Vr)/np.sqrt(rep)])
    VSC = [];RSC = [];
    for r in range(rep):
        VSC.append(results_array[r][N+1,0]);
        RSC.append(results_array[r][N+1,1]);
    VSC = np.array(VSC);RSC = np.array(RSC);
    Rn.append([np.mean(VSC)/np.mean(RSC),bootsdmfdr(VSC,RSC,rep,rep),
               np.mean(np.divide(VSC,np.maximum(RSC,np.ones(rep)))),
               np.std(np.divide(VSC,np.maximum(RSC,np.ones(rep))))/np.sqrt(rep),
              np.mean(RSC - VSC),np.std(RSC - VSC)/np.sqrt(rep)])
    VB = [];RB = [];
    for r in range(rep):
        VB.append(results_array[r][N+2,0]);
        RB.append(results_array[r][N+2,1]);
    VB = np.array(VB);RB = np.array(RB);
    Rn.append([np.mean(VB)/np.mean(RB),bootsdmfdr(VB,RB,rep,rep),
               np.mean(np.divide(VB,np.maximum(RB,np.ones(rep)))),
               np.std(np.divide(VB,np.maximum(RB,np.ones(rep))))/np.sqrt(rep),
              np.mean(RB - VB),np.std(RB - VB)/np.sqrt(rep)])
    VAB = [];RAB = [];
    for r in range(rep):
        VAB.append(results_array[r][N+3,0]);
        RAB.append(results_array[r][N+3,1]);
    VAB = np.array(VAB);RAB = np.array(RAB);
    Rn.append([np.mean(VAB)/np.mean(RAB),bootsdmfdr(VAB,RAB,rep,rep),
               np.mean(np.divide(VAB,np.maximum(RAB,np.ones(rep)))),
               np.std(np.divide(VAB,np.maximum(RAB,np.ones(rep))))/np.sqrt(rep),
              np.mean(RAB - VAB),np.std(RAB - VAB)/np.sqrt(rep)])
    RC.append(Rn);
print(RC)
np.save('RC_dd_03_15K.npy',np.array(RC));

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

def sim_lcfdr(ndraw,zipp):
    #print(ndraw)
    N = zipp[4]; scov = zipp[5]
    pi = zipp[0]; cov = zipp[1]; b=zipp[2]; tau = zipp[3];
    lcfdr_mat = [];
    np.random.seed(ndraw);
#     K = np.shape(cov)[1];
    H = np.random.binomial(1,pi,size=K);
    Z = np.multiply(H,b) + np.matmul(np.linalg.cholesky(cov+np.multiply(np.diag(H),pow(tau,2))),np.random.standard_normal(K));
    for n in np.arange(N+1):
        lcfdr_mat.append(locFDRS_GWAS_thread(Z,scov,n,b,pi,tau));
    return(lcfdr_mat);
    
def parallel_EM(zip2,zip1):
        np.random.seed(zip2);
        pi = zip1[0]; cov = zip1[1]; b=zip1[2]; tau = zip1[3]; N = zip1[4]; scov = zip1[5];
        H = np.random.binomial(1,pi,size=K);
        Z = np.multiply(H,b) + np.matmul(np.linalg.cholesky(cov+np.multiply(np.diag(H),pow(tau,2))),np.random.standard_normal(K));
        VR = [];
        for n in range(N+1):
            Hr = np.zeros(K);
            Hr[rejected_cutoff(locFDRS_GWAS_thread(Z,scov,n,b,pi,tau),t[n])]=1;
            Vr = np.size(np.intersect1d(np.where(H==0),np.where(Hr==1))); Rr = np.sum(Hr);
            VR.append([Vr,Rr])
        return(VR)

import multiprocessing 
########## Set the parameters of choice ##########
Cov = [];
### AR(1) covariance
rho = 0.5;
#B = 1;
K = 10000;
diagonal = np.concatenate((1/(1-pow(rho,2)),np.repeat(((1+pow(rho,2))/(1-pow(rho,2))),K-2),1/(1-pow(rho,2))),axis=None);
off_diag = -rho/(1-pow(rho,2));
prec = np.diag(diagonal);
for d in range(K-1):
    prec[d,d+1]=prec[d+1,d] = off_diag;
cov = np.linalg.inv(prec);

Cov.append(cov);

### Banded Covariance with bandwidth 1
rho = 0.5
#K = 1000
cov = np.diag(np.ones(K));
for i in range(K-1):
    cov[i,i+1] = cov[i+1,i] = rho;

Cov.append(cov);
### Equicorrelated Covariance
rho = 0.5
#K = 1000
cov = np.repeat(rho,K*K).reshape(K,K);
for i in range(K):
    cov[i,i] = 1;

Cov.append(cov);
### FGN Covariance
import numpy as np
H = 0.7;
#K = 1000;
cov = np.repeat(0.1,(K*K)).reshape(K,K)
for i in range(K):
    for j in range(K):
        cov[i,j] = (0.5)*(pow(abs(abs(i-j)+1),(2*H)) - (2*pow(abs(i-j),(2*H))) + pow(abs(abs(i-j)-1),(2*H)));

Cov.append(cov);
del(cov)
########## Set the parameters of choice ##########
RC = [];
for cov in Cov:
    pi = 0.3;
    tau = np.sqrt(4);
    b = 0.;
    N=2; ndraws = 500; alpha = 0.05;         ##### ndraws = 500 for K= 10000, ndraws = 3000 for K=1000
    scov = lil_matrix((K, K))
    for n in range((2*N)+1):
        scov.setdiag(cov.diagonal(n),n);
        scov.setdiag(cov.diagonal(n),-n);
    zipp = [pi,cov,b,tau,N,scov];
    start = time.perf_counter()
#     list_iter = [];
#     for ndraw in np.arange(ndraws):
#         list_iter.append([zip,N,ndraw])
    if __name__ == "__main__":
            num_processes = multiprocessing.cpu_count()
            with multiprocessing.Pool(processes=num_processes) as pool:
                results = pool.map(partial(sim_lcfdr, zipp=zipp), np.arange(ndraws));
            results_array = np.array(results)
            del(results)
    print("done");
    t = [];
    for n in range(N+1):
        lcfdr = np.array([]);
        for ndraw in range(ndraws):
            lcfdr = np.concatenate([lcfdr,results_array[ndraw][n]]);
        oo = np.argsort(lcfdr);
        ss = lcfdr[oo];
        stat = np.divide(np.cumsum(ss),np.arange(1,int(K*ndraws)+1,1));
        collection = np.max(np.append(np.where(stat<=alpha),0));
        t.append(ss[collection]);
    print(t)


    import time
    import concurrent.futures
    from functools import partial
    rep = 500;     ##### Number of replicates for determining estimates(3000 for K=1000, 500 for K = 10000)
    zip1 = [pi,cov,b,tau,N,scov];
    if __name__ == "__main__":
            num_processes = multiprocessing.cpu_count()
            with multiprocessing.Pool(processes=num_processes) as pool:
                results = pool.map(partial(parallel_EM, zip1=zip1), np.arange(rep))
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
    RC.append(Rn);
print(RC)
np.save('RC_03_10K.npy',np.array(RC));

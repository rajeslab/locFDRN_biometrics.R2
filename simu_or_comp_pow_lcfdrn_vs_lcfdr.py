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

def rejected_cutoff(lcfdr,cutoff):
    return(np.where(lcfdr<cutoff));

def sim_lcfdr(ndraw,zipp):
    print(ndraw)
    N = zipp[4]; scov = zipp[5]
    pi = zipp[0]; cov = zipp[1]; b=zipp[2]; tau = zipp[3]; rho = cov[0,1]; bandwidth=zipp[6];
    lcfdr_mat = [];
    np.random.seed(ndraw);
    #K = np.shape(cov)[1];
    H = np.random.binomial(1,pi,size=K);
    Z = np.multiply(H,b) + np.matmul(np.linalg.cholesky(cov+np.multiply(np.diag(H),pow(tau,2))),np.random.standard_normal(K));
    del(cov);
    for n in np.arange(N+1):
        lcfdr_mat.append(locFDRS_GWAS_thread(Z,scov,n,b,pi,tau));
    lcfdr_mat.append(locFDR_band(Z,rho,bandwidth,b,pi,tau))
    return(lcfdr_mat);

def locFDR_band(Z,rho,bandwidth,b,pi,tau):
    K = np.size(Z);
    blockcov = np.repeat(rho,bandwidth*bandwidth).reshape(bandwidth,bandwidth)+(1-rho)*np.identity(bandwidth);
    locFDR = np.zeros(K);
    for i in range(K):
        N = i%bandwidth;
        blockZ = Z[(int(i/bandwidth)*bandwidth):((int(i/bandwidth)+1)*bandwidth)];
        H = product([0,1], repeat=bandwidth);                                           
        sum1 = sum2 = 0;
        for s in H:
            if(s[N] == 0): 
                t = ((np.exp(-(1/2)*np.dot(np.inner((blockZ - np.multiply(s,b)), 
                                         np.linalg.inv(blockcov+np.multiply(np.diag(s),pow(tau,2)))),
                                  (blockZ - np.multiply(s,b))))* 
                pow(pi,sum(s)) * pow(1-pi,bandwidth-sum(s)))/np.sqrt(np.linalg.det(blockcov+np.multiply(np.diag(s),pow(tau,2)))));
                sum1 = sum1 + t;
                sum2 = sum2 + t;
            else:
                sum2 = sum2 + ((np.exp(-(1/2)*np.inner(np.dot(blockZ - np.multiply(s,b), 
                                         np.linalg.inv(blockcov+np.multiply(np.diag(s),pow(tau,2)))),
                                  (blockZ - np.multiply(s,b))))* 
               pow(pi,sum(s)) * pow(1-pi,bandwidth-sum(s)))/np.sqrt(np.linalg.det(blockcov+np.multiply(np.diag(s),pow(tau,2)))));
        locFDR[i]= sum1/sum2;
    return(locFDR)
    
def parallel_EM(zip2,zip1):
        print(zip2);
        np.random.seed(zip2);
        pi = zip1[0]; cov = zip1[1]; b=zip1[2]; tau = zip1[3]; N = zip1[4]; scov = zip1[5];
        H = np.random.binomial(1,pi,size=K);
        Z = np.multiply(H,b) + np.matmul(np.linalg.cholesky(cov+np.multiply(np.diag(H),pow(tau,2))),np.random.standard_normal(K));
        del(cov);
        VR = [];
        for n in range(N+1):
            Hr = np.zeros(K);
            start_time = time.perf_counter()
            Hr[rejected_cutoff(locFDRS_GWAS_thread(Z,scov,n,b,pi,tau),t[n])]=1;
            end_time = time.perf_counter()
            Vr = np.size(np.intersect1d(np.where(H==0),np.where(Hr==1))); Rr = np.sum(Hr);
            VR.append([Vr,Rr,end_time - start_time])
        Hr = np.zeros(K);
        start_time = time.perf_counter()
        Hr[rejected_cutoff(locFDR_band(Z,rho,bandwidth,b,pi,tau),t[N+1])]=1;
        end_time = time.perf_counter()
        Vr = np.size(np.intersect1d(np.where(H==0),np.where(Hr==1))); Rr = np.sum(Hr);
        VR.append([Vr,Rr,end_time - start_time])
        return(VR)
    
Rho = np.array([0.3,0.5,0.7]);
RC = [];
for rho in Rho:
    bandwidth = 7;
    K=1400;
    blockcov = np.repeat(rho,bandwidth*bandwidth).reshape(bandwidth,bandwidth)+(1-rho)*np.identity(bandwidth);
    cov = np.kron(np.eye(int(K/bandwidth),dtype=int),blockcov);

    pi = 0.1;
    tau = np.sqrt(2.5);
    b = 0.;
    N=3; ndraws = 3000; alpha = 0.05;   ##ndraws = 1000, N=5 for K=700,pi=0.3, ndraws = 3000, N=3 for K=1400,pi=0.1
    scov = lil_matrix((K, K))
    for n in range((2*N)+1):
        scov.setdiag(cov.diagonal(n),n);
        scov.setdiag(cov.diagonal(n),-n);
    zipp = [pi,cov,b,tau,N,scov,bandwidth];
    start = time.perf_counter()
    #     list_iter = [];
    #     for ndraw in np.arange(ndraws):
    #         list_iter.append([zip,N,ndraw])
    if __name__ == "__main__":
            num_processes = 25 #multiprocessing.cpu_count()
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
    lcfdr = np.array([]);
    for ndraw in range(ndraws):
        lcfdr = np.concatenate([lcfdr,results_array[ndraw][N+1]]);
    oo = np.argsort(lcfdr);
    ss = lcfdr[oo];
    stat = np.divide(np.cumsum(ss),np.arange(1,int(K*ndraws)+1,1));
    collection = np.max(np.append(np.where(stat<=alpha),0));
    t.append(ss[collection]);
    print(t)

    import time
    import concurrent.futures
    from functools import partial
    rep = 3000;     ##### Number of replicates for determining estimates(1000 for K=700,pi=0.3,3000 for K=1400, pi=0.1)
    zip1 = [pi,cov,b,tau,N,scov,bandwidth];
    if __name__ == "__main__":
            num_processes = 25 #multiprocessing.cpu_count()
            with multiprocessing.Pool(processes=num_processes) as pool:
                results = pool.map(partial(parallel_EM, zip1=zip1), np.arange(rep))
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
    Vr = [];Rr = [];Tr = [];
    for r in range(rep):
        Vr.append(results_array[r][N+1,0]);
        Rr.append(results_array[r][N+1,1]);
        Tr.append(results_array[r][N+1,2]);
    Vr = np.array(Vr);Rr = np.array(Rr);Tr = np.array(Tr);
    Rn.append([np.mean(Vr)/np.mean(Rr),bootsdmfdr(Vr,Rr,rep,rep),
               np.mean(np.divide(Vr,np.maximum(Rr,np.ones(rep)))),
               np.std(np.divide(Vr,np.maximum(Rr,np.ones(rep))))/np.sqrt(rep),
              np.mean(Rr - Vr),np.std(Rr - Vr)/np.sqrt(rep),np.mean(Tr),np.std(Tr)/np.sqrt(rep)])
    RC.append(Rn);


print(RC)
np.save('RC_com_pow_lcfdrNvslcfdr_pi_01.npy',np.array(RC));

import numpy as np
import math 
import itertools as itertools
from itertools import product
def locFDRS_GWAS_thread(Z,cov,B,b,pi,tau):
  K = np.size(Z);
  locFDR = np.zeros(K);
  for i in range(K):
    Z_sub = Z[max(i-B,0):(min(i+B,K-1)+1)];
    cov_sub = cov[max(i-B,0):(min(i+B,K-1)+1),max(i-B,0):(min(i+B,K-1)+1)];
    l = min(i+B,K-1) - max(i-B,0) + 1;
    H = product([0,1], repeat=l);                                           
    sum1 = sum2 = 0;
    for s in H:
      if(s[min(B,i)] == 0): 
        #t = (multivariate_normal.pdf(Z_sub,mean=np.multiply(s,b),cov = cov_sub+np.multiply(np.diag(s),pow(tau,2)))*pow(pi,sum(s)) * pow(1-pi,l-sum(s)))
        t = ((np.exp(-(1/2)*np.dot(np.inner((Z_sub - np.multiply(s,b)), 
                                         np.linalg.inv(cov_sub+np.multiply(np.diag(s),pow(tau,2)))),
                                  (Z_sub - np.multiply(s,b))))* 
          pow(pi,sum(s)) * pow(1-pi,l-sum(s)))/np.sqrt(np.linalg.det(cov_sub+np.multiply(np.diag(s),pow(tau,2)))));
        sum1 = sum1 + t;
        sum2 = sum2 + t;
      else:
        #sum2 = sum2 + (multivariate_normal.pdf(Z_sub,mean=np.multiply(s,b),cov = cov_sub+np.multiply(np.diag(s),pow(tau,2)))*pow(pi,sum(s)) * pow(1-pi,l-sum(s)))
        sum2 = sum2 + ((np.exp(-(1/2)*np.inner(np.dot(Z_sub - np.multiply(s,b), 
                                         np.linalg.inv(cov_sub+np.multiply(np.diag(s),pow(tau,2)))),
                                  (Z_sub - np.multiply(s,b))))* 
          pow(pi,sum(s)) * pow(1-pi,l-sum(s)))/np.sqrt(np.linalg.det(cov_sub+np.multiply(np.diag(s),pow(tau,2)))));
    locFDR[i]= sum1/sum2;
  
  
  return(locFDR);
### FGN Covariance
import numpy as np
np.random.seed(40) 
H = 0.9;
K = 1000;
cov = np.repeat(0.1,(K*K)).reshape(K,K)
for i in range(K):
  for j in range(K):
    cov[i,j] = (0.5)*(pow(abs(abs(i-j)+1),(2*H)) - (2*pow(abs(i-j),(2*H))) + pow(abs(abs(i-j)-1),(2*H)));

pi = 0.3;
tau = 2;
b = 0.2;
H = np.random.binomial(1,pi,size=K);
Z = np.multiply(H,b) + np.matmul(np.linalg.cholesky(cov+np.multiply(np.diag(H),pow(tau,2))),np.random.standard_normal(K));

lcfdr_0 = locFDRS_GWAS_thread(Z,cov,0,b,pi,tau);
lcfdr_1 = locFDRS_GWAS_thread(Z,cov,1,b,pi,tau);
lcfdr_2 = locFDRS_GWAS_thread(Z,cov,2,b,pi,tau);
lcfdr_3 = locFDRS_GWAS_thread(Z,cov,3,b,pi,tau);
lcfdr_4 = locFDRS_GWAS_thread(Z,cov,4,b,pi,tau);
lcfdr_5 = locFDRS_GWAS_thread(Z,cov,5,b,pi,tau);

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
# Note that even in the OO-style, we use `.pyplot.figure` to create the Figure.
fig, (ax1,ax2) = plt.subplots(1,2,figsize=(20, 5))
ax1.scatter((lcfdr_0), (lcfdr_1), label='$T_0$ vs $T_1$')  # Plot some data on the axes.
ax1.scatter((lcfdr_1), (lcfdr_2), label='$T_1$ vs $T_2$')  # Plot more data on the axes...
ax1.scatter((lcfdr_2), (lcfdr_3) , label='$T_2$ vs $T_3$')  # ... and some more.
ax1.scatter((lcfdr_3), (lcfdr_4) , label='$T_3$ vs $T_4$')  # ... and some more.
ax1.scatter((lcfdr_4), (lcfdr_5) , label='$T_4$ vs $T_5$')  # ... and some more.
# ax1.plot((lcfdr_5), (lcfdr_6) , label='T5 vs T6')  # ... and some more.
ax1.set_xlabel('$T_N$')  # Add an x-label to the axes.
ax1.set_ylabel('$T_{N+1}$')  # Add a y-label to the axes.
#ax1.set_title("AR(1), \rho=0.8, Ordering of the locfdrs")  # Add a title to the axes.
ax1.legend(prop={'size': 15});  # Add a legend.


from scipy.stats import gaussian_kde
# Note that even in the OO-style, we use `.pyplot.figure` to create the Figure.
#fig, (ax1) = plt.subplots(1,1,figsize=(10, 12))
density = gaussian_kde(lcfdr_0)
xs = np.linspace(0,1,500)
density.covariance_factor = lambda : .25
density._compute_covariance()
ax2.plot(xs,density(xs),label='$T_0$ ')
######
density = gaussian_kde(lcfdr_1)
xs = np.linspace(0,1,500)
density.covariance_factor = lambda : .25
density._compute_covariance()
ax2.plot(xs,density(xs),label='$T_1$' )
###### 
density = gaussian_kde(lcfdr_2)
xs = np.linspace(0,1,500)
density.covariance_factor = lambda : .25
density._compute_covariance()
ax2.plot(xs,density(xs),label='$T_2$')
######
density = gaussian_kde(lcfdr_3)
xs = np.linspace(0,1,500)
density.covariance_factor = lambda : .25
density._compute_covariance()
ax2.plot(xs,density(xs),label='$T_3$')
######
density = gaussian_kde(lcfdr_4)
xs = np.linspace(0,1,500)
density.covariance_factor = lambda : .25
density._compute_covariance()
ax2.plot(xs,density(xs),label='$T_4$')
######
density = gaussian_kde(lcfdr_5)
xs = np.linspace(0,1,500)
density.covariance_factor = lambda : .25
density._compute_covariance()
ax2.plot(xs,density(xs),label='$T_5$')

ax2.set_xlabel('$T_N$')  # Add an x-label to the axes.
#ax2.set_ylabel('$T_{N+1}$')  # Add a y-label to the axes.
#ax2.set_title("long-range, \rho=0.8, Ordering of the locfdrs")  # Add a title to the axes.
ax2.legend(prop={'size': 15});  # Add a legend.

#plt.plot(figsize=(10,12))
plt.show()

fig.savefig(fname = 'FGN0.9_point&hist.png')

##############################################################################

## AR(1) covariance
np.random.seed(42) 
rho = 0.8;
#B = 1;
K = 1000;
diagonal = np.concatenate((1/(1-pow(rho,2)),np.repeat(((1+pow(rho,2))/(1-pow(rho,2))),K-2),1/(1-pow(rho,2))),axis=None);
off_diag = -rho/(1-pow(rho,2));
prec = np.diag(diagonal);
for d in range(K-1):
  prec[d,d+1]=prec[d+1,d] = off_diag;
cov = np.linalg.inv(prec);

pi = 0.3;
tau = 2;
b = 0.2;
H = np.random.binomial(1,pi,size=K);
Z = np.multiply(H,b) + np.matmul(np.linalg.cholesky(cov+np.multiply(np.diag(H),pow(tau,2))),np.random.standard_normal(K));

lcfdr_0 = locFDRS_GWAS_thread(Z,cov,0,b,pi,tau);
lcfdr_1 = locFDRS_GWAS_thread(Z,cov,1,b,pi,tau);
lcfdr_2 = locFDRS_GWAS_thread(Z,cov,2,b,pi,tau);
lcfdr_3 = locFDRS_GWAS_thread(Z,cov,3,b,pi,tau);
lcfdr_4 = locFDRS_GWAS_thread(Z,cov,4,b,pi,tau);
lcfdr_5 = locFDRS_GWAS_thread(Z,cov,5,b,pi,tau);

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
# Note that even in the OO-style, we use `.pyplot.figure` to create the Figure.
fig, (ax1,ax2) = plt.subplots(1,2,figsize=(20, 5))
ax1.scatter((lcfdr_0), (lcfdr_1), label='$T_0$ vs $T_1$')  # Plot some data on the axes.
ax1.scatter((lcfdr_1), (lcfdr_2), label='$T_1$ vs $T_2$')  # Plot more data on the axes...
ax1.scatter((lcfdr_2), (lcfdr_3) , label='$T_2$ vs $T_3$')  # ... and some more.
ax1.scatter((lcfdr_3), (lcfdr_4) , label='$T_3$ vs $T_4$')  # ... and some more.
ax1.scatter((lcfdr_4), (lcfdr_5) , label='$T_4$ vs $T_5$')  # ... and some more.
# ax1.plot((lcfdr_5), (lcfdr_6) , label='T5 vs T6')  # ... and some more.
ax1.set_xlabel('$T_N$')  # Add an x-label to the axes.
ax1.set_ylabel('$T_{N+1}$')  # Add a y-label to the axes.
#ax1.set_title("AR(1), \rho=0.8, Ordering of the locfdrs")  # Add a title to the axes.
ax1.legend(prop={'size': 15});  # Add a legend.


from scipy.stats import gaussian_kde
# Note that even in the OO-style, we use `.pyplot.figure` to create the Figure.
#fig, (ax1) = plt.subplots(1,1,figsize=(10, 12))
density = gaussian_kde(lcfdr_0)
xs = np.linspace(0,1,500)
density.covariance_factor = lambda : .25
density._compute_covariance()
ax2.plot(xs,density(xs),label='$T_0$ ')
######
density = gaussian_kde(lcfdr_1)
xs = np.linspace(0,1,500)
density.covariance_factor = lambda : .25
density._compute_covariance()
ax2.plot(xs,density(xs),label='$T_1$' )
###### 
density = gaussian_kde(lcfdr_2)
xs = np.linspace(0,1,500)
density.covariance_factor = lambda : .25
density._compute_covariance()
ax2.plot(xs,density(xs),label='$T_2$')
######
density = gaussian_kde(lcfdr_3)
xs = np.linspace(0,1,500)
density.covariance_factor = lambda : .25
density._compute_covariance()
ax2.plot(xs,density(xs),label='$T_3$')
######
density = gaussian_kde(lcfdr_4)
xs = np.linspace(0,1,500)
density.covariance_factor = lambda : .25
density._compute_covariance()
ax2.plot(xs,density(xs),label='$T_4$')
######
density = gaussian_kde(lcfdr_5)
xs = np.linspace(0,1,500)
density.covariance_factor = lambda : .25
density._compute_covariance()
ax2.plot(xs,density(xs),label='$T_5$')

ax2.set_xlabel('$T_N$')  # Add an x-label to the axes.
#ax2.set_ylabel('$T_{N+1}$')  # Add a y-label to the axes.
#ax2.set_title("long-range, \rho=0.8, Ordering of the locfdrs")  # Add a title to the axes.
ax2.legend(prop={'size': 15});  # Add a legend.

#plt.plot(figsize=(10,12))
plt.show()

fig.savefig(fname = 'AR(1)0.8_point&hist.png')

##################################################################################
### Equicorrelated Covariance
np.random.seed(42) 
rho = 0.8
K = 1000
cov = np.repeat(rho,K*K).reshape(K,K);
for i in range(K):
  cov[i,i] = 1;


pi = 0.3;
tau = 2;
b = 0.2;
H = np.random.binomial(1,pi,size=K);
Z = np.multiply(H,b) + np.matmul(np.linalg.cholesky(cov+np.multiply(np.diag(H),pow(tau,2))),np.random.standard_normal(K));

lcfdr_0 = locFDRS_GWAS_thread(Z,cov,0,b,pi,tau);
lcfdr_1 = locFDRS_GWAS_thread(Z,cov,1,b,pi,tau);
lcfdr_2 = locFDRS_GWAS_thread(Z,cov,2,b,pi,tau);
lcfdr_3 = locFDRS_GWAS_thread(Z,cov,3,b,pi,tau);
lcfdr_4 = locFDRS_GWAS_thread(Z,cov,4,b,pi,tau);
lcfdr_5 = locFDRS_GWAS_thread(Z,cov,5,b,pi,tau);

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
# Note that even in the OO-style, we use `.pyplot.figure` to create the Figure.
fig, (ax1,ax2) = plt.subplots(1,2,figsize=(20, 5))
ax1.scatter((lcfdr_0), (lcfdr_1), label='$T_0$ vs $T_1$')  # Plot some data on the axes.
ax1.scatter((lcfdr_1), (lcfdr_2), label='$T_1$ vs $T_2$')  # Plot more data on the axes...
ax1.scatter((lcfdr_2), (lcfdr_3) , label='$T_2$ vs $T_3$')  # ... and some more.
ax1.scatter((lcfdr_3), (lcfdr_4) , label='$T_3$ vs $T_4$')  # ... and some more.
ax1.scatter((lcfdr_4), (lcfdr_5) , label='$T_4$ vs $T_5$')  # ... and some more.
# ax1.plot((lcfdr_5), (lcfdr_6) , label='T5 vs T6')  # ... and some more.
ax1.set_xlabel('$T_N$')  # Add an x-label to the axes.
ax1.set_ylabel('$T_{N+1}$')  # Add a y-label to the axes.
#ax1.set_title("AR(1), \rho=0.8, Ordering of the locfdrs")  # Add a title to the axes.
ax1.legend(prop={'size': 15});  # Add a legend.


from scipy.stats import gaussian_kde
# Note that even in the OO-style, we use `.pyplot.figure` to create the Figure.
#fig, (ax1) = plt.subplots(1,1,figsize=(10, 12))
density = gaussian_kde(lcfdr_0)
xs = np.linspace(0,1,500)
density.covariance_factor = lambda : .25
density._compute_covariance()
ax2.plot(xs,density(xs),label='$T_0$ ')
######
density = gaussian_kde(lcfdr_1)
xs = np.linspace(0,1,500)
density.covariance_factor = lambda : .25
density._compute_covariance()
ax2.plot(xs,density(xs),label='$T_1$' )
###### 
density = gaussian_kde(lcfdr_2)
xs = np.linspace(0,1,500)
density.covariance_factor = lambda : .25
density._compute_covariance()
ax2.plot(xs,density(xs),label='$T_2$')
######
density = gaussian_kde(lcfdr_3)
xs = np.linspace(0,1,500)
density.covariance_factor = lambda : .25
density._compute_covariance()
ax2.plot(xs,density(xs),label='$T_3$')
######
density = gaussian_kde(lcfdr_4)
xs = np.linspace(0,1,500)
density.covariance_factor = lambda : .25
density._compute_covariance()
ax2.plot(xs,density(xs),label='$T_4$')
######
density = gaussian_kde(lcfdr_5)
xs = np.linspace(0,1,500)
density.covariance_factor = lambda : .25
density._compute_covariance()
ax2.plot(xs,density(xs),label='$T_5$')

ax2.set_xlabel('$T_N$')  # Add an x-label to the axes.
#ax2.set_ylabel('$T_{N+1}$')  # Add a y-label to the axes.
#ax2.set_title("long-range, \rho=0.8, Ordering of the locfdrs")  # Add a title to the axes.
ax2.legend(prop={'size': 15});  # Add a legend.

#plt.plot(figsize=(10,12))
plt.show()

fig.savefig(fname = 'Equi0.8_point&hist.png')
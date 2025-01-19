1) Running the Code "simu_or_lcfdrn_scth.py": These instructions are for generating Table 1
   - Guideline for generating Table 1
   - Set K = 1000 at line 92
   - Set pi = 0.1 at line 132
   - set ndraws = 3000 at line 135
   - set rep = 3000 at line 168
   - set filename 'RC_01_1K.npy' to save simulation data.
   - Then run the following Python code to get dataset "RC_or_01_1K.csv" in CSV format
        - import numpy as np
        - arr = np.load("RC_01_1K.npy")
        - np.savetxt("RC_or_01_1K.csv", np.transpose(np.concatenate((arr[0],arr[1],arr[2],arr[3]))),fmt='%1.4f', delimiter=",")
   - The entries of the CSV file exactly match the entries of Table 1 in the paper.
2) Running the Code "simu_or_lcfdrn_scth.py": These instructions are for generating Table S1
   - Guideline for generating Table S1
   - Set K = 10000 at line 92
   - Set pi = 0.1 at line 132
   - set ndraws = 500 at line 135
   - set rep = 500 at line 168
   - set filename 'RC_01_10K.npy' to save simulation data.
   - Then run the following Python code to get dataset "RC_or_01_10K.csv" in CSV format
        - import numpy as np
        - arr = np.load("RC_01_10K.npy")
        - np.savetxt("RC_or_01_10K.csv", np.transpose(np.concatenate((arr[0],arr[1],arr[2],arr[3]))),fmt='%1.4f', delimiter=",")
   - The entries of the CSV file exactly match the entries of Table S1 in the paper.
3) Running the Code "simu_or_lcfdrn_scth.py": These instructions are for generating Table S2
   - Guideline for generating Table S2
   - Set K = 10000 at line 92
   - Set pi = 0.3 at line 132
   - set ndraws = 500 at line 135
   - set rep = 500 at line 168
   - set filename 'RC_03_10K.npy' to save simulation data.
   - Then run the following Python code to get dataset "RC_or_03_10K.csv" in CSV format
        - import numpy as np
        - arr = np.load("RC_03_10K.npy")
        - np.savetxt("RC_or_03_10K.csv", np.transpose(np.concatenate((arr[0],arr[1],arr[2],arr[3]))),fmt='%1.4f', delimiter=",")
   - The entries of the CSV file exactly match the entries of Table S2 in the paper.
4) Running the Code "simu_or_lcfdrn_scth.py": These instructions are for generating Table S3
   - Guideline for generating Table S3
   - Set K = 1000 at line 92
   - Set pi = 0.3 at line 132
   - set ndraws = 3000 at line 135
   - set rep = 3000 at line 168
   - set filename 'RC_03_1K.npy' to save simulation data.
   - Then run the following Python code to get dataset "RC_or_03_1K.csv" in CSV format
        - import numpy as np
        - arr = np.load("RC_03_1K.npy")
        - np.savetxt("RC_or_03_1K.csv", np.transpose(np.concatenate((arr[0],arr[1],arr[2],arr[3]))),fmt='%1.4f', delimiter=",")
   - The entries of the CSV file exactly match the entries of Table S3 in the paper.
5) Running the Code "simu_dd_lcfdrn_scth.py": These instructions are for generating Table 2
   - Guideline for generating Table 2
   - Set pi = 0.1 at line 197
   - set filename 'RC_dd_01_15K.npy' to save simulation data.
   - Set pi = 0.3 at line 197
   - set filename 'RC_dd_03_15K.npy' to save simulation data.
   - Then run the following Python code to get dataset "RC_dd_03_01_15K.csv" in CSV format
        - import numpy as np
        - arr1 = np.load("RC_dd_03_15K.npy")
        - arr2 = np.load("RC_dd_01_15K.npy")
        - array_3d =np.array([np.transpose(np.concatenate((arr1[0],arr1[1]))),np.transpose(np.concatenate((arr2[0],arr2[1])))])
        - array_2d = array_3d.reshape(-1, array_3d.shape[-1])
        - np.savetxt("RC_dd_03_01_15K.csv", array_2d,fmt='%1.4f', delimiter=",")
   - The entries of the CSV file exactly match the entries of Table 2 in the paper.
6) Running the Code "simu_or_compare_lcfdrn_vs_lcfdr.py": These instructions are for generating Table S4
   - Guideline for generating Table S4
   - Set K = 1400 at line 128
   - Set pi = 0.1 at line 132
   - Set N = 3, ndraws = 3000 in line 135 and rep = 3000 in line 175.
   - set filename 'RC_com_pow_lcfdrNvslcfdr_pi_01.npy' in line 214 to save simulation data.
   - Then run the following Python code to get dataset "RC_com_pow_lcfdrNvslcfdr_pi_01.csv" in CSV format
        - import numpy as np
        - load = np.load("RC_com_pow_lcfdrNvslcfdr_pi_01.npy");
        - np.savetxt("RC_com_pow_lcfdrNvslcfdr_pi_01.csv", np.transpose(np.concatenate((load[0],load[1],load[2]))),fmt='%1.4f', delimiter=",")
   - The entries of the CSV file exactly match the entries of Table S4 in the paper.
7) Running the Code "simu_or_compare_lcfdrn_vs_lcfdr.py": These instructions are for generating Table 3
   - Guideline for generating Table S4
   - Set K = 700 at line 128
   - Set pi = 0.3 at line 132
   - Set N = 5, ndraws = 1000 in line 135 and rep = 1000 in line 175.
   - set filename 'RC_com_pow_lcfdrNvslcfdr_pi_03.npy' in line 214 to save simulation data.
   - Then run the following Python code to get dataset "RC_com_pow_lcfdrNvslcfdr_pi_03.csv" in CSV format
        - import numpy as np
        - load = np.load("RC_com_pow_lcfdrNvslcfdr_pi_03.npy");
        - np.savetxt("RC_com_pow_lcfdrNvslcfdr_pi_03.csv", np.transpose(np.concatenate((load[0],load[1],load[2]))),fmt='%1.4f', delimiter=",")
   - The entries of the CSV file exactly match the entries of Table 3 in the paper.
8) Running the Code "simu_orsc_compare_lcfdrn.py": These instructions are for generating Table S5
   - Guideline for generating Table S5
   - set filename 'RC_orsc_AR1_03_1K.npy' in line 150 to save simulation data.
   - Then run the following Python code to get dataset "RC_com_pow_lcfdrN_AR1_03.csv" in CSV format
        - import numpy as np
        - load = np.load("RC_orsc_AR1_03_1K.npy");
        - np.savetxt("RC_com_pow_lcfdrN_AR1_03.csv", np.transpose(np.concatenate((load[0],load[1],load[2]))),fmt='%1.4f', delimiter=",")
   - The entries of the CSV file exactly match the entries of Table S5 in the paper.
9) Running the Code "simu_orsc_compare_lcfdrn.py": These instructions are for generating Table S6
   - Guideline for generating Table S6
   - set filename 'RC_orsc_eq_03_1K.npy' in line 150 to save simulation data.
   - Then run the following Python code to get dataset "RC_com_pow_lcfdrN_eq_03.csv" in CSV format
        - import numpy as np
        - load = np.load("RC_orsc_eq_03_1K.npy");
        - np.savetxt("RC_com_pow_lcfdrN_eq_03.csv", np.transpose(np.concatenate((load[0],load[1],load[2]))),fmt='%1.4f', delimiter=",")
   - The entries of the CSV file exactly match the entries of Table S6 in the paper.
10) Running the notebook "real_data_biometrics.ipynb":
   - This notebook contains instructions on applying our method to your dataset, our example is given with a simulated dataset (y, X).
   - Input data:
     - n by 1 vector of phenotype y
     - n by p matrix of covariates X with n>p.
   - ultimately the notebook shows the number of rejections by our method and off-the-shelf competitors.
11) Running the Code "biometrics figures.py": This code generates all the figures in Figure 1. Run the code without any modifications and you will get the figures in PNG format.
12) Reproducing the GWAS example in the paper
    - Running the Code "LD_Pruning.ipynb": This code prepares data for GWAS. Step by step guide is given in the notebook with the output.
    - Running the code "GWAS_biometrics.ipynb": This code runs GWAS analysis with input data prepared in the last step. Output is given in the notebook.
    - Running the code "biometrics_GWAS_analysis.ipynb": This code can be used to generate Figure 2 and Table 4. Step by step guide is given in the notebook with the output.
13) The folder "ancillary data" contains the data/functions that you need to keep in the working directory to run the codes.
14) The folder "paper data" contains the data that you will get after running our code.

   

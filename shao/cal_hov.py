import numpy as np
import sys, os
sys.path.insert(1,'../')
from utils.vvmtools import newVVMtools
import config

exp = 'op_1'
path = f'{config.vvmPath}/{exp}/'
newVVMTol = newVVMtools(path)
nt = 721
newVVMTol.DIM['t'] = 5 + np.arange(nt)*2/60 #hhour

# example
# domain_range=(None, None, None, None, None, None), # (k1, k2, j1, j2, i1, i2)
reg       = 'all' #'left', 'right'
drange = list(newVVMTol.get_domain_range(reg))
ilev   = np.argmin(np.abs(newVVMTol.DIM['zc']-300.))
ilev   = 1
drange[:2] = [ilev, ilev+1]
lev_str = '{:.0f}m'.format(newVVMTol.DIM['zc'][ilev])

args    = {\
           'time_steps':list(range(nt)),\
           'domain_range':drange,\
           'compute_mean':True,\
           'axis':0,\
           'cores':4,\
          }

tr01       =  newVVMTol.get_var_parallel('tr01',  **args)
tr02       =  newVVMTol.get_var_parallel('INERT', **args)
tr03       =  newVVMTol.get_var_parallel('tr02',  **args)
chem_no    =  newVVMTol.get_var_parallel('NO',    **args) 
no2        =  newVVMTol.get_var_parallel('NO2',   **args)
no3        =  newVVMTol.get_var_parallel('NO3',   **args)
o3         =  newVVMTol.get_var_parallel('O3',    **args)

np.savez(f'{config.datPath}/xt_hov_{exp}_{reg}_{lev_str}.npz',\
         x = newVVMTol.DIM['xc']/1e3, \
         y = newVVMTol.DIM['yc']/1e3, \
         z = newVVMTol.DIM['zc']/1e3, \
         t = newVVMTol.DIM['t'], \
         tr01_xt = tr01, \
         tr02_xt = tr02, \
         tr03_xt = tr03, \
         no = chem_no, \
         no2 = no2, \
         no3 = no3, \
         o3  = o3, \
         target_lev = lev_str,\
        )







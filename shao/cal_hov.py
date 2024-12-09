import numpy as np
import sys, os
sys.path.insert(1,'../')
from utils.vvmtools import newVVMtools
import config

exp = 'pbl_op_8dth_6tr'
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
           'cores':5,\
          }

data_dict = {}

# calculate tracer 1 to 6
for i in range(1,7):
  varn = f'tr{i:02d}'
  print(varn)
  data_dict[f'{varn}_xt'] = newVVMTol.get_var_parallel(varn, **args)

for varn in ['NO', 'NO2', 'INERT', 'O3']:
  print(varn)
  data_dict[f'{varn.lower()}_xt'] = newVVMTol.get_var_parallel(varn, **args)

for varn in ['u', 'v']:
  print(varn)
  data_dict[f'{varn.lower()}_xt'] = newVVMTol.get_var_parallel(varn, **args)

np.savez(f'{config.datPath}/xt_hov_{exp}_{reg}_{lev_str}.npz',\
         x = newVVMTol.DIM['xc']/1e3, \
         y = newVVMTol.DIM['yc']/1e3, \
         z = newVVMTol.DIM['zc']/1e3, \
         t = newVVMTol.DIM['t'], \
         target_lev = lev_str,\
         **data_dict\
        )







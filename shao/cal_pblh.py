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
reg       = 'all' #'left', 'right'
drange = newVVMTol.get_domain_range(reg)


# calculate tke, enstrophy, pblh_th0p5, pblh_maxgrad
args = {\
          'time_steps': list(range(nt)), \
          'func_config': {'domain_range':drange},\
          'cores': 5,\
       }

tke       = newVVMTol.func_time_parallel(newVVMTol.cal_TKE, **args)
enstrophy = newVVMTol.func_time_parallel(newVVMTol.cal_enstrophy, **args)
pblh_th0p5 = newVVMTol.func_time_parallel(newVVMTol.cal_pblh_0p5, **args)
pblh_maxgrad = newVVMTol.func_time_parallel(newVVMTol.cal_pblh_maxgrad, **args)

# calculate pblh_ens, pblh_tke
args = {\
          'time_steps': list(range(nt)), \
          #'func_config': {'domain_range':drange, 'ens_threshold':1.e-5},\
          'cores': 5,\
       }

pblh_ens = newVVMTol.func_time_parallel(\
           func        = newVVMTol.cal_pblh_ens, \
           func_config = {'domain_range':drange, 'threshold':1.e-5},\
           **args\
           )

pblh_tke = newVVMTol.func_time_parallel(\
           func        = newVVMTol.cal_pblh_tke, \
           func_config ={'domain_range':drange, 'threshold':0.1},\
           **args\
           )

# calculate read tracer and chemicals
args  = {\
          'time_steps': list(range(nt)),\
          'domain_range': drange,\
          'compute_mean': True,\
          'axis': (1,2),\
          'cores': 4,\
        }

tr01       =  newVVMTol.get_var_parallel( 'tr01', **args)
tr02       =  newVVMTol.get_var_parallel( 'INERT', **args)
tr03       =  newVVMTol.get_var_parallel( 'tr02', **args)


# save data
np.savez(f'{config.datPath}/tz_series_{exp}_{reg}.npz',\
         x = newVVMTol.DIM['xc']/1e3, \
         y = newVVMTol.DIM['yc']/1e3, \
         z = newVVMTol.DIM['zc']/1e3, \
         t = newVVMTol.DIM['t'], \
         tke_tz = tke.T, \
         enstrophy_tz = enstrophy.T, \
         tr01_tz = tr01.T, \
         tr02_tz = tr02.T, \
         tr03_tz = tr03.T, \
         pblh_th0p5_1d = pblh_th0p5/1e3, \
         pblh_maxgrad_1d = pblh_maxgrad/1e3,\
         pblh_ens = pblh_ens/1e3,\
         pblh_tke = pblh_tke/1e3,\
        )







import numpy as np
import sys, os
sys.path.insert(1,'../')
from utils.vvmtools import newVVMtools
import config

exp = 'pbl_op_8dth_6tr'
exp = 'pbl_op_11dth'
exp = 'pbl_ou'
path = f'{config.vvmPath}/{exp}/'
newVVMTol = newVVMtools(path)
nt = 721
newVVMTol.DIM['t'] = 5 + np.arange(nt)*2/60 #hhour

# example
reg       = 'all' #'left', 'right'
drange = newVVMTol.get_domain_range(reg)

data_dict = {}


# calculate tke, enstrophy, pblh_th0p5, pblh_maxgrad
args = {\
          'time_steps': list(range(nt)), \
          'func_config': {'domain_range':drange},\
          'cores': 5,\
       }

data_dict['tke_tz']       = np.transpose(newVVMTol.func_time_parallel(newVVMTol.cal_TKE, **args))
data_dict['enstrophy_tz'] = np.transpose(newVVMTol.func_time_parallel(newVVMTol.cal_enstrophy, **args))
data_dict['pblh_th0p5_1d'] = newVVMTol.func_time_parallel(newVVMTol.cal_pblh_0p5, **args)
data_dict['pblh_maxgrad_1d'] = newVVMTol.func_time_parallel(newVVMTol.cal_pblh_maxgrad, **args)
data_dict['wth_tz']          = np.transpose(newVVMTol.func_time_parallel(newVVMTol.cal_wpthpbar, **args))

# calculate pblh_ens, pblh_tke
args = {\
          'time_steps': list(range(nt)), \
          #'func_config': {'domain_range':drange, 'ens_threshold':1.e-5},\
          'cores': 5,\
       }

data_dict['pblh_ens'] = 1e-3 * newVVMTol.func_time_parallel(\
           func        = newVVMTol.cal_pblh_ens, \
           func_config = {'domain_range':drange, 'threshold':1.e-5},\
           **args\
           )

data_dict['pblh_tke'] = 1e-3 * newVVMTol.func_time_parallel(\
           func        = newVVMTol.cal_pblh_tke, \
           func_config ={'domain_range':drange, 'threshold':0.08},\
           **args\
           )

pblh_wth3 = newVVMTol.func_time_parallel(\
            func       = newVVMTol.cal_pblh_wpthpbar, \
            func_config = {'domain_range':drange, 'threshold':1e-3},\
            **args\
            )
data_dict['pblh_wth_p2n'] = pblh_wth3[:,0]/1e3
data_dict['pblh_wth_min'] = pblh_wth3[:,1]/1e3
data_dict['pblh_wth_n2p'] = pblh_wth3[:,2]/1e3



# calculate read tracer and chemicals
args  = {\
          'time_steps': list(range(nt)),\
          'domain_range': drange,\
          'compute_mean': True,\
          'axis': (1,2),\
          'cores': 4,\
        }

## # calculate tracer 1 to 6
## for i in range(1,7):
##   varn = f'tr{i:02d}'
##   print(varn)
##   data_dict[f'{varn}_tz'] = np.transpose(newVVMTol.get_var_parallel(varn, **args))

for varn in ['NO', 'NO2', 'INERT', 'O3']:
  print(varn)
  data_dict[f'{varn.lower()}_tz'] = np.transpose(newVVMTol.get_var_parallel(varn, **args))

for varn in ['u', 'v', 'w', 'th']:
  print(varn)
  data_dict[f'{varn.lower()}_tz'] = np.transpose(newVVMTol.get_var_parallel(varn, **args))

# save data
np.savez(f'{config.datPath}/tz_series_{exp}_{reg}.npz',\
         x = newVVMTol.DIM['xc']/1e3, \
         y = newVVMTol.DIM['yc']/1e3, \
         z = newVVMTol.DIM['zc']/1e3, \
         t = newVVMTol.DIM['t'], \
         **data_dict\
        )







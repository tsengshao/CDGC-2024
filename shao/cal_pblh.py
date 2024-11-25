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
wth          = newVVMTol.func_time_parallel(newVVMTol.cal_wpthpbar, **args)

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
           func_config ={'domain_range':drange, 'threshold':0.08},\
           **args\
           )

pblh_wth3 = newVVMTol.func_time_parallel(\
            func       = newVVMTol.cal_pblh_wpthpbar, \
            func_config = {'domain_range':drange, 'threshold':1e-3},\
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
tr02       =  newVVMTol.get_var_parallel( 'tr02', **args)
tr03       =  newVVMTol.get_var_parallel( 'tr03', **args)
tr04       =  newVVMTol.get_var_parallel( 'tr04', **args)
tr05       =  newVVMTol.get_var_parallel( 'tr05', **args)
tr06       =  newVVMTol.get_var_parallel( 'tr06', **args)

no2       =  newVVMTol.get_var_parallel( 'NO2', **args)
no        =  newVVMTol.get_var_parallel( 'NO', **args)
INERT     =  newVVMTol.get_var_parallel( 'INERT', **args)



# save data
np.savez(f'{config.datPath}/tz_series_{exp}_{reg}.npz',\
         x = newVVMTol.DIM['xc']/1e3, \
         y = newVVMTol.DIM['yc']/1e3, \
         z = newVVMTol.DIM['zc']/1e3, \
         t = newVVMTol.DIM['t'], \
         tke_tz = tke.T, \
         enstrophy_tz = enstrophy.T, \
         wth_tx  = wth.T, \
         tr01_tz = tr01.T, \
         tr02_tz = tr02.T, \
         tr03_tz = tr03.T, \
         tr04_tz = tr04.T, \
         tr05_tz = tr05.T, \
         tr06_tz = tr06.T, \
         no2_tz  = no2.T, \
         no_tz   = no.T, \
         inert_tz = INERT.T, \
         pblh_th0p5_1d = pblh_th0p5/1e3, \
         pblh_maxgrad_1d = pblh_maxgrad/1e3,\
         pblh_ens = pblh_ens/1e3,\
         pblh_tke = pblh_tke/1e3,\
         pblh_wth_p2n = pblh_wth3[:,0]/1e3,\
         pblh_wth_min = pblh_wth3[:,1]/1e3,\
         pblh_wth_n2p = pblh_wth3[:,2]/1e3,\
        )







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
tke = newVVMTol.func_time_parallel(\
          func       = newVVMTol.cal_TKE, \
          time_steps = list(range(nt)), \
          func_config = {'domain_range':drange},\
          cores = 5,\
         )

enstrophy = newVVMTol.func_time_parallel(\
          func       = newVVMTol.cal_enstrophy, \
          time_steps = list(range(nt)), \
          func_config = {'domain_range':drange},\
          cores = 5,\
         )

pblh_th0p5 = newVVMTol.func_time_parallel(\
              func       = newVVMTol.cal_pblh_0p5, \
              time_steps = list(range(nt)), \
              func_config = {'domain_range':drange},\
              cores = 5,\
             )

pblh_maxgrad = newVVMTol.func_time_parallel(\
              func       = newVVMTol.cal_pblh_maxgrad, \
              time_steps = list(range(nt)), \
              func_config = {'domain_range':drange},\
              cores = 5,\
             )

tr01       =  newVVMTol.get_var_parallel(
               'tr01', \
               time_steps=list(range(nt)),\
               domain_range=drange,\
               compute_mean=True,\
               axis=(1,2),\
               cores=4,\
              )

tr02       =  newVVMTol.get_var_parallel(
               'INERT', \
               time_steps=list(range(nt)),\
               domain_range=drange,\
               compute_mean=True,\
               axis=(1,2),\
               cores=4,\
              )

tr03       =  newVVMTol.get_var_parallel(
               'tr02', \
               time_steps=list(range(nt)),\
               domain_range=drange,\
               compute_mean=True,\
               axis=(1,2),\
               cores=4,\
              )

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
        )







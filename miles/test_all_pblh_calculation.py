import numpy as np
import sys, os
sys.path.insert(1,'../')
from utils.vvmtools import newVVMtools
import config


if __name__=='__main__':
    exp = 'op_1'
    path = f'{config.vvmPath}/{exp}/'
    vvm = newVVMtools(path)
    # test example for
    reg       = 'all' #'left', 'right'
    drange = vvm.get_domain_range(reg)
    pblh_ens = vvm.func_time_parallel(\
               func       = vvm.cal_pblh_ens, \
               time_steps = np.arange(360), \
               func_config = {'domain_range':drange, 'threshold':1.e-5},\
               cores = 5,\
              )
    pblh_tke = vvm.func_time_parallel(\
               func       = vvm.cal_pblh_tke, \
               time_steps = np.arange(360), \
               func_config = {'domain_range':drange, 'threshold':0.1},\
               cores = 5,\
              )
    pblh_th0p5 = vvm.func_time_parallel(\
                 func       = vvm.cal_pblh_0p5, \
                 time_steps = np.arange(360), \
                 func_config = {'domain_range':drange,},\
                 cores = 5,\
              )
    pblh_maxgrad = vvm.func_time_parallel(\
                 func       = vvm.cal_pblh_maxgrad, \
                 time_steps = np.arange(360), \
                 func_config = {'domain_range':drange,},\
                 cores = 5,\
              )



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
    drange = list(vvm.get_domain_range(reg))
    ilev   = 1
    drange[:2] = [ilev, ilev+1]
    lev_str = '{:.0f}m'.format(vvm.DIM['zc'][ilev])

    args    = {\
               'time_steps':list(range(vvm.nt)),\
               'domain_range':drange,\
               'compute_mean':True,\
               'axis':0,\
               'cores':4,\
              }

    tr01  = vvm.get_var_parallel('tr01',  **args)

    np.save('data/tr01.npy',tr01)

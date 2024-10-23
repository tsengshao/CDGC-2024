import numpy as np
import sys, os
sys.path.insert(1,'../')
from utils.vvmtools import newVVMtools
from utils.plottools import dataPlotters
import config
import matplotlib.pyplot as plt

import numpy as np
import sys, os
sys.path.insert(1,'../')
from utils.vvmtools import newVVMtools
import config

if __name__=='__main__':
    exp = 'op_1'
    path = f'{config.vvmPath}/{exp}/'
    figpath = f'./fig/'
    vvm = newVVMtools(path)
    # test example for
    reg       = 'all' #'left', 'right'
    drange = vvm.get_domain_range(reg)
    '''
    ens = vvm.func_time_parallel(\
          func       = vvm.cal_enstrophy,\
               time_steps = np.arange(vvm.nt), \
               func_config = {'domain_range':drange,},\
               cores = 5,\
              ).T
    print(ens.shape)

    pblh_ens = vvm.func_time_parallel(\
               func       = vvm.cal_pblh_ens, \
               time_steps = np.arange(vvm.nt), \
               func_config = {'domain_range':drange, 'ens_threshold':1.e-5},\
               cores = 5,\
              )
    print(pblh_ens.shape)
    pblh_tke = vvm.func_time_parallel(\
               func       = vvm.cal_pblh_tke, \
               time_steps = np.arange(vvm.nt), \
               func_config = {'domain_range':drange, 'tke_threshold':0.1},\
               cores = 5,\
              )
    pblh_th0p5 = vvm.func_time_parallel(\
                 func       = vvm.cal_pblh_0p5, \
                 time_steps = np.arange(vvm.nt), \
                 func_config = {'domain_range':drange,},\
                 cores = 5,\
              )
    pblh_maxgrad = vvm.func_time_parallel(\
                 func       = vvm.cal_pblh_maxgrad, \
                 time_steps = np.arange(vvm.nt), \
                 func_config = {'domain_range':drange,},\
                 cores = 5,\
              )
    '''
    pblh_ens=np.load('./data/pblh_ens.npy')
    pblh_tke=np.load('./data/pblh_tke.npy')
    pblh_maxgrad=np.load('./data/pblh_maxgrad.npy')
    ens=np.load('./data/ens.npy')
    pblh_th0p5=np.load('./data/pblh_th0p5.npy')
    data_dims = {'x':vvm.DIM['xc']/1e3,\
                 'y':vvm.DIM['yc']/1e3,\
                 'z':vvm.DIM['zc']/1e3,\
                 't':[np.datetime64('2024-01-01 05:00:00')\
                      +np.timedelta64(2*tt,'m') for tt in range(vvm.nt)],\
                }    
    tick_dims = {'x':np.arange(data_dims['x'][0], data_dims['x'][-1]+0.0001, 6.4),\
                 'y':np.arange(data_dims['y'][0], data_dims['y'][-1]+0.0001, 6.4),\
                 'z':np.arange(0, data_dims['z'].max()+0.00001,0.2),\
                 't':1,\
                }
    data_dim_units = {'x':'km',\
                      'y':'km',\
                      'z':'km',\
                      't':'hour',\
                     }

    dplot = dataPlotters(exp, figpath, data_dims, data_dim_units, tick_dims)
    fig, ax = dplot.draw_zt(data = ens, \
                          levels = np.linspace(0,1e-4,21), \
                          extend = 'max', \
                          pblh_dicts={'th0p5': pblh_th0p5,\
                                      'maxgrad': pblh_maxgrad,\
                                      'enstrophy': pblh_ens,\
                                      'TKE': pblh_tke,\
                                     },\
                          title_left  = 'tr01 [normalize by maximum]', \
                          title_right = f'{reg}', \
                          xlim        = (np.datetime64('2024-01-01 06:00:00'),\
                                         np.datetime64('2024-01-01 18:00:00')),\
                          #ylim        = (0, 2),\
                          figname     = 'tr01.png',\
                   )
    plt.show()


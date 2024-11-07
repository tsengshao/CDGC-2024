import numpy as np
import sys, os
sys.path.insert(1,'../')
from utils.vvmtools import newVVMtools
from VVMTools.vvmtools.plot import DataPlotter
#from utils.plottools import dataPlotters
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

    pblh={}
    for var in ['pblh_th0p5','pblh_maxgrad','pblh_tke','pblh_ens','pblh_wth_p','pblh_wth_min','pblh_wth_n']:
        pblh[var]=np.load(f'./data/{var}.{exp}.npy')/1.e3
    wth = np.load(f'data/wpthpbar.{exp}.npy')
    data_dims = {'x':vvm.DIM['xc']/1.e3,\
                 'y':vvm.DIM['yc']/1.e3,\
                 'z':vvm.DIM['zc']/1.e3,\
                 't':np.arange(721)*np.timedelta64(2,'m')+np.datetime64('2024-01-01 05:00:00'),\
                }
    tick_dims = {'x':np.arange(data_dims['x'][0], data_dims['x'][-1]+0.0001, 6.4),\
                 'y':np.arange(data_dims['y'][0], data_dims['y'][-1]+0.0001, 6.4),\
                 'z':np.arange(0, data_dims['z'].max()+0.00001,0.2),\
                 't':[np.datetime64(f'2024-01-01 00:00')+np.timedelta64(i,'h') for i in [5,6,12,18,24]],\
                }
    data_dim_units = {'x':'km',\
                      'y':'km',\
                      'z':'km',\
                      't':'LT',\
                     } 
    #dplot = dataPlotters(exp, figpath, data_dims, data_dim_units, tick_dims)
    dplot = DataPlotter(exp, figpath, data_dims, data_dim_units, tick_dims)
    #plot 1: entire time span
    fig, ax, cax = dplot.draw_zt(data = wth.T, \
                          levels = np.linspace(-0.04,0.04,17), \
                          extend = 'both', \
                          pblh_dicts = pblh,\
                          title_left  = '$\overline{w^{\prime}\\theta^{\prime}}(K m s^{-1})$', \
                          title_right = f'{reg}', \
                          #xlim        = (np.datetime64('2024-01-01 06:00:00'),\
                          #               np.datetime64('2024-01-01 18:00:00')),\
                          #ylim        = (0, 2),\
                          figname     = 'entire_time_span.png',\
                   )
    

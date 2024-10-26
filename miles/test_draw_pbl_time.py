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

    pblh_ens=np.load('./data/pblh_ens.npy')/1.e3
    pblh_tke=np.load('./data/pblh_tke.npy')/1.e3
    pblh_maxgrad=np.load('./data/pblh_maxgrad.npy')/1.e3
    pblh_th0p5=np.load('./data/pblh_th0p5.npy')/1.e3
    ens=np.load('./data/ens.npy')
    data_dims = {'x':vvm.DIM['xc']/1e3,\
                 'y':vvm.DIM['yc']/1e3,\
                 'z':vvm.DIM['zc']/1e3,\
                 't':np.datetime64('2024-01-01 05:00:00')+np.arange(vvm.nt)*np.timedelta64(2,'m')
                }    
    tick_dims = {'x':np.arange(data_dims['x'][0], data_dims['x'][-1]+0.0001, 6.4),\
                 'y':np.arange(data_dims['y'][0], data_dims['y'][-1]+0.0001, 6.4),\
                 'z':np.arange(0, data_dims['z'].max()+0.00001,0.2),\
                 't':[np.datetime64('2024-01-01 00:00:00')+np.timedelta64(hr,'h') for hr in [5,6,12,18,24]],\
                }
    data_dim_units = {'x':'km',\
                      'y':'km',\
                      'z':'km',\
                      't':'LST',\
                     }

    dplot = dataPlotters(exp, figpath, data_dims, data_dim_units, tick_dims)
    #plot 1: entire time span
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
                          #xlim        = (np.datetime64('2024-01-01 06:00:00'),\
                          #               np.datetime64('2024-01-01 18:00:00')),\
                          #ylim        = (0, 2),\
                          figname     = 'entire_time_span.png',\
                   )
    
    #plot 2: daytime
    plt.close()
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
                          #xlim        = (8,17),\
                          #ylim        = (0, 2),\
                          figname     = 'daytime.png',\
                   )
    #plot 3: 22hr
    plt.close()
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
                          xlim        = (np.datetime64('2024-01-01 05:00:00'),\
                                         np.datetime64('2024-01-02 03:00:00')),\
                          #xlim        = (8,17),\
                          #ylim        = (0, 2),\
                          figname     = 'entire_22hr.png',\
                   )
    '''
    #plot 3: noon with minute ticks
    dplot.DIM_UNITS['t'] = 'minute'
    dplot.DIM_TICKS['t'] = 10
    dplot.update_time_mapper()
    plt.close()
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
                          xlim        = (np.datetime64('2024-01-01 11:00:00'),\
                                         np.datetime64('2024-01-01 13:00:00')),\
                          #ylim        = (0, 2),\
                          figname     = 'noon.png',\
                   )

'''

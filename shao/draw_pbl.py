import numpy as np
import sys, os
sys.path.insert(1,'../')
from utils.vvmtools import newVVMtools
from utils.plottools import dataPlotters
import config
import matplotlib.pyplot as plt

exp = 'op_1'
path = f'{config.vvmPath}/{exp}/'
figpath = f'./fig/'
reg  = 'all'

data = np.load(f'{config.datPath}/tz_series_{exp}_{reg}.npz')
data_dims = {'x':data['x'],\
             'y':data['y'],\
             'z':data['z'],\
             't':data['t'],\
            }
tick_dims = {'x':np.arange(data['x'][0], data['x'][-1]+0.0001, 6.4),\
             'y':np.arange(data['x'][0], data['y'][-1]+0.0001, 6.4),\
             'z':np.arange(0, data['z'].max()+0.00001,0.2),\
             't':np.arange(0, data['t'].max()+1,1),\
            }
tke             = data['tke_tz']
enstrophy       = data['enstrophy_tz']
pblh_th0p5_1d   = data['pblh_th0p5_1d']
pblh_maxgrad_1d = data['pblh_maxgrad_1d']
tr01            = data['tr01_tz']
tr02            = data['tr02_tz']
tr03            = data['tr03_tz']

dplot = dataPlotters(exp, figpath, data_dims, tick_dims)
fig, ax = dplot.draw_zt(data = tr01, \
                      levels = np.arange(0,2.1,0.2), \
                      extend = 'max', \
                      pblh_dicts={'th0p5': pblh_th0p5_1d,\
                                  'maxgrad': pblh_maxgrad_1d,\
                                 },\
                      title_left  = 'tr01', \
                      title_right = f'{reg}', \
                      figname     = 'tr01.png',\
               )
plt.show()


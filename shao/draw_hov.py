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
lev_str = '20m'

data = np.load(f'{config.datPath}/xt_hov_{exp}_{reg}_{lev_str}.npz')
data_dims = {'x':data['x'],\
             'y':data['y'],\
             'z':data['z'],\
             't':data['t'],\
            }
tick_dims = {'x':np.linspace(data['x'][0], data['x'][-1], 5),\
             'y':np.arange(data['x'][0], data['y'][-1]+0.0001, 6.4),\
             'z':np.arange(0, data['z'].max()+0.00001,0.2),\
             't':np.append([5],np.arange(6, data['t'].max()+1,3)),\
            }
data_dim_units = {'x':'km',\
                  'y':'km',\
                  'z':'km',\
                  't':'hr',\
                 }

tr01            = data['tr01_xt']
tr02            = data['tr02_xt']
tr03            = data['tr03_xt']
cNO              = data['no']
no2             = data['no2']
no3             = data['no3']
o3              = data['o3']

dplotter = dataPlotters(exp, figpath, data_dims, data_dim_units, tick_dims)

var        = tr03.copy()
vname      = 'tr03'
draw_data  = np.where(var>0, var, np.nan)/np.max(var)
fig, ax = dplotter.draw_xt(data = draw_data,\
                  levels = np.arange(0, 0.6, 0.1), \
                  extend = 'max', \
                  x_axis_dim = 'x',\
                  title_left  = f'{vname} [norm]', \
                  title_right = f'{reg} / @{lev_str} / y-mean', \
                  xlim = None, \
                  ylim = None,\
                  figname     = 'hov_tr01.png',\
           )

plt.show()


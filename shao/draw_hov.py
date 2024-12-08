import numpy as np
import sys, os
sys.path.insert(1,'../')
from utils.vvmtools import newVVMtools
from utils.plottools import dataPlotters
import config
import matplotlib.pyplot as plt

exp = 'pbl_op_8dth_6tr'
path = f'{config.vvmPath}/{exp}/'
figpath = f'./fig/'
reg  = 'all'
lev_str = '40m'
lev_name = 'nearSuf.'
nt = 721

data = np.load(f'{config.datPath}/xt_hov_{exp}_{reg}_{lev_str}.npz')
data_dims = {'x':data['x'],\
             'y':data['y'],\
             'z':data['z'],\
             't':np.arange(nt)*np.timedelta64(2,'m')+np.datetime64('2024-01-01 05:00:00'),\
            }
tick_dims = {'x':np.arange(data['x'][0], data['x'][-1]+0.0001, 6.4),\
             'y':np.arange(data['x'][0], data['y'][-1]+0.0001, 6.4),\
             'z':np.arange(0, data['z'].max()+0.00001,0.2),\
             't':[np.datetime64(f'2024-01-01 00:00')+np.timedelta64(i,'h') for i in [5,6,12,18,24]],\
            }
data_dim_units = {'x':'km',\
                  'y':'km',\
                  'z':'km',\
                  't':'LT',\
                 }

# var example : tr01_xt, no_xt, no2_xt ...
dplotter = dataPlotters(exp, figpath, data_dims, data_dim_units)

vname      = 'NOx'
var        = data[f'no_xt'] + data['no2_xt']
draw_data  = var.copy()
fig, ax, cax = dplotter.draw_xt(data = draw_data,\
                  levels = np.arange(0, 30.001,2),
                  extend = 'max', \
                  cmap_name   = 'OrRd', \
                  x_axis_dim = 'x',\
                  title_left  = f'{vname} (ppb)', \
                  title_right = f'{reg} / @{lev_name} / y-mean', \
                  figname     = f'hov_{vname}.png',\
           )

plt.show()

vname      = 'NOx_anomaly'
var        = data[f'no_xt'] + data['no2_xt']
var        = var - np.mean(var, axis=1, keepdims=True)
var        = np.where(var<1, np.nan, var)
draw_data  = var.copy()
fig, ax, cax = dplotter.draw_xt(data = draw_data,\
                  levels = np.arange(1,15.001,0.5),
                  extend = 'max', \
                  cmap_name   = 'RdYlGn_r', \
                  x_axis_dim = 'x',\
                  title_left  = f'{vname} (ppb)', \
                  title_right = f'{reg} / @{lev_name} / y-mean', \
                  figname     = f'hov_{vname}.png',\
           )
plt.show()

vname      = 'u'
var        = data[f'u_xt']
draw_data  = var.copy()
fig, ax, cax = dplotter.draw_xt(data = draw_data,\
                  levels = np.arange(-3, 3.001,0.25),
                  extend = 'both', \
                  cmap_name   = 'RdBu_r', \
                  x_axis_dim = 'x',\
                  title_left  = f'{vname} (m/s)', \
                  title_right = f'{reg} / @{lev_name} / y-mean', \
                  figname     = f'hov_{vname}.png',\
           )
plt.show()


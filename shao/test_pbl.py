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
data_dims = {'x':np.arange(128)*0.2,\
             'y':np.arange(128)*0.2,\
             'z':np.arange(50)*0.04,\
             't':np.arange(721)*np.timedelta64(2,'m')+np.datetime64('2024-01-01 05:00:00'),\
            }
data_dim_units = {'x':'km',\
                  'y':'km',\
                  'z':'km',\
                  't':'LST',\
                 }
pblh_th0p5_1d   = data['pblh_th0p5_1d']
pblh_maxgrad_1d = data['pblh_maxgrad_1d']
pblh_ens_1d     = data['pblh_ens']
pblh_tke_1d     = data['pblh_tke']
tr01            = data['tr01_tz']

#dplot = dataPlotters(exp, figpath, data_dims, data_dim_units, tick_dims)
dplot = dataPlotters(exp, figpath, data_dims, data_dim_units)
fig, ax = dplot.draw_zt(data = tr01/np.max(tr01), \
                      levels = np.arange(0,1.1,0.1), \
                      extend = 'max', \
                      pblh_dicts={'th0p5':     pblh_th0p5_1d,\
                                  'maxgrad':   pblh_maxgrad_1d,\
                                  'ens(1e-5)': pblh_ens_1d,\
                                  'tke(0.1)':  pblh_tke_1d,\
                                 },\
                      title_left  = 'tr01 [normalize by maximum]', \
                      title_right = f'{reg}', \
                      figname     = 'tr01.png',\
               )
plt.show()


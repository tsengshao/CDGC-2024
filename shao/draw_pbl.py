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
nt = 721

data = np.load(f'{config.datPath}/tz_series_{exp}_{reg}.npz')
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
tke             = data['tke_tz']
enstrophy       = data['enstrophy_tz']
tr01            = data['tr01_tz']
tr02            = data['tr02_tz']

pblh_th0p5_1d   = data['pblh_th0p5_1d']
pblh_maxgrad_1d = data['pblh_maxgrad_1d']
pblh_ens_1d     = data['pblh_ens']
pblh_tke_1d     = data['pblh_tke']
pblh_wth_p2n    = data['pblh_wth_p2n']
pblh_wth_min    = data['pblh_wth_min']
pblh_wth_n2p    = data['pblh_wth_n2p']


#dplot = dataPlotters(exp, figpath, data_dims, data_dim_units, tick_dims)
dplot = dataPlotters(exp, figpath, data_dims, data_dim_units)
draw_data = tr02 / np.max(tr02)

hei_dicts = {'tr01':'@nearSurf./ocean',\
             'tr02':'@750m/ocean',\
             'tr03':'@1500m/ocean',\
             'tr04':'@nearSurf./pasture',\
             'tr05':'@750m/pasture',\
             'tr06':'@1500m/pasture',\
            }

def draw(varn):
  varn_data = f'{varn}_tz'
  tr_loc    = hei_dicts[varn]
  draw_data = data[varn_data]/data[varn_data].max()
  fig, ax, cax = dplot.draw_zt(data = draw_data, \
                        pblh_dicts={'th0p5':     pblh_th0p5_1d,\
                                    'maxgrad':   pblh_maxgrad_1d,\
                                    'ens(1e-5)': pblh_ens_1d,\
                                    'tke(0.08)':  pblh_tke_1d,\
                                    'wth_p2n':  pblh_wth_p2n,\
                                    'wth_min':  pblh_wth_min,\
                                    'wth_n2p':  pblh_wth_n2p,\
                                   },\
                        title_left  = f'tracer {tr_loc}', \
                        title_right = f'{reg} / normalized by max', \

                        cmap_name   = 'Greys',\
                        levels = np.arange(0,1.1,0.1), \
                        extend = 'neither', \
                        figname     = f'',\
                 )
  #if tr_loc.split('/')[0] != '@nearSurf.':
  if tr_loc.split('/')[0] != 'rSurf.':
    ax.get_legend().remove()
  plt.savefig(f'{dplot.FIGPATH}/{varn}.png', dpi=200)
  plt.close('all')
  #plt.show(block=False)

for i in range(1,7):
  draw(f'tr{i:02d}')


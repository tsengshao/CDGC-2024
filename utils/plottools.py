import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import os, sys


class dataPlotters:
    def __init__(self, exp, figpath, dims, ticks):
        self.exp = exp
        self.figpath = figpath
        self.x       = dims['x']
        self.y       = dims['y']
        self.z       = dims['z']
        self.t       = dims['t']
        self.x_ticks  = ticks['x']
        self.y_ticks  = ticks['y']
        self.z_ticks  = ticks['z']
        self.t_ticks  = ticks['t']
        self._check_create_figpath()

    def _check_create_figpath(self):
        if not os.path.isdir(self.figpath):
            print(f'create fig folder ... {self.figpath}')
            os.system(f'mkdir -p {self.figpath}')

    def _default_setting(self):
        plt.rcParams.update({'font.size':17,
                             'axes.linewidth':2,
                             'lines.linewidth':2})

    def _create_figure(self, figsize):
        self._default_setting()
        fig     = plt.figure(figsize=figsize)
        ax      = fig.add_axes([0.1,0.1,0.8,0.8])
        cax     = fig.add_axes([0.92,0.1,0.02,0.8])
        return fig, ax, cax

    def _get_cmap(self, cmap_name='jet'):
        if cmap_name=='':
            pass
        else:
           cmap = mpl.colormaps[cmap_name]
        return cmap_name

    def draw_zt(self, data, \
                      levels, \
                      extend, \
                      pblh_dicts={},\
                      title_left = '', \
                      title_right = '', \
                      xlim=None,\
                      ylim=None,\
                      savefig=True,\
                      figname=None,\
               ):
        if type(xlim) == type(None): 
            xlim = (self.t.min(), self.t.max())
        if type(ylim) == type(None):
            ylim = (self.z.min(), self.z.max())
        if type(figname) == type(None) and savfig:
            figname = 'test.png'
        fig, ax, cax = self._create_figure(figsize=(10,6))
        plt.sca(ax)
        cmap = self._get_cmap('Reds')
        norm = mpl.colors.BoundaryNorm(boundaries=levels, \
                  ncolors=256, extend=extend)
        PO = plt.pcolormesh(self.t, self.z, data, \
                       cmap=cmap, norm=norm, \
                      )
        plt.colorbar(PO, cax=cax)
        if (len(pblh_dicts) > 0):
            for key, value in pblh_dicts.items():
                plt.plot(self.t, value, label=key, zorder=10)
            plt.legend()
        plt.xticks(self.t_ticks)
        plt.yticks(self.z_ticks)
        plt.xlim(xlim)
        plt.ylim(ylim)
        plt.grid()
        plt.title(f'{title_right}\n{self.exp}', loc='right', fontsize=15)
        plt.title(f'{title_left}', loc='left', fontsize=20, fontweight='bold')
        if savefig: plt.savefig(f'{self.figpath}/{figname}', dpi=200)
        return fig, ax


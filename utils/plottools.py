import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import os, sys
import logging

class dataPlotters:
    def __init__(self, exp, figpath, domain, units, ticks=None, time_fmt='%H'):
        self.EXP       = exp
        self.FIGPATH   = figpath
        self.DOMAIN      = domain
        self.DOMAIN_UNITS = units
        self.CUSTOM_TIME_FMT  = time_fmt
        self.DOMAIN_TICKS = ticks or self._default_dim_ticks()

    def _default_dim_ticks(self):
        dim_ticks = {}
        for key, value in self.DOMAIN.items():
            #_, dim_ticks[key]  = self._determine_ticks_and_lim(ax_name=key, ax_lim=None)
            dim_ticks[key]  = self._get_clear_ticks( ax_name = key )
        return dim_ticks

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
        if figsize[0] / figsize[1] >= 1:
            ax      = fig.add_axes([0.1,0.1,0.8,0.8])
            cax     = fig.add_axes([0.92,0.1,0.02,0.8])
        else:
            ax      = fig.add_axes([0.1,  0.1, 0.75, 0.8])
            cax     = fig.add_axes([0.88, 0.1, 0.05, 0.8])
        return fig, ax, cax

    def _get_cmap(self, cmap_name='jet'):
        if cmap_name=='':
            # define custom colormap
            pass
        else:
           cmap = mpl.pyplot.get_cmap(cmap_name)
        return cmap

    def _get_clear_ticks(self, ax_name, ax_lim=None):
        # subdomain default ticks
        lim = ax_lim or (self.DOMAIN[ax_name].min(), self.DOMAIN[ax_name].max())
        nticks = 11
        if ax_name=='t':
            #align with hourly location
            length=(lim[1] - lim[0])

            if length  // np.timedelta64(1,'D') > 1:
              self.TIME_FMT = '%D'
              delta = np.timedelta64(1,'D') 
              left = (lim[0]-np.timedelta64(1,'s')).astype('datetime64[D]')
            elif length  // np.timedelta64(1,'h') > 1:
              self.TIME_FMT = '%H'
              delta = np.timedelta64(1,'h') 
              left = (lim[0]-np.timedelta64(1,'s')).astype('datetime64[h]')
            else :
              self.TIME_FMT = '%H:%M'
              delta = np.timedelta64(10,'m')
              mn = int(lim[0].astype(str)[11:19].split(':')[-1])
              left = (lim[0] - np.timedelta64(mn%10,'m'))
            
            ticks=np.arange(left,left+length+delta*2, delta)
        elif ax_name=='z':
            length = (lim[1]-lim[0])
            interval = length / (nticks - 1)
            if interval >= 1e-3:
              interval = np.round(interval,2)
            ticks = np.arange(lim[0],lim[1]+interval,interval)
        else:
            ticks = np.linspace(lim[0], lim[1], nticks)
        return ticks
 

    def _determine_ticks_and_lim(self, ax_name, ax_lim):
        if type(ax_lim) == type(None):
            # use the ticks and limit in class setting
            self.TIME_FMT = self.CUSTOM_TIME_FMT
            lim   = (self.DOMAIN[ax_name].min(), self.DOMAIN[ax_name].max())
            ticks = self.DOMAIN_TICKS[ax_name]
        else:
            lim   = ax_lim
            ticks = self._get_clear_ticks(ax_name, ax_lim)

        return  lim, ticks

    def draw_xt(self, data, \
                      levels, \
                      extend, \
                      x_axis_dim = 'x',\
                      title_left = '', \
                      title_right = '', \
                      xlim = None, \
                      ylim = None,\
                      figname='',\
               ):
        xlim, xticks = self._determine_ticks_and_lim(ax_name=x_axis_dim, ax_lim=xlim)
        ylim, yticks = self._determine_ticks_and_lim(ax_name='t', ax_lim=ylim)

        fig, ax, cax = self._create_figure(figsize=(7,10))
        plt.sca(ax)
        cmap = self._get_cmap('Blues')
        norm = mpl.colors.BoundaryNorm(boundaries=levels, \
                  ncolors=256, extend=extend)
        PO = plt.pcolormesh(self.DOMAIN[x_axis_dim], self.DOMAIN['t'], data, \
                       cmap=cmap, norm=norm, \
                      )
        plt.colorbar(PO, cax=cax)
        plt.xticks(xticks)
        plt.yticks(yticks)
        ax.yaxis.set_major_formatter(mpl.dates.DateFormatter(self.TIME_FMT))
        plt.xlim(xlim)
        plt.ylim(ylim)
        plt.ylabel(f'time [{self.DOMAIN_UNITS["t"]}]')
        plt.xlabel(f'{x_axis_dim} [{self.DOMAIN_UNITS[x_axis_dim]}]')
        plt.grid()
        plt.title(f'{title_right}\n{self.EXP}', loc='right', fontsize=15)
        plt.title(f'{title_left}', loc='left', fontsize=20, fontweight='bold')
        if len(figname)>0:
            plt.savefig(f'{self.FIGPATH}/{figname}', dpi=200)
        return fig, ax

    def draw_zt(self, data, \
                      levels, \
                      extend, \
                      pblh_dicts={},\
                      title_left = '', \
                      title_right = '', \
                      xlim = None, \
                      ylim = None,\
                      figname='',\
               ):
        xlim, xticks = self._determine_ticks_and_lim(ax_name='t', ax_lim=xlim)
        ylim, yticks = self._determine_ticks_and_lim(ax_name='z', ax_lim=ylim)

        fig, ax, cax = self._create_figure(figsize=(10,6))
        plt.sca(ax)
        cmap = self._get_cmap('Reds')
        norm = mpl.colors.BoundaryNorm(boundaries=levels, \
                  ncolors=256, extend=extend)
        PO = plt.pcolormesh(self.DOMAIN['t'], self.DOMAIN['z'], data, \
                       cmap=cmap, norm=norm, \
                      )
        plt.colorbar(PO, cax=cax)
        if (len(pblh_dicts) > 0):
            for key, value in pblh_dicts.items():
                plt.plot(self.DOMAIN['t'], value, label=key, zorder=10)
            plt.legend()
        plt.xticks(xticks)
        ax.xaxis.set_major_formatter(mpl.dates.DateFormatter(self.TIME_FMT))
        plt.yticks(yticks)
        plt.xlim(xlim)
        plt.ylim(ylim)
        plt.xlabel(f'time [{self.DOMAIN_UNITS["t"]}]')
        plt.ylabel(f'z [{self.DOMAIN_UNITS["z"]}]')
        plt.grid()
        plt.title(f'{title_right}\n{self.EXP}', loc='right', fontsize=15)
        plt.title(f'{title_left}', loc='left', fontsize=20, fontweight='bold')
        if len(figname)>0:
            plt.savefig(f'{self.FIGPATH}/{figname}', dpi=200)
        return fig, ax


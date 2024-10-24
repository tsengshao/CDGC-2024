import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import os, sys
import logging

class dataPlotters:
    def __init__(self, exp, figpath, dims, units, ticks=None):
        self.EXP       = exp
        self.FIGPATH   = figpath
        self.DIMS      = dims
        self.DIM_UNITS = units
        self.DIM_TICKS = ticks or self._default_dim_ticks()
        if 't' in self.DIMS.keys():
            self.update_time_mapper()

    def update_time_mapper(self):
        self.TIME_MAPPER={'hour':[mpl.dates.HourLocator(interval=self.DIM_TICKS['t']),'%H'],\
                          'minute':[mpl.dates.MinuteLocator(interval=self.DIM_TICKS['t']),'%H\n%M'],\
                          }

    def _default_dim_ticks(self, nticks=11):
        dim_ticks = {}
        for key, value in self.dims.items():
            if key=='t':
                if self.DIM_UNITS['t']=='minute':
                    dim_ticks[key]=10 #set default time tick to 10 minute
                else:
                    dim_ticks[key]=1  #set default time tick to 1 unit(hour/day/month...)
            else:
                dim_ticks[key] = np.linspace(self.dims[key].min(), \
                                         self.dims[key].max(), \
                                         nticks,\
                                        )
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

    def _determine_ticks_and_lim(self, ax_name, ax_lim):
        if ax_name=='t':
            if type(ax_lim) == type(None):
                return (self.DIMS[ax_name][0],self.DIMS[ax_name][-1]), None
            else:
                return ax_lim, None
        else:
            if type(ax_lim) == type(None):
                # use the ticks and limit in class setting
                lim   = (self.DIMS[ax_name].min(), self.DIMS[ax_name].max())
                ticks = self.DIM_TICKS[ax_name]
            else:
                # subdomain default ticks
                nticks = 11
                lim = ax_lim
                ticks  = np.linspace(lim[0], lim[-1], nticks)

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
        PO = plt.pcolormesh(self.DIMS[x_axis_dim], self.DIMS['t'], data, \
                       cmap=cmap, norm=norm, \
                      )
        plt.colorbar(PO, cax=cax)
        plt.xticks(xticks)
        plt.yticks(yticks)
        plt.xlim(xlim)
        plt.ylim(ylim)
        plt.ylabel(f'time [{self.DIM_UNITS["t"]}]')
        plt.xlabel(f'{x_axis_dim} [{self.DIM_UNITS[x_axis_dim]}]')
        plt.grid()
        plt.title(f'{title_right}\n{self.EXP}', loc='right', fontsize=15)
        plt.title(f'{title_left}', loc='left', fontsize=20, fontweight='bold')
        if len(figname)==0:
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
        PO = plt.pcolormesh(self.DIMS['t'], self.DIMS['z'], data, \
                       cmap=cmap, norm=norm, \
                      )
        plt.colorbar(PO, cax=cax)
        if (len(pblh_dicts) > 0):
            for key, value in pblh_dicts.items():
                plt.plot(self.DIMS['t'], value, label=key, zorder=10)
            plt.legend()
        #ignore xticks call and set x-axis as a time axis
        #plt.xticks(xticks)
        ax.xaxis.set_major_locator(self.TIME_MAPPER[self.DIM_UNITS['t']][0])
        ax.xaxis.set_major_formatter(mpl.dates.DateFormatter(self.TIME_MAPPER[self.DIM_UNITS['t']][1]))
        plt.yticks(yticks)
        plt.xlim(xlim)
        plt.ylim(ylim)
        plt.xlabel(f'time [{self.DIM_UNITS["t"]}]')
        plt.ylabel(f'z [{self.DIM_UNITS["z"]}]')
        plt.grid()
        plt.title(f'{title_right}\n{self.EXP}', loc='right', fontsize=15)
        plt.title(f'{title_left}', loc='left', fontsize=20, fontweight='bold')
        if len(figname)!=0:
            plt.savefig(f'{self.FIGPATH}/{figname}', dpi=200)
        return fig, ax


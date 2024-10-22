import numpy as np
import matplotlib.pyplot as plt
import sys, os
sys.path.insert(1,'../')
from VVMTools.vvmtools import VVMTools
import multiprocessing

class newVVMtools(VVMTools):
    def __init__(self, cpath):
        super().__init__(cpath)
        self.nx = len(self.DIM['xc'])
        self.ny = len(self.DIM['yc'])
        self.nz = len(self.DIM['zc'])
        self.nt = len(self.time_array_str)

    def get_domain_range(self, region):
        # domain_range=(None, None, None, None, None, None), # (k1, k2, j1, j2, i1, i2)
        if   region == 'left':
             domain_range = (0,self.nz,0,self.ny,0,self.nx//2)
        elif region == 'right':
             domain_range = (0,self.nz,0,self.ny,self.nx//2+1,self.nx)
        elif region == 'all':
             domain_range = (0,self.nz,0,self.ny,0,self.nx)
        return domain_range

    def cal_TKE(self, t, func_config):
        u = np.squeeze(self.get_var('u', t, numpy=True, domain_range=func_config["domain_range"]) )
        v = np.squeeze(self.get_var('v', t, numpy=True, domain_range=func_config["domain_range"]) )
        w = np.squeeze(self.get_var('w', t, numpy=True, domain_range=func_config["domain_range"]) )
        u_inter = (u[:, :, 1:] + u[:, :, :-1])[1:, 1:] / 2
        v_inter = (v[:, 1:] + v[:, :-1])[1:, :, 1:] / 2
        w_inter = (w[1:] + w[:-1])[:, 1:, 1:] / 2
        tke = np.nanmean(u_inter**2 + v_inter**2 + w_inter**2, axis=(1,2))
        return tke

    def cal_enstrophy(self, t, func_config):
        eta  = np.squeeze(self.get_var('eta_2', t, numpy=True, \
                                       domain_range=func_config["domain_range"]),\
                                      )
        zeta = np.squeeze(self.get_var('zeta',  t, numpy=True, \
                                       domain_range=func_config["domain_range"]),\
                                      )
        xi   = np.squeeze(self.get_var('xi',    t, numpy=True, \
                                       domain_range=func_config["domain_range"]),\
                                      )
        enstrophy = np.nanmean(eta**2 + zeta**2 + xi**2, axis=(1,2))
        return enstrophy

    def cal_wpthpbar(self, t, func_config):
        w  = np.squeeze(self.get_var('w',   t, numpy=True,\
                                       domain_range=func_config["domain_range"]),\
                                      )
        th = np.squeeze(self.get_var('th',  t, numpy=True,\
                                       domain_range=func_config["domain_range"]),\
                                      )
        w[1:, :, :] = (w[1:, :, :] + w[:-1, :, :]) / 2
        w[0, :, :] = 0.0
        w_bar    = np.nanmean(w,  axis=(1,2), keep_dims=True)
        th_bar   = np.nanmean(th, axis=(1,2), keep_dims=True)
        output = np.nanmean((w - w_bar) * (th - th_bar), axis=(1,2))
        return output

    def cal_pblh_0p5(self,t, func_config):
        th = np.squeeze(self.get_var("th",t,\
                domain_range=func_config["domain_range"], \
                numpy=True, \
                compute_mean=True, \
                axis=(1,2),\
               ))
        zc     = self.DIM["zc"]
        idxz   = np.argmin( np.abs( th-(th[1]+0.5) ) )
        return zc[idxz]

    def cal_pblh_maxgrad(self,t, func_config):
        th = np.squeeze(self.get_var("th",t,\
                domain_range=func_config["domain_range"], \
                numpy=True, \
                compute_mean=True, \
                axis=(1,2),\
               ))
        del_th  = np.gradient(th, self.DIM['zc'], axis=0)
        max_idx = np.nanargmax(del_th,axis=0)
        return self.DIM['zc'][max_idx]


if __name__=='__main__':
    exp = 'op_1'
    path = f'../../vvm/{exp}/'
    newVVMTol = newVVMtools(path)
    nt = 10

    # example
    reg       = 'all' #'left', 'right'
    drange = newVVMTol.get_domain_range(reg)
    tke = newVVMTol.func_time_parallel(\
              func       = newVVMTol.cal_TKE, \
              time_steps = list(range(nt)), \
              func_config = {'domain_range':drange},\
              cores = 5,\
             )

## pblh_th0p5 = newVVMTol.func_time_parallel(\
##               func       = newVVMTol.cal_pblh_0p5, \
##               time_steps = list(range(nt)), \
##               drange     = newVVMTol.get_domain_range('all'), \
##               cores = 5,\
##              )
## 
## pblh_th0p5 = newVVMTol.func_time_parallel(\
##               func       = newVVMTol.cal_pblh_0p5, \
##               time_steps = list(range(nt)), \
##               drange     = newVVMTol.get_domain_range('all'), \
##               cores = 5,\
##              )
## 
## wth_bar = newVVMTol.func_time_parallel(\
##           func       = newVVMTol.cal_wpthpbar, \
##           time_steps = list(range(nt)), \
##           drange     = newVVMTol.get_domain_range(reg), \
##           cores = 5,\
##          )



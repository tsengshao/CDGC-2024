import numpy as np
import pandas as pd
import os, sys
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
import pickle
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision.transforms import transforms
from torchinfo import summary
from torch.utils.data import Dataset, DataLoader, TensorDataset, random_split
import config
from ml_model import CNN1D
from sklearn.preprocessing import normalize
import sys, os
import matplotlib as mpl
from datetime import datetime, timedelta

def fig_default_setting():
    plt.close('all')
    plt.rcParams.update({'font.size':17,
                         'axes.linewidth':2,
                         'lines.linewidth':5})


def create_input_from_profile(initial_profile):
    norm  = normalize(initial_profile.reshape(1,-1), norm='max', axis=1)
    torch_x = torch.tensor(norm).to(device, dtype=torch.float)
    torch_x = torch_x.unsqueeze(0)
    return torch_x


def integrate(initial_profile, model, device, nt):
    #a = torch.tensor(initial_profile).to(device, dtype=torch.float)
    output = [initial_profile]
    for it in range(nt):
        torch_x = create_input_from_profile(output[-1])
        pred=vcnn(torch_x)
        output.append(output[-1]+pred[0,:].detach().cpu().numpy())
    return np.array(output)

if __name__=='__main__':
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device='cpu'
    print(device)

    mlname   = 'VVM-1DCNN_evenloss'
    figpath  = f'./fig/{mlname}/'
    os.system(f'mkdir -p {figpath}')
    casen    = 'pbl_op_8dth_6tr'
    #casen    = 'pbl_op_11dth'
    fname = config.datPath+f'/tz_series_{casen}_all.npz'
    data  = np.load(fname)
    th0   = data['th_tz'].T
    z     = data['z']
    time  = data['t']
    
    vcnn=torch.load(config.datPath+f'/{mlname}.pkl').to(device)
    vcnn.eval()

    # hindcast
    bins=np.arange(-1,1.0001,0.02)
    error_map = np.zeros((z.size, bins.size-1))

    itt=30 # integrate time
    fig0, ax0 = plt.subplots(figsize=(13,10))
    iz0 = np.argmin(np.abs(z-0.25))
    ax0.plot(time, th0[:,iz0], c='k')
    for i0 in np.arange(0, 720-itt+1, 1): # hindcast start time
        print(i0)
        result = integrate(th0[i0], vcnn, device, nt=itt)

        # timeseries
        if i0 % 5 == 0:
            ax0.plot(time[i0:i0+itt+1], result[:,iz0], c='C0', lw=1)

        # bias cfad
        bias = result[itt] - th0[i0+itt]
        if bias.min() < bins.min():
            print(bias.min(), bins.min())
            sys.exit('please extend the lower bound of bins')
        ibin = ( (bias - bins[0]) / np.diff(bins)[0] ).astype(int)
        iz   = np.arange(z.size, dtype=int)
        error_map[iz, ibin] += 1

        if i0%30 == 0:
            fig_default_setting()
            fig, ax = plt.subplots(figsize=(8,10))
            plt.plot(th0[i0], z, label=f'initial_{i0/30+5}', c='k')
            plt.plot(th0[i0+itt], z, label=f'+{itt/30}hr true', c='0.5')
            plt.plot(result[itt], z, label=f'+{itt/30}hr pred', c='C0', ls='--')
            plt.legend()
            plt.xlim(292, 300)
            plt.ylim(0,0.8)
            plt.ylabel('[km]')
            plt.xlabel('[K]')
            plt.grid(True)
            plt.title(f'Hindcast (+{itt/30}hr)\n'+r'$/Theta$'+' profile', loc='left', fontweight='bold', fontsize=15)
            plt.title(f'ini: {i0/30+5} LT \n{casen} / {mlname}', fontsize=15, loc='right')
            plt.tight_layout()
            outpath=f'{figpath}/hprofile_{itt/30:.0f}hr'
            os.system(f'mkdir -p {outpath}')
            plt.savefig(f'{outpath}/{i0:06d}.png',dpi=200)

    plt.sca(ax0)
    plt.xlim(time[0], time[-1])
    plt.grid(True)
    plt.xlabel('time [hr]')
    plt.ylabel('Theta [K]')
    plt.title(f'Hindcast (+{itt/30}hr) Timeseries @{z[iz0]}km', weight='bold', loc='left', fontsize=20)
    plt.title(f'{casen}\n{mlname}', loc='right', fontsize=20)
    plt.savefig(f'{figpath}/hseries_{z[iz0]:.2f}.png',dpi=200)


    x = ( bins[:-1] + bins[1:] ) /2
    fig, ax = plt.subplots(figsize=(8,10))
    levels = np.arange(0,0.500001,0.02)
    norm = mpl.colors.BoundaryNorm(boundaries=levels, \
              ncolors=256, extend='max')
    ddata = np.where(error_map<=0, np.nan, error_map) / np.sum(error_map,axis=1, keepdims=True)
    PC = plt.pcolormesh(x, z, ddata, norm=norm, cmap=plt.cm.jet)
    CB = plt.colorbar(PC)
    CB.ax.set_yticks(levels[::2])
    plt.xlabel('[K]')
    plt.ylabel('[km]')
    plt.xlim(-0.5, 0.5)
    plt.ylim(0,2)
    plt.title(f'Theta bias\nHindcast +{itt/30:.0f}hr compare to VVM', loc='left', fontweight='bold', fontsize=12)
    sample = np.sum(error_map,axis=1)[0]
    plt.title(f'{casen} / {mlname}\n#{sample:.0f} profiles', loc='right', fontsize=12)
    plt.tight_layout()
    plt.savefig(f'{figpath}/hbias_{itt/30:.0f}hr.png',dpi=200)


    # integrate 24 hr
    nt = 720
    result = integrate(th0[0], vcnn, device, nt=720)
    outpath = f'{figpath}/integrate24h/'
    os.system(f'mkdir -p {outpath}')
    for it in range(0,721,30):
        print(it)
        nowdate = datetime(2024,1,1,5) + it*timedelta(minutes=2)
        fig_default_setting()
        fig, ax = plt.subplots(figsize=(12,10))
        plt.plot(th0[it], z, label='vvm', c='k')
        plt.plot(result[it], z, label='ml', c='C0')
        plt.title(f'integrated Theta profile\nforcast: {it*2/60:.1f}LT / initial: 05LT', loc='left', fontsize=20, fontweight='bold')
        plt.title(f'{casen}\n{mlname}', loc='right', fontsize=20)
        plt.grid(True)
        plt.legend()
        plt.xlim(290, 310)
        plt.ylim(0,2)
        plt.ylabel('[km]')
        plt.xlabel('[K]')
        plt.savefig(f'{outpath}/{it:06d}.png', dpi=200)



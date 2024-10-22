import numpy as np
import sys, os
sys.path.insert(1,'../')
from utils.vvmtools import newVVMtools
import config



exp = 'op_1'
path = f'{config.vvmPath}/{exp}/'
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




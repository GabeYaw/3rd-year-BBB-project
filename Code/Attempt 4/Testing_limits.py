from itertools import product
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import scipy.stats
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as utils
from dmipy.core.acquisition_scheme import acquisition_scheme_from_bvalues
from dmipy.core.modeling_framework import MultiCompartmentSphericalMeanModel
from dmipy.signal_models import cylinder_models, gaussian_models, sphere_models
from scipy.io import savemat
from scipy.special import erf
from tqdm import tqdm

def sim_sig_np(bf,be,tm,adc,sigma,axr):
    be = np.expand_dims(be, axis=0)
    bf = np.expand_dims(bf, axis=0)
    tm = np.expand_dims(tm, axis=0)

    if adc.size != 1:
        adc = np.expand_dims(adc, axis=1)
        sigma = np.expand_dims(sigma, axis=1)
        axr = np.expand_dims(axr, axis=1)                                   

    tm[(tm == np.min(tm)) & (bf == 0)] = np.inf

    adc_prime = adc * (1 - sigma* np.exp(-tm*axr))
    normalised_signal = np.exp(-adc_prime * be)

    return normalised_signal, adc_prime

nvox = 10 # number of voxels to simulate

bf = np.array([0, 0, 250, 250, 250, 250, 250, 250]) * 1e-3   # filter b-values [ms/um2]
be = np.array([0, 250, 0, 250, 0, 250, 0, 250]) * 1e-3       # encoding b-values [ms/um2]
tm = np.array([20, 20, 20, 20, 200, 200, 400, 400], dtype=np.float32) * 1e-3 # mixing time [s]

adc_lb = 0.1        #[um2/ms]
adc_ub = 3.5        #[um2/ms]
sig_lb = 0          #[a.u.]
sig_ub = 1          #[a.u.]
axr_lb = 0.01        #[s-1]
axr_ub = 0.1         #[s-1]

sim_adc = np.random.uniform(adc_lb,adc_ub,nvox)                 # ADC, simulated [um2/ms]
sim_sigma = np.random.uniform(sig_lb,sig_ub,nvox)               # sigma, simulated [a.u.]
sim_axr = np.random.uniform(axr_lb,axr_ub,nvox)                 # AXR, simulated [s-1]

sim_E_vox, sim_adc_prime = sim_sig_np(bf,be,tm,sim_adc,sim_sigma,sim_axr)
sorted_sim_E_vox = np.sort(sim_E_vox, axis=1)
sorted_sim_adc_prime = np.sort(sim_adc_prime, axis=1)


"""adc_prime = adc * (1 - sigma* np.exp(-tm*axr))
normalised_signal = np.exp(-adc_prime * be)"""
test4 = np.exp(-np.inf*0)
# if multiplier is 0, then output is nan

print("test4:",test4)
test = 1*(1 - np.exp(-tm*1e-7))
#when the multiplier is 1e-7, the result rounds to 1.
print(test)
test1 = np.exp(-test*0)
print(test1)

tm = torch.tensor(tm, dtype=torch.float64)
test2 = 1*(1 - torch.exp(-tm*1e-8))
#when the multiplier is 1e-8, the result rounds to 1.
print(test2)


'''generate values:

Check if new value is nan

    print old value

update old value to new value'''
import time
import numpy as np
import matplotlib.pyplot as plt
import nibabel as nib
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as utils
from tqdm import tqdm
from scipy.special import erf
import scipy.stats

from dmipy.core.acquisition_scheme import acquisition_scheme_from_bvalues
from dmipy.core.modeling_framework import MultiCompartmentSphericalMeanModel
from dmipy.signal_models import sphere_models, cylinder_models, gaussian_models

from scipy.io import savemat
np.random.seed(4231314)
from Simulations import *

torch.autograd.set_detect_anomaly(True)
#torch.autograd.detect_anomaly


E_vox = sim_E_vox

start_time = time.time()

class Net(nn.Module): # this is the neural network

    def __init__(self, be, bf ,tm, nparams,limits):
        super(Net, self).__init__()

        self.be = be
        self.bf = bf
        self.tm = tm
        #self.tm[(self.tm == torch.min(self.tm)) & (self.bf == 0)] = float('inf')
        self.limits = limits

        self.layers = nn.ModuleList()
        for i in range(3): # 3 fully connected hidden layers
            self.layers.extend([nn.Linear(len(be), len(be)), nn.PReLU()])
        self.encoder = nn.Sequential(*self.layers, nn.Linear(len(be), nparams))

    def forward(self, X):
        
        if torch.isinf(X).any():
            print("X contains inf")
            print(X)
        if torch.isnan(X).any():
            print("X contains nan")
            print(X)

        params = torch.nn.functional.softplus(self.encoder(X))

        if torch.isinf(self.encoder(X)).any():
            print("encoder(X) contains inf")
            print(self.encoder(X))
        if torch.isnan(self.encoder(X)).any():
            print("encoder(X) contains nan")
            print(self.encoder(X))

        if torch.isinf(params).any():
            print("params contains inf")
            print(params)
        if torch.isnan(params).any():
            print("params contains nan")
            print(params)
        adc = torch.clamp(params[:,0].unsqueeze(1), min=self.limits[0,0], max=self.limits[0,1]) # parameter constraints
        sigma = torch.clamp(params[:,1].unsqueeze(1), min=self.limits[1,0], max=self.limits[1,1])
        axr = torch.clamp(params[:,2].unsqueeze(1), min=self.limits[2,0], max=self.limits[2,1])


        """if tm == np.inf:
            adc_prime = adc * (1 - sigma * torch.exp(-self.tm * axr))
        else:
            adc_prime = adc * (1 - sigma * torch.exp(-self.tm * axr))"""
        adc_prime = adc * (1 - sigma * torch.exp(-self.tm * axr))
            
        if torch.isinf(adc_prime).any():
            print("adc_prime contains inf")
            print(adc_prime)
        if torch.isnan(adc_prime).any():
            print("adc_prime contains nan")
            print(adc_prime)

        X = torch.exp(-adc_prime * self.be)        
        if torch.isinf(X).any():
            print("X contains inf")
            print(X)
        if torch.isnan(X).any():
            print("X contains nan")
            print(X)

        return X, adc, sigma, axr, adc_prime

# define network
nparams = 3
be = torch.FloatTensor(be)
bf = torch.FloatTensor(bf)
tm = torch.FloatTensor(tm)
limits = torch.FloatTensor(limits)

net = Net(be,bf, tm, nparams,limits)

#create batch queues for data
batch_size = 128
num_batches = len(E_vox) // batch_size
trainloader = utils.DataLoader(torch.from_numpy(E_vox.astype(np.float32)),
                                batch_size = batch_size, 
                                shuffle = True,
                                num_workers = 0, #was 2 previously
                                drop_last = True)

# loss function and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(net.parameters(), lr = 0.00001)

# best loss
best = 1e16
num_bad_epochs = 0
patience = 500

# train
for epoch in range(10000): 
    print("-----------------------------------------------------------------")
    print("epoch: {}; bad epochs: {}".format(epoch, num_bad_epochs))
    net.train()
    running_loss = 0.

    for i, X_batch in enumerate(tqdm(trainloader), 0):
        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        X_pred, adc_pred, sigma_pred, axr_pred, adc_prime_pred,  = net(X_batch)
        loss = criterion(X_pred, X_batch)
        if torch.isnan(X_pred).any():
            print("X_pred contains nan")
            print(X_pred)
        if torch.isinf(X_pred).any():
            print("X_pred contains inf")
            print(X_pred)
        if torch.isnan(X_batch).any():
            print("X_batch contains nan")
            print(X_batch)
        if torch.isinf(X_batch).any():
            print("X_batch contains inf")
            print(X_batch)

        if torch.isnan(loss).any():
            print("loss contains nan")
            print(loss)

        loss.backward()
        for name, param in net.named_parameters():
            if param.grad is not None:
                #print(f'Parameter: {name}, Gradient: {param.grad}')
                pass
            else:
                #print(f'Parameter: {name}, Gradient: None')
                pass
        optimizer.step()
        running_loss += loss.item()
      
    print("loss: {}".format(running_loss))
    # early stopping
    if running_loss < best:
        print("####################### saving good model #######################")
        final_model = net.state_dict()
        best = running_loss
        num_bad_epochs = 0
    else:
        num_bad_epochs = num_bad_epochs + 1
        if num_bad_epochs == patience:
            print("done, best loss: {}".format(best))
            break
print("done")

net.load_state_dict(final_model)

net.eval()
with torch.no_grad():
    X_final_pred, adc_final_pred, sigma_final_pred, axr_final_pred, adc_prime_final_pred = net(torch.from_numpy(E_vox.astype(np.float32)))

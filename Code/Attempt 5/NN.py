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
        #the following line has no impact on the output. Whatever the RHS is, the output is the same.
        #I left it out so that there are no infs, although it doesn't matter now
        #self.tm[(self.tm == torch.min(self.tm)) & (self.bf == 0)] = float('inf')
        self.limits = limits

        self.layers = nn.ModuleList()
        for i in range(3): # 3 fully connected hidden layers
            self.layers.extend([nn.Linear(len(be), len(be)), nn.PReLU()])
        self.encoder = nn.Sequential(*self.layers, nn.Linear(len(be), nparams))

    def forward(self, adc_prime):
    #def forward(self, X):
        #params = torch.nn.functional.softplus(self.encoder(X))
        params = torch.nn.functional.softplus(self.encoder(adc_prime))

        adc = torch.clamp(params[:,0].unsqueeze(1), min=self.limits[0,0], max=self.limits[0,1]) # parameter constraints
        sigma = torch.clamp(params[:,1].unsqueeze(1), min=self.limits[1,0], max=self.limits[1,1])
        axr = torch.clamp(params[:,2].unsqueeze(1), min=self.limits[2,0], max=self.limits[2,1])

        adc_prime_1 = adc.expand(adc_prime.shape[0], 2)

        #adc_prime_1 = adc.expand(X.shape[0], 2)
        X1 = torch.exp(-adc_prime_1 * self.be[:2])

        adc_prime_2 =  adc * (1 - sigma * torch.exp(-self.tm[-6:] * axr))
        X2 = torch.exp(-adc_prime_2 * self.be[-6:])

        X = torch.cat((X1, X2), 1)
        adc_prime = torch.cat((adc_prime_1, adc_prime_2), 1)        

        return X, adc, sigma, axr, adc_prime

# define network
nparams = 3
be = torch.FloatTensor(be)
bf = torch.FloatTensor(bf)
tm = torch.FloatTensor(tm)
limits = torch.FloatTensor(limits)

batch_size = 128
net = Net(be,bf, tm, nparams,limits)

dataset = torch.utils.data.TensorDataset(torch.from_numpy(sim_adc_prime.astype(np.float32)), 
                             torch.from_numpy(sim_axr.astype(np.float32)))
#create batch queues for data
num_batches = len(E_vox) // batch_size
#trainloader = utils.DataLoader(torch.from_numpy(E_vox.astype(np.float32)),
trainloader = utils.DataLoader(dataset,
                                batch_size = batch_size, 
                                shuffle = True,
                                num_workers = 0, #was 2 previously
                                drop_last = True)

# loss function and optimizer
learning_rate = 1e-2
#default lr: 1e-3
criterion1 = nn.MSELoss()
criterion2 = nn.MSELoss()
optimizer = optim.Adam(net.parameters(),lr=learning_rate)



# best loss
best = 1e16
num_bad_epochs = 0
patience = 100

# train
for epoch in range(10000): 
    print("-----------------------------------------------------------------")
    print("epoch: {}; bad epochs: {}".format(epoch, num_bad_epochs))
    net.train()
    running_loss = 0.

    #for i, X_batch in enumerate(tqdm(trainloader), 0):
    #for i, adc_prime_batch in enumerate(tqdm(trainloader), 0):
    for i, (adc_prime_batch, axr_batch) in enumerate(tqdm(trainloader), 0):
        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        #X_pred, adc_pred, sigma_pred, axr_pred, adc_prime_pred  = net(X_batch)
        X_pred, adc_pred, sigma_pred, axr_pred, adc_prime_pred = net(adc_prime_batch)
        loss1 = criterion1(adc_prime_pred, adc_prime_batch)
        
        axr_batch = axr_batch.unsqueeze(1)
        loss2 = criterion2(axr_pred, axr_batch)
        #loss = criterion(X_pred, X_batch)
        
        #loss = torch.max(loss1, loss2)
        loss = loss1 + loss2
        loss = loss.squeeze(0)
        loss.backward()

        optimizer.step()
        running_loss += loss.item()
      
    print(f"ADC Prime Loss: {loss1}\tAXR Loss: {loss2}")
    #this is only from the final batch

    print("loss: {}".format(running_loss))
    print("best loss: {}".format(best))
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
    #X_final_pred, adc_final_pred, sigma_final_pred, axr_final_pred, adc_prime_final_pred = net(torch.from_numpy(E_vox.astype(np.float32)))
    X_final_pred, adc_final_pred, sigma_final_pred, axr_final_pred, adc_prime_final_pred = net(torch.from_numpy(sim_adc_prime.astype(np.float32)))


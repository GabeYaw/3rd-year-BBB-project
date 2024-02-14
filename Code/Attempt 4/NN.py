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

from Simulations import *
# Creating the neural network
class Net(nn.Module): # this is the neural network
    #defining the init and foward pass functions. 

    def __init__(self,be,bf,tm,nparams,limits):
        super(Net, self).__init__()

        self.be = be
        self.bf = bf
        self.tm = tm
        self.limits = limits
        self.tm[(self.tm == torch.min(self.tm)) & (self.bf == 0)] = float('inf')

        #defining the layers that we want. 
        # 3 layers with no. of be nodes. 
        self.layers = nn.ModuleList()
        for i in range(3): # 3 fully connected hidden layers
            self.layers.extend([nn.Linear(len(be), len(be)), nn.PReLU()])
            #https://pytorch.org/docs/stable/generated/torch.nn.PReLU.html
        self.encoder = nn.Sequential(*self.layers, nn.Linear(len(be), nparams))

    def forward(self, E_vox):

        params = torch.nn.functional.softplus(self.encoder(E_vox))
        #running a forward pass through the network

        #SoftPlus is a smooth approximation to the ReLU function and can be used to constrain the output of a machine to always be positive
        #params contains batch_size x nparams outputs, so each row is adc, sigma and axr.

        #unsqueeze adds an additional dimension. 
        #parameter constraints from Elizabeth matlab 

        adc = torch.clamp(params[:, 0].unsqueeze(1), min=limits[0,0], max=limits[0,1])
        sigma = torch.clamp(params[:, 1].unsqueeze(1), min=limits[1,0], max=limits[1,1])
        axr = torch.clamp(params[:, 2].unsqueeze(1), min=limits[2,0], max=limits[2,1])
        axr_unclamped = params[:, 2].unsqueeze(1)

        

        adc_prime = adc * (1 - sigma * torch.exp(-tm * axr))
        E_vox = torch.exp(-adc_prime * be)

        """print("tm:", self.tm.shape)
        print("be:", self.be.shape)
        print("bf:", self.bf.shape)
        print("adc:", adc.shape)
        print("sigma:", sigma.shape)
        print("axr:", axr.shape)
        print("evox:", E_vox.shape)"""
        
        """print("self.encoder(E_vox)", self.encoder(E_vox)[0,:])
        print("params:", params[0,:])
        print("adc:", adc)
        print("sigma:", sigma)
        print("axr:", axr)
        print("evox:", E_vox)"""

        return E_vox, adc_prime, adc, sigma, axr, axr_unclamped


# NN continued
# define network
nparams = 3
#because of adc, sigma and axr

#converting numpy arrays to pytorch tensors. 
#be = torch.tensor(be)
#bf = torch.tensor(bf)
#tm = torch.tensor(tm)
be = torch.tensor(be, dtype=torch.float32)
bf = torch.tensor(bf, dtype=torch.float32)
tm = torch.tensor(tm, dtype=torch.float32)

batch_size = 128

#initilise network
net = Net(be, bf, tm, nparams,limits)

#create batch queues for data
#// means divide and round down. 
num_batches = len(sim_E_vox) // batch_size

#import the sim_E_vox array into the dataloader
#drop_last ignores the last batch if it is the wrong size. 
#num_workers is about performance. 

trainloader = utils.DataLoader(torch.from_numpy(sim_E_vox.astype(np.float32)),
                                batch_size = batch_size, 
                                shuffle = True,
                                num_workers = 0, #was 2 previously
                                drop_last = True)

# loss function and optimizer
#choosing which loss function to use. 
#not sure what the optmizer is
criterion = nn.MSELoss()
optimizer = optim.Adam(net.parameters(), lr = 0.0001)

# best loss
best = 1e16
num_bad_epochs = 0
#can increase patience a lot, speed not an issue.
patience = 100

# Training
# train
loss_progress = np.empty(shape=(0,))

adc_progress = np.empty(shape=(0,)) 
sigma_progress = np.empty(shape=(0,)) 
axr_progress = np.empty(shape=(0,))
signal_progress = np.empty(shape=(0,))
adc_prime_progress = np.empty(shape=(0,))

axr_unclamped_progress = np.empty(shape=(0,))

for epoch in range(10000): 
    print("-----------------------------------------------------------------")
    print("epoch: {}; bad epochs: {}".format(epoch, num_bad_epochs))
    net.train()
    running_loss = 0.

    #tqdm shows a progress bar. 
    for i, sim_E_vox_batch in enumerate(tqdm(trainloader), 0):
        
        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        pred_E_vox, pred_adc_prime, pred_adc, pred_sigma, pred_axr, axr_unclamped = net(sim_E_vox_batch)
        
        """print(sim_E_vox_batch)
        print("pred_E_vox:", pred_E_vox)
        print("pred_adc:", pred_adc)
        print("pred_sigma:", pred_sigma)
        print("pred_axr:", pred_axr)"""

        if torch.isnan(pred_E_vox).any():
            print("evox nan found in batch",i,"epoch",epoch)
        if torch.isnan(pred_adc).any():
            print("pred_adc nan found in batch",i,"epoch",epoch)
        if torch.isnan(pred_axr).any():
            print("pred_axr nan found in batch",i,"epoch",epoch)
        if torch.isnan(pred_sigma).any():
            print("pred_sigma nan found in batch",i,"epoch",epoch)
        
        loss = criterion(pred_E_vox, sim_E_vox_batch)

        #loss_prime = criterion(pred_adc_prime,sim_adc_prime)
        #loss prime needs sim_adc_prime to be batched. 
        loss.backward()
        optimizer.step()
        running_loss += loss.item()


        if i == 0:
            adc_progress = np.append(adc_progress, pred_adc[0].detach().numpy())
            sigma_progress = np.append(sigma_progress, pred_sigma[0].detach().numpy())
            axr_progress = np.append(axr_progress, pred_axr[0].detach().numpy())

            axr_unclamped_progress = np.append(axr_unclamped_progress, axr_unclamped[0].detach().numpy())

            signal_progress = np.append(signal_progress, pred_E_vox[:,0].detach().numpy())
            adc_prime_progress = np.append(adc_prime_progress, pred_adc_prime[:,0].detach().numpy())

        
    print("loss: {}".format(running_loss))
    # early stopping
    if running_loss < best:
        print("####################### saving good model #######################")
        final_model = net.state_dict()
        best = running_loss
        num_bad_epochs = 0
        loss_progress = np.append(loss_progress, best)
    else:

        num_bad_epochs = num_bad_epochs + 1
        loss_progress = np.append(loss_progress, best)
        if num_bad_epochs == patience:
            print("done, best loss: {}".format(best))
            break
print("done")

net.load_state_dict(final_model)

net.eval()
with torch.no_grad():
    final_pred_E_vox, final_pred_adc_prime, final_pred_adc_repeated, final_pred_sigma_repeated, final_pred_axr_repeated, _ = net(torch.from_numpy(sim_E_vox.astype(np.float32)))
    # adc sigma and axr will have 8 columns which are all the same

final_pred_adc = final_pred_adc_repeated[:, 0]
final_pred_sigma = final_pred_sigma_repeated [:, 0]
final_pred_axr = final_pred_axr_repeated[:, 0]
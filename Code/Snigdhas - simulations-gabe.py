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
start_time = time.time()
# simulate data according to verdict model - there is a function for each of the three compartments
# when you do it for bbb-fexi, this will change

def sphere(r):

    SPHERE_TRASCENDENTAL_ROOTS = np.r_[
        # 0.,
        2.081575978, 5.940369990, 9.205840145,
        12.40444502, 15.57923641, 18.74264558, 21.89969648,
        25.05282528, 28.20336100, 31.35209173, 34.49951492,
        37.64596032, 40.79165523, 43.93676147, 47.08139741,
        50.22565165, 53.36959180, 56.51327045, 59.65672900,
        62.80000055, 65.94311190, 69.08608495, 72.22893775,
        75.37168540, 78.51434055, 81.65691380, 84.79941440,
        87.94185005, 91.08422750, 94.22655255, 97.36883035,
        100.5110653, 103.6532613, 106.7954217, 109.9375497,
        113.0796480, 116.2217188, 119.3637645, 122.5057870,
        125.6477880, 128.7897690, 131.9317315, 135.0736768,
        138.2156061, 141.3575204, 144.4994207, 147.6413080,
        150.7831829, 153.9250463, 157.0668989, 160.2087413,
        163.3505741, 166.4923978, 169.6342129, 172.7760200,
        175.9178194, 179.0596116, 182.2013968, 185.3431756,
        188.4849481, 191.6267147, 194.7684757, 197.9102314,
        201.0519820, 204.1937277, 207.3354688, 210.4772054,
        213.6189378, 216.7606662, 219.9023907, 223.0441114,
        226.1858287, 229.3275425, 232.4692530, 235.6109603,
        238.7526647, 241.8943662, 245.0360648, 248.1777608,
        251.3194542, 254.4611451, 257.6028336, 260.7445198,
        263.8862038, 267.0278856, 270.1695654, 273.3112431,
        276.4529189, 279.5945929, 282.7362650, 285.8779354,
        289.0196041, 292.1612712, 295.3029367, 298.4446006,
        301.5862631, 304.7279241, 307.8695837, 311.0112420,
        314.1528990
    ]

    D = 2
    gamma = 2.67e2
    radius = r

    b_values = np.array([1e-6, 0.090, 1e-6, 0.500, 1e-6, 1.5, 1e-6, 2, 1e-6, 3])
    Delta = np.array([23.8, 23.8, 23.8, 31.3, 23.8, 43.8, 23.8, 34.3, 23.8, 38.8])
    delta = np.array([3.9, 3.9, 3.9, 11.4, 3.9, 23.9, 3.9, 14.4, 3.9, 18.9])

    gradient_strength = np.array([np.sqrt(b_values[i])/(gamma*delta[i]*np.sqrt(Delta[i]-delta[i]/3)) for i,_ in enumerate(b_values)])

    alpha = SPHERE_TRASCENDENTAL_ROOTS / radius
    alpha2 = alpha ** 2
    alpha2D = alpha2 * D

    first_factor = -2 * (gamma * gradient_strength) ** 2 / D
    
    summands = np.zeros((len(SPHERE_TRASCENDENTAL_ROOTS),len(b_values)))
    for i,_ in enumerate(delta):
        summands[:,i] = (
            alpha ** (-4) / (alpha2 * radius ** 2 - 2) *
            (
                2 * delta[i] - (
                    2 +
                    np.exp(-alpha2D * (Delta[i] - delta[i])) -
                    2 * np.exp(-alpha2D * delta[i]) -
                    2 * np.exp(-alpha2D * Delta[i]) +
                    np.exp(-alpha2D * (Delta[i] + delta[i]))
                ) / (alpha2D)
            )
        )
    
    E = np.exp(
        first_factor *
        summands.sum()
    )

    return E

def ball(d):

    bvals = np.array([1e-6, 0.090, 1e-6, 0.500, 1e-6, 1.5, 1e-6, 2, 1e-6, 3])
    E_ball = np.exp(-bvals * d)
    
    return E_ball

def astrosticks(l):

    bvals = np.array([1e-6, 0.090, 1e-6, 0.500, 1e-6, 1.5, 1e-6, 2, 1e-6, 3])
    lambda_par = l
    E_mean = np.ones_like(bvals)
    E_mean = ((np.sqrt(np.pi) * erf(np.sqrt(bvals * lambda_par))) /
                (2 * np.sqrt(bvals * lambda_par)))

    return E_mean


nvox = 5000 # number of voxels to simulate Try with 1000
radii = np.random.uniform(0.001,15,nvox) # free parameter - cell radius
dees = np.random.uniform(0.5,3,nvox) # free parameter - EES diffusivity
lambdapar = np.repeat(2,nvox) # fixed parameter
E_stick = np.array([astrosticks(l) for l in lambdapar]) 
E_ball = np.array([ball(d) for d in dees])
E_sphere = np.array([sphere(r) for r in radii])
fic = np.expand_dims(np.random.uniform(0.001, 0.999, nvox), axis=1) # free parameter - IC volume fraction
fees = np.expand_dims(np.random.uniform(0.001, 0.999, nvox), axis=1) # free parameter - EES volume fraction
fvasc = 1 - fic - fees # calculate VASC volume fraction
fvasc = fvasc/(fic + fees + fvasc)
A = fvasc
normA = A - min(A)
fvasc = 0.2 * (normA/max(normA)) # constraining fvasc to be realistic for prostate tissue (under 0.2)
fic = fic/(fic + fees + fvasc)
fees = fees/(fic + fees + fvasc)
E_vox = fees*E_ball + fic*E_sphere + fvasc*E_stick
E_vox_real = E_vox + np.random.normal(scale=0.02, size=np.shape(E_vox)) # adding rician noise, snr = 50
E_vox_imag = np.random.normal(scale=0.02, size=np.shape(E_vox))
E_vox = np.sqrt(E_vox_real**2 + E_vox_imag**2) # these are the simulated signals

print("the code ran until the first breakpoint")

'''

## this section will be useful when you want to compare it to NLLS fitting, i'm commenting it out for now

b_values = np.array([1e-6, 90, 1e-6, 500, 1e-6, 1500, 1e-6, 2000, 1e-6, 3000])
bvaluesSI = np.array([i * 1e6 for i in b_values])
Delta = np.array([0.0238, 0.0238, 0.0238, 0.0313, 0.0238, 0.0438, 0.0238, 0.0343, 0.0238, 0.0388])
delta = np.array([0.0039, 0.0039, 0.0039, 0.0114, 0.0039, 0.0239, 0.0039, 0.0144, 0.0039, 0.0189])
gradient_directions = np.loadtxt('./verdict_graddirs.txt', delimiter=',')
acq_scheme = acquisition_scheme_from_bvalues(bvaluesSI, gradient_directions, delta, Delta)

spheresim = sphere_models.S4SphereGaussianPhaseApproximation(diffusion_constant=2e-9)
ballsim = gaussian_models.G1Ball()
sticksim = cylinder_models.C1Stick(lambda_par=2e-9)
astro = sticksim.spherical_mean(acq_scheme)

from dmipy.core.modeling_framework import MultiCompartmentModel
verdict_mod = MultiCompartmentModel(models=[spheresim, ballsim, sticksim])
verdict_mod.set_parameter_optimization_bounds('G1Ball_1_lambda_iso', [0.5e-9, 3e-9])
verdict_mod.set_parameter_optimization_bounds('partial_volume_0', [0.001, 0.999])
verdict_mod.set_parameter_optimization_bounds('partial_volume_1', [0.001, 0.999])
verdict_mod.set_parameter_optimization_bounds('partial_volume_1', [0.001, 0.199])
verdict_mod.set_parameter_optimization_bounds('G1Ball_1_lambda_iso', [0.5e-9, 3e-9])


res = verdict_mod.fit(acq_scheme, E_vox)
f_ic = res.fitted_parameters['partial_volume_0']
f_ees = res.fitted_parameters['partial_volume_1']
f_vasc = res.fitted_parameters['partial_volume_2']
r = res.fitted_parameters['S4SphereGaussianPhaseApproximation_1_diameter']/2
d_ees = res.fitted_parameters['G1Ball_1_lambda_iso']
'''

class Net(nn.Module): # this is the neural network

    def __init__(self, b_values, Delta, delta, gradient_strength, nparams):
        super(Net, self).__init__()

        self.b_values = b_values
        self.Delta = Delta
        self.delta = delta
        self.gradient_strength = gradient_strength

        self.layers = nn.ModuleList()
        for i in range(3): # 3 fully connected hidden layers
            self.layers.extend([nn.Linear(len(b_values), len(b_values)), nn.PReLU()])
        self.encoder = nn.Sequential(*self.layers, nn.Linear(len(b_values), nparams))

    def forward(self, X):
        
        params = torch.nn.functional.softplus(self.encoder(X))
        f_ic = torch.clamp(params[:,0].unsqueeze(1), min=0.001, max=0.999) # parameter constraints
        f_ees = torch.clamp(params[:,1].unsqueeze(1), min=0.001, max=0.999)
        r = torch.clamp(params[:,2].unsqueeze(1), min=0.001, max=14.999)
        d_ees = torch.clamp(params[:,3].unsqueeze(1), min=0.5, max=3)

        SPHERE_TRASCENDENTAL_ROOTS = np.r_[
        # 0.,
        2.081575978, 5.940369990, 9.205840145,
        12.40444502, 15.57923641, 18.74264558, 21.89969648,
        25.05282528, 28.20336100, 31.35209173, 34.49951492,
        37.64596032, 40.79165523, 43.93676147, 47.08139741,
        50.22565165, 53.36959180, 56.51327045, 59.65672900,
        62.80000055, 65.94311190, 69.08608495, 72.22893775,
        75.37168540, 78.51434055, 81.65691380, 84.79941440,
        87.94185005, 91.08422750, 94.22655255, 97.36883035,
        100.5110653, 103.6532613, 106.7954217, 109.9375497,
        113.0796480, 116.2217188, 119.3637645, 122.5057870,
        125.6477880, 128.7897690, 131.9317315, 135.0736768,
        138.2156061, 141.3575204, 144.4994207, 147.6413080,
        150.7831829, 153.9250463, 157.0668989, 160.2087413,
        163.3505741, 166.4923978, 169.6342129, 172.7760200,
        175.9178194, 179.0596116, 182.2013968, 185.3431756,
        188.4849481, 191.6267147, 194.7684757, 197.9102314,
        201.0519820, 204.1937277, 207.3354688, 210.4772054,
        213.6189378, 216.7606662, 219.9023907, 223.0441114,
        226.1858287, 229.3275425, 232.4692530, 235.6109603,
        238.7526647, 241.8943662, 245.0360648, 248.1777608,
        251.3194542, 254.4611451, 257.6028336, 260.7445198,
        263.8862038, 267.0278856, 270.1695654, 273.3112431,
        276.4529189, 279.5945929, 282.7362650, 285.8779354,
        289.0196041, 292.1612712, 295.3029367, 298.4446006,
        301.5862631, 304.7279241, 307.8695837, 311.0112420,
        314.1528990
        ]
        
        alpha = torch.FloatTensor(SPHERE_TRASCENDENTAL_ROOTS) / (r)
        alpha2 = alpha ** 2
        alpha2D = alpha2 * 2
        alpha = alpha.unsqueeze(1)
        alpha2 = alpha2.unsqueeze(1)
        alpha2D = alpha2D.unsqueeze(1)

        gamma = 2.675987e2
        first_factor = -2*(gamma*self.gradient_strength)**2 / 2

        delta = self.delta.unsqueeze(0).unsqueeze(2)
        Delta = self.Delta.unsqueeze(0).unsqueeze(2)
        
        summands = (alpha ** (-4) / (alpha2 * (r.unsqueeze(2))**2 - 2) * (
                            2 * delta - (
                            2 +
                            torch.exp(-alpha2D * (Delta - delta)) -
                            2 * torch.exp(-alpha2D * delta) -
                            2 * torch.exp(-alpha2D * Delta) +
                            torch.exp(-alpha2D * (Delta + delta))
                        ) / (alpha2D)
                    )
                )
        
        # this is where you will eventually need to tweak the network to be bbb-fexi rather than verdict

        xi = (1 - f_ic - f_ees) * ((np.sqrt(np.pi) * torch.erf(np.sqrt(self.b_values * 2))) /
                (2 * np.sqrt(self.b_values * 2)))
        xii = f_ic * torch.exp(torch.FloatTensor(first_factor) * torch.sum(summands, 2))
        xiii = f_ees * torch.exp(-self.b_values * d_ees)        
        X = xi + xii + xiii

        return X, f_ic, f_ees, r, d_ees

print("this is the second breakpoint")

# define network
nparams = 4
b_values = torch.FloatTensor([1e-6, 0.090, 1e-6, 0.500, 1e-6, 1.5, 1e-6, 2, 1e-6, 3])
Delta = torch.FloatTensor([23.8, 23.8, 23.8, 31.3, 23.8, 43.8, 23.8, 34.3, 23.8, 38.8])
delta = torch.FloatTensor([3.9, 3.9, 3.9, 11.4, 3.9, 23.9, 3.9, 14.4, 3.9, 18.9])
gamma = 2.67e2
gradient_strength = torch.FloatTensor([np.sqrt(b_values[i])/(gamma*delta[i]*np.sqrt(Delta[i]-delta[i]/3)) for i,_ in enumerate(b_values)])
net = Net(b_values, Delta, delta, gradient_strength, nparams)

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
optimizer = optim.Adam(net.parameters(), lr = 0.01)

# best loss
best = 1e16
num_bad_epochs = 0
patience = 10

print("third breakpoint")

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
        X_pred, f_ic_pred, f_ees_pred, r_pred, d_ees_pred = net(X_batch)
        loss = criterion(X_pred, X_batch)
        loss.backward()
        for name, param in net.named_parameters():
            if param.grad is not None:
                print(f'Parameter: {name}, Gradient Norm: {torch.norm(param.grad)}')
            else:
                print(f'Parameter: {name}, Gradient: None')
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
    X_real_pred, f_ic, f_ees, r, d_ees = net(torch.from_numpy(E_vox.astype(np.float32)))

f_vasc = 1 - f_ic - f_ees

f_vasc = f_vasc/(f_ic + f_ees + f_vasc)
A = f_vasc
normA = A - min(A)
f_vasc = 0.2 * (normA/max(normA))
f_ic = f_ic/(f_ic + f_ees + f_vasc)
f_ees = f_ees/(f_ic + f_ees + f_vasc)

print("fourth breakpoint")

# check predicted signal against simulated signal

plt.scatter(b_values, E_vox[0,:], label='simulated')
plt.scatter(b_values, X_real_pred[0,:], label='predicted')
plt.legend()

# plot scatter plots to analyse correlation of predicted free params against ground truth

param = [fic, fees, fvasc, radii, dees]
param_f = [f_ic, f_ees, f_vasc, r, d_ees]
param_name = ['fIC', 'fEES', 'fVASC', 'R', 'dEES']
rvals = []

for i,_ in enumerate(param):
    plt.rcParams['font.size'] = '16'
    plt.scatter(param[i], param_f[i], s=2, c='navy')
    plt.xlabel(param_name[i] + ' Ground Truth')
    plt.ylabel(param_name[i] + ' Prediction')
    rvals.append(scipy.stats.pearsonr(np.squeeze(param[i]), np.squeeze(param_f[i])))
    plt.tight_layout
    plt.show()

print(rvals)


print("fifth breakpoint")

## bias-variance calculations

bias_fic = torch.mean(f_ic - fic)
bias_fees = torch.mean(f_ees - fees)
bias_r = torch.mean(r - radii)
bias_dees = torch.mean(d_ees - dees)

var_fic = torch.mean((f_ic - torch.mean(f_ic))**2)
var_fees = torch.mean((f_ees - torch.mean(f_ees))**2)
var_r = torch.mean((r - torch.mean(r))**2)
var_dees = torch.mean((d_ees - torch.mean(d_ees))**2)

mse_fic = torch.mean((f_ic - fic)**2)
mse_fees = torch.mean((f_ees - fees)**2)
mse_r = torch.mean((r - radii)**2)
mse_dees = torch.mean((d_ees - dees)**2)

print(bias_fic, bias_fees, bias_r, bias_dees)
print(var_fic, var_fees, var_r, var_dees)
print(mse_fic, mse_fees, mse_r, mse_dees)


r = r*1e6
d_ees = d_ees*1e9
'''
bias_fic = np.mean(f_ic - fic)
bias_fees = np.mean(f_ees - fees)
bias_r = np.mean(r - radii)
bias_dees = np.mean(d_ees - dees)

var_fic = np.mean((f_ic - np.mean(f_ic))**2)
var_fees = np.mean((f_ees - np.mean(f_ees))**2)
var_r = np.mean((r - np.mean(r))**2)
var_dees = np.mean((d_ees - np.mean(d_ees))**2)

mse_fic = np.mean((f_ic - fic)**2)
mse_fees = np.mean((f_ees - fees)**2)
mse_r = np.mean((r - radii)**2)
mse_dees = np.mean((d_ees - dees)**2)

print(bias_fic, bias_fees, bias_r, bias_dees)
print(var_fic, var_fees, var_r, var_dees)
print(mse_fic, mse_fees, mse_r, mse_dees)'''
print("sixth breakpoint")
end_time = time.time()
elapsed_time = end_time - start_time
print(f"Elapsed time: {elapsed_time} seconds")
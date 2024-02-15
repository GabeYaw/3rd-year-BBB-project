# NN Plots

from NN import *
#from NN_adc_prime import *


plt.figure()
plt.plot(range(1, len(loss_progress) + 1), loss_progress, linestyle='-')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Loss per Epoch')
plt.grid(True)
plt.show(block=False)

plt.figure()
plt.plot(range(1, len(adc_progress) + 1), adc_progress, linestyle='-')
plt.xlabel('Epoch')
plt.ylabel('ADC first voxel')
plt.title('ADC per Epoch')
plt.grid(True)
plt.show(block=False)

plt.figure()
plt.plot(range(1, len(sigma_progress) + 1), sigma_progress, linestyle='-')
plt.xlabel('Epoch')
plt.ylabel('sigma first voxel')
plt.title('sigma per Epoch')
plt.grid(True)
plt.show(block=False)


plt.figure()
plt.plot(range(1, len(axr_progress) + 1), axr_progress, linestyle='-')
plt.xlabel('Epoch')
plt.ylabel('axr first voxel')
plt.title('axr per Epoch')
plt.grid(True)
plt.show(block=False)


plt.figure()
#for i in range(axr_unclamped_progress.shape[0]):
for i in range(1):
    plt.plot(range(1, axr_unclamped_progress.shape[1] + 1), axr_unclamped_progress[i], linestyle='-')
#plt.plot(range(1, len(axr_unclamped_progress) + 1), axr_unclamped_progress, linestyle='-')
plt.xlabel('Epoch')
plt.ylabel('axr unclamped first voxel')
plt.title('axr unclamped per Epoch')
plt.grid(True)
plt.show(block=False)

plt.figure()
#for i in range(adc_unclamped_progress.shape[0]):
for i in range(1):
    plt.plot(range(1, adc_unclamped_progress.shape[1] + 1), adc_unclamped_progress[i], linestyle='-')
plt.xlabel('Epoch')
plt.ylabel('adc unclamped first voxel')
plt.title('adc unclamped per Epoch')
plt.grid(True)
plt.show(block=False)

plt.figure()
#for i in range(sigma_unclamped_progress.shape[0]):
for i in range(1):
    plt.plot(range(1, sigma_unclamped_progress.shape[1] + 1), sigma_unclamped_progress[i], linestyle='-')
plt.xlabel('Epoch')
plt.ylabel('sigma unclamped first voxel')
plt.title('sigma unclamped per Epoch')
plt.grid(True)
plt.show(block=False)


# these 2 plots make less sense
'''plt.figure()
plt.plot(range(1, len(signal_progress) + 1), signal_progress, linestyle='-')
plt.xlabel('Epoch')
plt.ylabel('signal first voxel')
plt.title('signal per Epoch')
plt.grid(True)
plt.show()

plt.figure()
plt.plot(range(1, len(adc_prime_progress) + 1), adc_prime_progress, linestyle='-')
plt.xlabel('Epoch')
plt.ylabel('adc_prime first voxel')
plt.title('adc_prime per Epoch')
plt.grid(True)
plt.show()'''

final_pred_E_vox_detached = final_pred_E_vox.detach().numpy()
"""Was having numpy pytorch issues, so this line helps fix it a bit."""
plt.figure()
plt.scatter(be, sim_E_vox[0,:], label='simulated')
plt.scatter(be, final_pred_E_vox_detached[0,:], label='predicted')
plt.legend()

# plot scatter plots to analyse correlation of predicted free params against ground truth
plt.figure()

param_sim = [sim_adc, sim_sigma, sim_axr]
param_pred = [final_pred_adc, final_pred_sigma, final_pred_axr]
param_name = ['ADC', 'Sigma', 'AXR']

rvals = []

for i,_ in enumerate(param_sim):
    plt.rcParams['font.size'] = '16'
    plt.scatter(param_sim[i], param_pred[i], s=2, c='navy')
    plt.xlabel(param_name[i] + ' Ground Truth')
    plt.ylabel(param_name[i] + ' Prediction')
    rvals.append(scipy.stats.pearsonr(np.squeeze(param_sim[i]), np.squeeze(param_pred[i])))
    plt.tight_layout
    plt.show()

print(rvals)
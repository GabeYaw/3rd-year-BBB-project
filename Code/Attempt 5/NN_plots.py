# NN Plots

from NN import *
import os
import datetime

final_pred_E_vox_detached = X_final_pred.detach().numpy()
"""Was having numpy pytorch issues, so this line helps fix it a bit."""
plt.figure()
plt.scatter(be, sim_E_vox[0,:], label='simulated')
plt.scatter(be, final_pred_E_vox_detached[0,:], label='predicted')
plt.legend()

# plot scatter plots to analyse correlation of predicted free params against ground truth

param_sim = [sim_adc, sim_sigma, sim_axr]
param_pred = [adc_final_pred, sigma_final_pred, axr_final_pred]
param_name = ['ADC', 'Sigma', 'AXR']

rvals = []

# Create a new folder with the current datetime
now = datetime.datetime.now()
folder_name = now.strftime("%Y-%m-%d_%H-%M-%S")+" nvox = "+str(nvox)+" patience = "+str(patience)
folder_path = os.path.join('/Users/admin/Downloads', str(folder_name))

os.makedirs(folder_path)

for i,_ in enumerate(param_sim):
    plt.rcParams['font.size'] = '16'
    plt.figure()  # Create a new figure for each loop
    plt.scatter(param_sim[i], param_pred[i], s=2, c='navy')
    plt.xlabel(param_name[i] + ' Ground Truth')
    plt.ylabel(param_name[i] + ' Prediction')
    r_value,p_value = scipy.stats.pearsonr(np.squeeze(param_sim[i]), np.squeeze(param_pred[i]))
    plt.text(0.95, 0.05, f"r = {r_value:.2f}", ha='right', va='bottom', transform=plt.gca().transAxes)
    rvals.append([r_value, p_value])
    plt.tight_layout()
    image_path = os.path.join(folder_path, f'{param_name[i]}_scatter.png')
    plt.savefig(image_path)
    plt.show(block=False)

plt.show()  # Display all the plots at the same time on their respective figures

print("Pearson correlation coefficient",rvals)

adc_final_pred = adc_final_pred.detach().numpy()
sigma_final_pred = sigma_final_pred.detach().numpy()
axr_final_pred = axr_final_pred.detach().numpy()


bias_adc = np.mean(adc_final_pred - sim_adc)
bias_sigma = np.mean(sigma_final_pred - sim_sigma)
bias_axr = np.mean(axr_final_pred - sim_axr)

var_adc = np.mean((adc_final_pred - np.mean(sim_adc))**2)
var_sigma = np.mean((sigma_final_pred - np.mean(sim_sigma))**2)
var_axr = np.mean((axr_final_pred - np.mean(sim_axr))**2)

mse_adc = np.mean((adc_final_pred - sim_adc)**2)
mse_sigma = np.mean((sigma_final_pred - sim_sigma)**2)
mse_axr = np.mean((axr_final_pred - sim_axr)**2)

print(f"Bias ADC: {bias_adc}, Bias Sigma: {bias_sigma}, Bias AXR: {bias_axr}")
print(f"Variance ADC: {var_adc}, Variance Sigma: {var_sigma}, Variance AXR: {var_axr}")
print(f"MSE ADC: {mse_adc}, MSE Sigma: {mse_sigma}, MSE AXR: {mse_axr}")
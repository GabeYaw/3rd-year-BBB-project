# NLLS Plots

from NLLS import *
import os
import datetime


#plotting the sse
"""plt.figure()
plt.hist(sses.flatten(), bins=200)
plt.xlabel("SSE histogram")"""

plt.figure()
# for first voxel
plt.scatter(be, sim_E_vox[0,:], label='simulated')
plt.scatter(be, NLLS_E_vox_all[0,:], label='predicted')
plt.xlabel("be")
plt.ylabel("Normalised signal")
plt.legend()

# plot scatter plots to analyse correlation of predicted free params against ground truth

param_sim = [sim_adc, sim_sigma, sim_axr]
param_pred = [NLLS_adc_all, NLLS_sigma_all, NLLS_axr_all]
param_name = ['ADC', 'Sigma', 'AXR']

rvals = []

now = datetime.datetime.now()
folder_name = now.strftime("%Y-%m-%d_%H-%M-%S")+" nvox = "+str(nvox)
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

print("Pearson correlation coefficient",rvals)

bias_adc = np.mean(NLLS_adc_all - sim_adc)
bias_sigma = np.mean(NLLS_sigma_all - sim_sigma)
bias_axr = np.mean(NLLS_axr_all - sim_axr)

var_adc = np.mean((NLLS_adc_all - np.mean(sim_adc))**2)
var_sigma = np.mean((NLLS_sigma_all - np.mean(sim_sigma))**2)
var_axr = np.mean((NLLS_axr_all - np.mean(sim_axr))**2)

mse_adc = np.mean((NLLS_adc_all - sim_adc)**2)
mse_sigma = np.mean((NLLS_sigma_all - sim_sigma)**2)
mse_axr = np.mean((NLLS_axr_all - sim_axr)**2)

print("Bias ADC: ", bias_adc, "Bias Sigma: ", bias_sigma, "Bias AXR: ", bias_axr)
print("Variance ADC: ", var_adc, "Variance Sigma: ", var_sigma, "Variance AXR: ", var_axr)
print("MSE ADC: ", mse_adc, "MSE Sigma: ", mse_sigma, "MSE AXR: ", mse_axr)



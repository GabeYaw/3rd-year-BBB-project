## Lizzie's Matlab code
bf =    0-250   * 1e6    [s/m2] = 2.5 * 1e-7 [ms/um2]
be =    0-250   * 1e6    [s/m2] = 2.5 * 1e-7 [ms/um2]
tm =    20-400  * 1e-3   [s]

ADC =   0.1-3.5 * 1e-9   [m2/s] = 3.5 * 1e6 [mm2/s]
Sigma = 0-1              [a.u.]
AXR =   0.1-20           [s-1]

## NN and sim files in python
bf =    0-250   * 1e-3   [ms/um2] or [ks/m2] = 250[s/m2]
be =    0-250   * 1e-3   [ms/um2] or [ks/m2] = 250[s/m2]
tm =    20-400  * 1e-3   [s]

ADC =   0.1-3.5          [um2/ms] or [mm/s2] = 3.5 * 1e-3 [m2/s]
Sigma = 0-1              [a.u.]
AXR =   0.1-20           [s-1]

## Comparison
tm, sigma and AXR are the same
be bf and ADC are different so need to be checked.

adc_prime = adc * (1 - sigma * torch.exp(- tm * axr))
E_vox = torch.exp(- adc_prime * be)
%==========================================================================
% For given variable, what would the signal be
% Generate signals using AXR FEXI model
%  (Lasic 2011, MRM)
%
% Use: s = axr_sim(adc, sigma, axr, bf, be, tm)
%
% Inputs    - adc:      apparent diffusion coefficient [m2/s]
%           - sigma:    filter efficiency
%           - axr:      exchange rate [1/s]
%           - bf:       filter block b-value [m2/s]
%           - be:       encoding block b-value [m2/s]
%           - tm:       mixing time [s]
%
% Output: 	- s:        signal (sum of the magnetisations)
%
% Author: E Powell, 23/08/23
%
%==========================================================================    
function s = axr_sim(adc, sigma, axr, bf, be, tm)

    % find equilibrium acquisition (bf==0, tm==min(tm)), and set tm=inf for this
    % why?
    tm(bf==0 & tm==min(tm)) = inf;
    
    % calculate ADC as fnc of mixing time
    adc_tm = adc * (1 - sigma*exp(-axr*tm));
    
    % compute signal
    s = exp(-adc_tm.*be);
    
end
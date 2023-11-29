% example code for Gabe
% E Powell, 24/11/2023

clearvars; clc;

bf = [0 0 250 250 250 250 250 250]'*1e6;     % filter b-values [s/m2]
be = [0 250 0 250 0 250 0 250]'*1e6;         % encoding b-values [s/m2]
tm = [20 20 20 20 200 200 400 400]'*1e-3;    % mixing time [s]

sim_adc = 1e-9;                             % ADC, simulated [m2/s]
sim_sig = .2;                               % sigma, simulated [a.u.]
sim_axr = 3;                                % AXR, simulated [s-1]

% simulate signals
s = axr_sim(sim_adc, sim_sig, sim_axr, bf, be, tm); 

% fit model to simulated signals and estimate parameters
init = [1.1e-9 .15 3.5];
lb = [.1e-9 0 .1];
ub = [3.5e-9 1 20];

[fit_adc, fit_sig, fit_axr] = axr_fit(bf, be, tm, s, init, lb, ub); 

% print and compare simulated vs fitted
[sim_adc, sim_sig, sim_axr].*[1e9 1 1]
[fit_adc, fit_sig, fit_axr].*[1e9 1 1]
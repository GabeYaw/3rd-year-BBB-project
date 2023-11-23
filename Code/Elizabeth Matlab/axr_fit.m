%==========================================================================
% Estimate AXR FEXI model parameters
%  (Lasic 2011, MRM)
%
% Use: [adc, sigma, axr] = axr_fit(bf, be, tm, smeas, init, lb, ub)
%
% Inputs    - bf:       filter block b-value [m^2/s] 20x1
%           - be:       encoding block b-value [m^2/s] 20x1 
%           - tm:       mixing time [s] 20x1 
%           - smeas:    measured signal (normalised) 20x1?
%           - init:     initial values  [adc, sigma, axr] [m2/s a.u. 1/s] 3x1
%           - lb:       lower bounds   [adc, sigma, axr] [m2/s a.u. 1/s] 3x1
%           - ub:       upper bounds   [adc, sigma, axr] [m2/s a.u. 1/s] 3x1
%
% Output: 	- adc:      fitted ADC [m2/s] single value
%           - sigma:	fitted filter efficiency single value 
%           - axr:      fitted AXR [1/s] single value
%
% Author: E Powell, 23/08/23
%
%==========================================================================
function [adc, sigma, axr] = axr_fit(bf, be, tm, smeas, init, lb, ub)
    
    % only use parpool if multiple inits
    if size(init,1) >= 25
        useparpool = true;
    else
        useparpool = false;
    end
    
    % scale sequence values (values to be fitted, i.e. diffusivities, need to be ~1)
    bf = bf * 1e-9;
    be = be * 1e-9;

    % scale initial parameter values and bounds
    all_params = init .* [1e9 ones(1, size(init,2)-1)];
    lb(1) = lb(1)*1e9;
    ub(1) = ub(1)*1e9;
	          
    % find if any parameters are fixed
    idx_free = find(lb~=ub);
    idx_fixed = find(lb==ub);
        
    % select initial values and bounds only for free params
    free_params = all_params(:,idx_free);
    lb = lb(idx_free);
    ub = ub(idx_free);
    
    opt = optimset('display', 'off'); 
    fitting = @(free_params) fit_axr_sse(free_params, all_params, idx_free, bf, tm, be, smeas);
    if useparpool
        parfor i = 1:size(init,1)
            opt = optimset('Display','off');
            [x(i,:), fval(i)] = fminsearchbnd(fitting, free_params(i,:), lb, ub, opt);
        end
    else
        for i = 1:size(init,1)
            opt = optimset('Display','off');
            [x(i,:), fval(i)] = fminsearchbnd(fitting, free_params(i,:), lb, ub, opt);
        end

    end
    idx = find(min(fval));
    if isempty(idx) % happens when signal = 0 in all vols (e.g. at image edges) 
        x = lb; 
        fval = nan; 
    else
        x = x(idx,:);
        fval = fval(idx);
    end
    
    % extract parameters from fitting procedure
    fitted_params = zeros(size(all_params));
    fitted_params(idx_free) = x;
    fitted_params(idx_fixed) = all_params(idx_fixed);
    
    adc = fitted_params(1);
    sigma = fitted_params(2);
    axr = fitted_params(3);

    % revert diffusivity scaling
    adc = adc*1e-9;
    
end

%==========================================================================
% Estimate S(tm) given D1, D2, f1_eq, f1_0, k, 2 compartments
% Inputs    - free_params:  parameter values being fitted
%                           [adc sigma axr] 
%           - scheme:       acquisition parameters [nx3]
%                           [bf, be, tm]
%           - all_params:   array of all parameters
%                           [adc sigma axr] 
%           - idx_free:     indices of free parameters within free_params
%           - idx_adc:      indices into scheme of acquisitions used for ADC calc
%           - idx_s0sf:     indices into scheme of acquisitions used for S0/Sf calc
%                       - NORMALISED to b=0 for each bf, tm combination so that S0 = Sbf = 0
%           
% Outputs   - sfit:     fitted signal [1, ntm x nbval]
%
function sse = fit_axr_sse(free_params, all_params, idx_free, bf, tm, be, smeas)  
    
	% extract dependent variables
    all_params = all_params(1,:);
    all_params(idx_free) = free_params;
    
    adc = all_params(1);
    sigma = all_params(2);
    axr = all_params(3);

    % calculate ADC'(tm) from measured data
    nsf = size(unique([bf(:),tm(:)],'rows'),1);                             % number of unique bf,tm combos
    univols = unique([bf(:),tm(:)],'rows','stable');                        % find the pairs of data points used to calculate each ADC'(tm)
    adc_tm_calc = zeros(1,nsf);
    for v = 1:nsf
        ix1 = find(sum(univols(v,:)==[bf,tm],2)==2 & be==0);
        ix2 = find(sum(univols(v,:)==[bf,tm],2)==2 & be>0);
        adc_tm_calc(v) = -1/(be(ix2)-be(ix1)) * log(smeas(ix2)/smeas(ix1));
    end

    % find equilibrium acquisition (bf==0, tm==min(tm)), and set tm=inf for this
    idx_eq = find(univols(:,1)==0 & univols(:,2)==min(univols(:,2)));       % equilibrium scans without filter
    tm = univols(:,2);
    tm(idx_eq) = inf;
    
    % estimate ADC'(tm) by fitting model
    adc_tm_fit = adc * (1 - sigma*exp(-tm*axr));
    
    % sos difference
    sse = sum((adc_tm_calc(:) - adc_tm_fit(:)).^2);

end


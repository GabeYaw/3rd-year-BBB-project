%==========================================================================
% for a given signal estimate the variables that produced it. 
% Estimate adc, sigma and axr
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
    % size(init,1) is num of rows in init
    if size(init,1) >= 25
        %par pool is a way to split up task computationally. 
        %Does it have a python equivalent and will inits ever be larger
        %than 3x1?
        useparpool = true;
    else
        useparpool = false;
    end
    
    % scale sequence values (values to be fitted, i.e. diffusivities, need to be ~1)
    bf = bf * 1e-9;
    be = be * 1e-9;

    % scale initial parameter values and bounds
    % size(init,2) is num of columns in init
    % times the first column in init by 1e9, and leave rest the same. 
    all_params = init .* [1e9 ones(1, size(init,2)-1)];
    %first column
    lb(1) = lb(1)*1e9;
    ub(1) = ub(1)*1e9;
	          
    % find if any parameters are fixed
    % ~= is not equal
    % It creates a 1d array where the values are the indices which satifies
    % the condition.
    idx_free = find(lb~=ub);
    idx_fixed = find(lb==ub);
        
    % select initial values and bounds only for free params
    free_params = all_params(:,idx_free);
    lb = lb(idx_free);
    ub = ub(idx_free);
    
    %hide output during iterations. Not sure what purpose is or what is
    %python equivalent
    opt = optimset('display', 'off'); 

    %create an anon function called fitting which calls fit_axr_sse with
    %all the other parameters, but only free_params used as a arguement
    %that will vary. other 6 will stay same between iterations. 
    % the output of the function is the sum of squares error (sse) of the mixing time. 
    fitting = @(free_params) fit_axr_sse(free_params, all_params, idx_free, bf, tm, be, smeas);


    if useparpool
        %for rows in init
        parfor i = 1:size(init,1)
            opt = optimset('Display','off');

            %fminsearchbnd finds min val of function, in this case fitting
            %function, ie where sse is lowest.
            % Free_params is the inital values to start search at.
            % x(i,:) is the location of the minimum
            % fval(i) is the value of function when evaluated at the 
            % minimum
            % the output containts the same number of rows as free_params
            % (which comes from init)
            [x(i,:), fval(i)] = fminsearchbnd(fitting, free_params(i,:), lb, ub, opt);
        end
    
    else
        for i = 1:size(init,1)
            opt = optimset('Display','off');
            [x(i,:), fval(i)] = fminsearchbnd(fitting, free_params(i,:), lb, ub, opt);
        end
    end

    %see how many occurrances of the minimum value of the function
    idx = find(min(fval));

    if isempty(idx) % happens when signal = 0 in all vols (e.g. at image edges) 
        %set location of minimum to lower bound
        x = lb; 
        %set value of minimun to nan
        fval = nan; 

    else
        %filter x to just the minimum locations
        x = x(idx,:);
        % filter fval to just the minimum value. 
        fval = fval(idx);
    end
    
    % extract parameters from fitting procedure
    fitted_params = zeros(size(all_params));
    %set value to x (min locations) at all idx_free points
    fitted_params(idx_free) = x;
    %copy value from all params for the fixed params params
    fitted_params(idx_fixed) = all_params(idx_fixed);
    
    adc = fitted_params(1);
    sigma = fitted_params(2);
    axr = fitted_params(3);

    % revert diffusivity scaling
    adc = adc*1e-9;
    
end

%==========================================================================
% Calculate sum of square error between mixing time estimate and true value. 

% I think this is old: Estimate S(tm) given D1, D2, f1_eq, f1_0, k, 2 compartments
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
    %first row only
    all_params = all_params(1,:);
    % set param to free value where index say to do so
    all_params(idx_free) = free_params;
    
    adc = all_params(1);
    sigma = all_params(2);
    axr = all_params(3);

    % calculate ADC'(tm) from measured data

    % using bf and tm as column vectors and as first 2 columns, see how 
    % many rows they contain
    % size(unique(...),1) takes num of rows.
    % unique() keeps only unique combinations.
    nsf = size(unique([bf(:),tm(:)],'rows'),1);                             % number of unique bf,tm combos
    
    %stable keeps order it was put in (with duplicates removed)
    univols = unique([bf(:),tm(:)],'rows','stable');                        % find the pairs of data points used to calculate each ADC'(tm)
    
    %same length as bf and tm
    adc_tm_calc = zeros(1,nsf);

    % [bf,tm] has some duplicate rows
    % univols is [bf,tm] without duplicates

    for v = 1:nsf
        %ix1 is an index 
        % when b value is zero and where both values in row v of univols
        % are the same as the values in [bf,tm]
        % what is the physical meaning of this?

        ix1 = find(sum(univols(v,:)==[bf,tm],2)==2 & be==0);

        % when b value is none zero and where both values in row v of univols
        % are the same as the values in [bf,tm]
        % what is the physical meaning of this?
        ix2 = find(sum(univols(v,:)==[bf,tm],2)==2 & be>0);

        %smeas is normalised signal
        %calculate the adc value and make it row v of adc_tm_calc
        % see eq 7 of https://doi.org/10.1016/j.neuroimage.2020.117039
        adc_tm_calc(v) = -1/(be(ix2)-be(ix1)) * log(smeas(ix2)/smeas(ix1));
    end

    % find equilibrium acquisition (bf==0, tm==min(tm)), and set tm=inf for this
    
    % create an index of when bf is 0, and tm is minimum
    idx_eq = find(univols(:,1)==0 & univols(:,2)==min(univols(:,2)));       % equilibrium scans without filter
    tm = univols(:,2);
    %set the tm to infinity where condition was met
    %why?
    tm(idx_eq) = inf;
    
    % estimate ADC'(tm) by fitting model
    % eq 2 from https://doi.org/10.1016/j.neuroimage.2020.117039
    adc_tm_fit = adc * (1 - sigma*exp(-tm*axr));
    
    % sum of squares difference
    sse = sum((adc_tm_calc(:) - adc_tm_fit(:)).^2);

end


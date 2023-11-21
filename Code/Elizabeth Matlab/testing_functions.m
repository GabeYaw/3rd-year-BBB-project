adc = 5;
sigma = 2;
axr = 4;
bf = rand(1, 20);
be = rand(1, 20);
tm = rand(1, 20);


output = axr_sim(adc,sigma,axr,bf,be,tm);
[adc, sigma, axr] = axr_fit(bf, be, tm, smeas, init, lb, ub);
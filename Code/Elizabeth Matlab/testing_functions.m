adc = 5;
sigma = 2;
axr = 4;
be =[0,0,0,0,0,0,0,0,0,0,250,250,250,250,250,250,250,250,250,250]';
bf =[1e-6, 0.090, 1e-6, 0, 1e-6, 1.5, 1e-6, 2, 1e-6, 3,1e-6, 0, 1e-6, 0.500, 1e-6, 1.5, 1e-6, 2, 1e-6, 3]';
tm =[1e-6, 0.090, 1e-6, -1, 1e-6, 1.5, 1e-6, 2, 1e-6, 3,1e-6, 0.090, 1e-6, 0.500, 1e-6, 1.5, 1e-6, 2, 1e-6, 3]';
smeas = rand(20,1);
init = rand(1,3);
lb = rand(1,3);
ub = lb + 5;


%output = axr_sim(adc,sigma,axr,bf,be,tm);
%[adc, sigma, axr] = axr_fit(bf, be, tm, smeas, init, lb, ub);


ix1=[1:20]';
ix2=[21:40]';
v=1;
univols = unique([bf(:),tm(:)],'rows','stable'); 
adc_tm_calc(v) = -1/(be(ix2)-be(ix1)) * log(smeas(ix2)/smeas(ix1));
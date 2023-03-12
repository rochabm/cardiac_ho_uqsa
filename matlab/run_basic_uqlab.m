% UQLab Code for the paper
% Polynomial chaos expansion surrogate modeling of passive cardiac 
% mechanics using the Holzapfel-Ogden constitutive model
% Authors: J. O. Campos, R. M. Guedes, Y. B. Werneck, L. P. S. Barra, 
%          R. W. dos Santos, B. M. Rocha
% MATLAB Code Author: Yan B. Werneck
% Model: Holzapfel-Ogden Full/Original Model with 8 parameters

% Load Train Data
Nt=1500;
TrainSet=load("trainData.mat");
TrainSet=TrainSet.data;
Xtrain=TrainSet(1:Nt,1:8);
Ytrain=TrainSet(1:Nt,9:14);

% Load Validation (Test) Data
Nv=1000;
ValSet=load("testData.mat");
ValSet=ValSet.data;
Xval=ValSet(1:Nv,1:8);
Yval=ValSet(1:Nv,9:14);

% Define model using uqlab
nP=8 % number of input parameters

% Parameters: a b af bf as bs afs bfs
% Baseline values
vals=[150.0 6.0 116.85 11.83425 372.0 5.16 410.0 11.3]; 
for ii = 1:nP
    InputOpts.Marginals(ii).Type = 'Uniform';
    InputOpts.Marginals(ii).Parameters = [0.7*vals(ii),1.3*vals(ii)]; % 0.7-1.3 variation
end
myInput = uq_createInput(InputOpts);

% Define the experiment using uqlab

% OLS ordinary least square method
MetaOpts.Type = 'Metamodel';
MetaOpts.MetaType = 'PCE';
MetaOpts.Method = 'OLS'; % OLS or OMP
MetaOpts.Degree = 2:4;
MetaOpts.ExpDesign.X = Xtrain;
MetaOpts.ExpDesign.Y = Ytrain;
MetaOpts.ValidationSet.X = Xval;
MetaOpts.ValidationSet.Y = Yval; 

% Calculate PCE coeficients, we'll perform the adpative experiment,
% exploring basis of 2:6 degrees, once for each QoI
myPCE = uq_createModel(MetaOpts);

% Sample validation data with emulator
YPCE = uq_evalModel(myPCE,Xval);

% Calculate validation metrics
dif   = abs(YPCE-Yval)./abs(Yval);  % Relative error
dif2  =  dif./Yval;
meanE = mean((dif));
minE  = min((dif));
maxE  = max((dif));

corrcoefs = zeros([6,1]);
seldeg = zeros([6,1]);
for i = 1:6
    corrcoefs(i) = corr(Yval(:,i),YPCE(:,i));
    seldeg(i) = myPCE.PCE(i).Basis.Degree;
end
avgcoef = mean(corrcoefs);

fprintf("Fitting and validation completed! \n");

fprintf("Selected degrees \n");
disp(seldeg)
fprintf(" \n");

fprintf("Corr Coef \n");
disp(corrcoefs)

fprintf("Average: %f \n",avgcoef);
fprintf(" \n");

fprintf("Mins \n");
disp(minE)
fprintf(" \n");

fprintf("Max \n");
disp(maxE)
fprintf(" \n");

fprintf("Mean \n");
disp(meanE)
fprintf(" \n");
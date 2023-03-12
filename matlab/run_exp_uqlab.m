% UQLab Code for the paper
% Polynomial chaos expansion surrogate modeling of passive cardiac 
% mechanics using the Holzapfel-Ogden constitutive model
% Authors: J. O. Campos, R. M. Guedes, Y. B. Werneck, L. P. S. Barra, 
%          R. W. dos Santos, B. M. Rocha
%
% MATLAB Code Author: Yan B. Werneck
% Model: Holzapfel-Ogden Full/Original Model with 8 parameters

clc; clear all;
uqlab
%hold off

% Set results folder
folder = "OMP_new2"
mkdir(folder);
folder = folder + "/"

% Read sets
%TrainSet=readmatrix("trainData.txt");
TrainSet = load("trainData.mat");
TrainSet = TrainSet.data;

%ValSet=readmatrix("testData.txt");
ValSet = load("testData.mat");
ValSet = ValSet.data;

% Define set sizes to be used
NMax = 1500;
SamplesSizes = 10:10:NMax;      % 0 to 1500, step 10
[trash,N] = size(SamplesSizes); % N=150

% Store results: one row for each sample size, one column for each Qoi
meansE=zeros([N,6]);
rmseArray=zeros([N,6]);
rsqds=zeros([N,6]);
seldegs=zeros([N,6]);
avgcoefs=zeros([N,1]);

[valSetSize,trash]=size(ValSet);

valR=zeros([N,valSetSize,6]);

sblFos=zeros(N,6,8);
sblTos=zeros(N,6,8);

avgs=zeros([N,6]);
stds=zeros([N,6]);
covs=zeros([N,6]);

% Fit PCEs using different number of samples and store data
for i=1:N
    nsamples = SamplesSizes(i)    
    fprintf('Caso %d - Training size %d \n\n', i, nsamples)

    % Perform fitting and validation for each sample size
    [avgcoef,seldeg,rsqd,rmse,meanE,em,avg,std,cov,sblFo,sblT]=train_validate(nsamples,false,TrainSet,ValSet);
    
    % Load validation data to main results matrix
    meansE(i,:) = meanE;
    rmseArray(i,:) = rmse;
    rsqds(i,:) = rsqd;
    seldegs(i,:) = seldeg;
    avgcoefs(i) = avgcoef;
    
    avgs(i,:) = avg;
    covs(i,:) = cov;
    stds(i,:) = std;
    
    % Sobol Indices
    sblFos(i,:,:) = sblFo; 
    sblTos(i,:,:) = sblT; 
    valR(i,:,:) = em;
end

% Load data
for i=1:6
    %subplot(2,3,i);
    [coef,melhor]=max(rsqds(:,i));

    MM(i,1:4)=[i,seldegs(melhor,i),coef,SamplesSizes(melhor)]; % Best emulator (degree and Sample size) for each qoi
    MF(i,:)=sblFos(melhor,i,:); % First Order Sobol
    MT(i,:)=sblTos(melhor,i,:); % Total Order Sobol each, using the best emulator for each qoi
    T(1:3,i)= [avgs(melhor,i),stds(melhor,i),covs(melhor,i) ]; %% for the table 3 , using eachs qoi's best emulator
    MD(:,i)=seldegs(:,i); % For each run and each qoi, wich degree was selected
    MC(:,i)=rsqds(:,i); % For each run, corr coef of each qoi
    M(:,i)=valR(melhor,:,i); % For each qoi, the best emulator response for the validation set, orderned like the validation set


end

% Write data to output files within the 'folder'

header={'#X1','X2','X3','X4','X5','X6','X7','X8'};
write_data_to_txt(fullfile(folder, 'SobolFirsOrderData.txt'), header,MF);
write_data_to_txt(fullfile(folder, 'SobolTotalOrderData.txt'), header,MT);

header={'#QOI','SEL DEG','CORR COEF','Samples'};
write_data_to_txt(fullfile(folder, 'Table3.txt'), header,T);

header={'#QOI','SEL DEG','CORR COEF','Samples'};
write_data_to_txt(fullfile(folder, 'Selected_bestPCE_data.txt'), header,MM);

header={'#QOI1','QOI2','QOI3','QOI4',"QOI5",'QOI6'};

%write_matrix_to_txt(fullfile(folder, 'Samples_Size_used.txt'), SA);
write_data_to_txt(fullfile(folder, 'HistData.txt'), header, M);

% Data (rows) for each Qoi (columns)
write_data_to_txt(fullfile(folder, 'CorreCoefs_Data.txt'), header, MC);
write_data_to_txt(fullfile(folder, 'Selected_Degree_Data.txt'), header, MD);
write_data_to_txt(fullfile(folder, 'MEANSERR.txt'), header, meansE);
write_data_to_txt(fullfile(folder, 'RMSE.txt'), header, rmseArray);

fprintf('\nDone\n');
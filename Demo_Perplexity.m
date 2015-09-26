%% Demo Matlba code for:
% Mingyuan Zhou, "Beta-negative binomial process and exchangeable random partitions
% for mixed-membership modeling," NIPS2014, Montreal, Canada, Dec. 2014.

figure
addpath('data')
%download your document data into this folder
dataset = 1 %2,3

%Topic Dirichlet hyper-parameter
eta = 0.05 %[0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.10, 0.25, 0.50]

%random seed
state = 0 %1, 2,3, 4

%Alogrithm to use:

%Collapsed Gibbs sampler for beta-negative binomial process
sampler = 'Beta_NB_collapsed' 

%Direct Assignment Gibbs sampler for hierachical Dirichlet process
%sampler = 'HDP_DirectAssignment'

%Collapsed sampler for gamma-negative binomial process (the gamma process
%weights r_k are not marginalized out)
%sampler = 'Gamma_NB_partially_collapsed'

switch dataset
    case 1
        disp('psychreview')
        load 'bagofwords_psychreview';
        load 'words_psychreview';
        %This dataset is available at
        %http://psiexp.ss.uci.edu/research/programs_data/toolbox.htm
        X = sparse(WS,DS,1,max(WS),max(DS));
        dex = (sum(X>0,2)<5);
        X = X(~dex,:);
        WO = WO(~dex);
        datasetname = 'PsyReview';
    case 2
        disp('JACM')
        %The JACM dataset is available at
        %http://www.cs.princeton.edu/~blei/downloads/
        [X,WO] = InitJACM;
        %load JACM_MZ.mat
        dex=(sum(X>0,2)<=0);
        WO = WO(~dex);
        datasetname = 'JACM';
    case 3
        %The NIPS12 is available at
        %http://www.cs.nyu.edu/~roweis/data/nips12raw_str602.mat
        load nips12raw_str602
        X = counts;
        datasetname = 'NIPS';
end


Burnin = 1000;
Collection = 1500;
maxIter=Burnin+Collection;
CollectionStep = 1;

percentage=0.5;
%use 50% of the word tokens in each document for training

model = 'PFA';

K_init = 0; 
%K_init = 400
%K_init is the intial number of topics


[ave,Phi,Theta,output]=PFA_MultTM(X,model,sampler,eta,percentage,K_init,state,Burnin,Collection,CollectionStep);
state
sampler
eta=output.eta
aveK = mean(ave.K(Burnin+1:end))
Perplexity = exp(-ave.loglike(end))





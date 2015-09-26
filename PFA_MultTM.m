function [ave,Phi,Theta,output]=...
    PFA_MultTM(X,model,sampler,eta,percentage,K,state,burnin,collection,CollectionStep)

%%
% Collapsed Gibbs sampling for the beta-negative binomial process
% multinomial topic model and Poisson factor analysis

% Matlba code for:
% Mingyuan Zhou, "Beta-negative binomial process and exchangeable random partitions
% for mixed-membership modeling," NIPS2014, Montreal, Canada, Dec. 2014.

%First Version: May, 2014
% Second Version: July, 2015
% Updated: Septermber, 2015
%
% Coded by Mingyuan Zhou,
% http://mingyuanzhou.github.io/
% Copyright (C) 2015, Mingyuan Zhou.

%% Related code for 
% Blocked Gibbs sampling with finte truncation is described in:
%
% M. Zhou and L. Carin, "Negative Binomial Process Count and Mixture Modeling,"
% IEEE Trans. Pattern Analysis and Machine Intelligence, vol. 37, pp. 307-320, Feb. 2015.
% Code for that paper (NBP_PFA_mex_v1.m) can be download from 
% http://mingyuanzhou.github.io/Softwares/NBP_PFA_v1.zip


if ~exist('model','var')
    model = 'PFA';
end
if ~exist('sampler','var')
    sampler = 'Beta_NB_collapsed';
end
if ~exist('eta','var')
    eta=0.05;
end
if ~exist('percentage','var')
    percentage=101;
end
if ~exist('K','var')
    K=400;
end
if ~exist('state','var')
    state=1;
end
if ~exist('burnin','var')
    burnin=500;
end
if ~exist('collection','var')
    collection=500;
end
if ~exist('CollectionStep','var')
    CollectionStep=5;
end



maxIter=burnin + collection;

K_star=0;

IsPlot = true;
[V,N] = size(X);
P=V;
Phi = rand(P,K);
Phi = bsxfun(@rdivide,Phi,sum(Phi,1));
Theta = zeros(K,N)+1/K;


rng(state,'twister');
% rand('state',state)
% randn('state',state)
%% X is the input term-document count matrix
% Partition X into training and testing, Xtrain is the training count matrix, WS and DS are the indices for word tokens
[Xtrain,Xtest,WS,DS,WordTrainS,DocTrainS]= PartitionX(X,percentage);
%[Xtrain,Xtest,WS,DS,WordTrainS,DocTrainS]= PartitionX_v1(X,percentage);
WS = WS(WordTrainS);
DS = DS(WordTrainS);
ZS = DS-DS;

Yflagtrain = Xtrain>0;
Yflagtest = Xtest>0;
loglikeTrain = []; loglike=[];
ave.loglike=[]; ave.PhiTheta = sparse(V,N); ave.Count = 0;
ave.PhiThetaSum = zeros(1,N);
ave.K=zeros(1,maxIter);  ave.gamma0=zeros(1,maxIter);


%% Intilization
c=1; gamma0=1;
p_i=0.5*ones(1,N);
r_i=0.1*ones(1,N);
r_star = 1;
r_k = ones(K,1);

disp(sampler)
disp(['state=',num2str(state)])
disp(['eta=',num2str(eta)])
disp(['K_init=',num2str(K)])
disp(['Percentage=',num2str(percentage)])

XtrainSparse= sparse(Xtrain);
Xmask=sparse(X);

if K==0
    DSZS = zeros(N,K);
    WSZS = zeros(V,K);
    TS = ZS-ZS+0;
else
    % K=400;
    ZS = randi(K,length(DS),1);
    DSZS = full(sparse(DS,ZS,1,N,K));
    WSZS = full(sparse(WS,ZS,1,V,K));
    TS=ones(size(ZS));
end
n_dot_k = sum(DSZS,1)';
ell_dot_k=sum(DSZS>0,1)';

a0=1e-2; b0=1e-2; a0=1e-2; b0=1e-2;
loglike=[];
loglikeTrain=[];

fprintf('\n');
text=[];
for iter=1:maxIter
    
    switch sampler
        case {'Gamma_NB_partially_collapsed', 'HDP_DirectAssignment'}
            %% Random scan, optional
            dex111=randperm(length(ZS));
            ZS=ZS(dex111); DS=DS(dex111); WS=WS(dex111);
            %% Partially collapsed inference for the gamma-negative binomial process, the gamma process weights r_k are not marginalized out
            [WSZS,DSZS,n_dot_k,r_k,r_star,ZS] = PFA_GNBP_partial(WSZS,DSZS,n_dot_k,r_k,r_star,ZS,WS,DS,c,gamma0,eta);
            %% Delete unused atoms
            K=nnz(n_dot_k);
            [kk,kki,kkj] = unique(ZS);
            ZS=kkj;
            DSZS=DSZS(:,kk);
            WSZS=WSZS(:,kk);
            n_dot_k=n_dot_k(kk);
            r_k=r_k(kk);
            %% Sample model parameters for the gamma-negative binomial process
            if strcmp(sampler,'Gamma_NB_partially_collapsed')
                sumlogpi = sum(log(max(1-p_i,realmin)));
                p_prime = -sumlogpi./(c-sumlogpi);
                c = randg(1 + gamma0)/(1+sum(r_k)+r_star);
                gamma0 = gamrnd(a0 + K,1/(b0 - log(max(1-p_prime,realmin))));
                L_k = CRT_sum_mex_matrix(sparse(DSZS),r_k')';
                r_k = gamrnd(L_k, 1./(-sumlogpi+ c));
                r_star = gamrnd(gamma0, 1./(-sumlogpi+ c));
                p_i = betarnd(a0 + sum(DSZS,2)',b0+sum(r_k)+r_star);   
                if iter>burnin && mod(iter,CollectionStep)==0
                    Phi = dirrnd([WSZS,zeros(V,K_star)] + eta);
                    Theta = bsxfun(@times,randg([DSZS' + r_k(:,ones(1,N)) ; r_star/K_star*ones(K_star,N)]), p_i);
                end
            end
            %% Sample model parameters for the hierachical Dirichlet process
            if strcmp(sampler,'HDP_DirectAssignment')
                if iter==1
                    alpha_concentration = 1;
                end
                L_k = CRT_sum_mex_matrix(sparse(DSZS),r_k')';
                for iii=1:1
                    w0 = betarnd(gamma0+1, sum(L_k));
                    pi0 = (a0+K-1)/((a0+K-1)+sum(L_k)*(b0-log(w0)));
                    %gamma0 = pi0*gamrnd(a0+K0,1/(b0-log(w0))) + (1-pi0)*gamrnd(a0+K0-1,1/(b0-log(w0)));
                    gamma0 = gamrnd(a0+K-(rand(1)>pi0),1/(b0-log(w0)));
                    r_tilde_k = dirrnd([L_k;gamma0]);
                    wj = betarnd(alpha_concentration+1, full(sum(DSZS,2)'));
                    sj = sum(rand(1,N)<sum(DSZS,2)'./(sum(DSZS,2)'+alpha_concentration));
                    alpha_concentration = gamrnd(sum(L_k)+a0 - sj,1/(b0-sum(log(wj))));
                end
                r_k = alpha_concentration*r_tilde_k(1:end-1,:);
                r_star = alpha_concentration*r_tilde_k(end);
                if iter>burnin && mod(iter,CollectionStep)==0
                    Phi = dirrnd([WSZS,zeros(V,K_star)] + eta);
                    Theta = dirrnd([DSZS' + r_k(:,ones(1,N)) ; r_star/K_star*ones(K_star,N)]);
                end
            end
            
        case 'Beta_NB_collapsed'
            %% Random scan, optional
            dex111=randperm(length(ZS));
            ZS=ZS(dex111); DS=DS(dex111); WS=WS(dex111);
            %% Fully collapsed inference, the beta process weights p_k are marginalized out
            [WSZS,DSZS,n_dot_k,ZS] = PFA_BNBP_collapsed(WSZS,DSZS,n_dot_k,ZS,WS,DS,r_i,c,gamma0,eta);
            
            %% Delete unsed atoms
            K=nnz(n_dot_k);
            [kk,kki,kkj] = unique(ZS);
            ZS=kkj;
            DSZS=DSZS(:,kk);
            WSZS=WSZS(:,kk);
            n_dot_k=n_dot_k(kk);
            %% Sample model parameters
            gamma0 = gamrnd(a0+K,1./(b0+psi(c+sum(r_i))-psi(c)));
            p_k = betarnd(n_dot_k,c+sum(r_i));
            p_star = logBeta_rnd(1,gamma0,c+sum(r_i));
            L_i = CRT_sum_mex_matrix(sparse(DSZS'),r_i);
            sumlogpk = sum(log(max(1-p_k,realmin)));
            r_i = gamrnd(a0 + L_i, 1./(-sumlogpk + p_star + b0));
            if 0
                c = sample_c(c,p_k,p_star,r_i,gamma0,DSZS',1,1)
            else
                %% Sample c using griddy-Gibbs
                ccc = 0.01:0.01:0.99;
                c = ccc./(1-ccc);
                % c = 0.001:0.001:1;
                temp = -gamma0*(psi(sum(r_i)+c)-psi(c)) +...
                    K*gammaln(c+sum(r_i)) - sum(gammaln(bsxfun(@plus,c,sum(r_i)+n_dot_k)),1) ;
                temp = exp(temp - max(temp));
                temp(isnan(temp))=0;
                cdf =cumsum(temp);
                c = c(sum(rand(1)*cdf(end)>cdf)+1);
            end
            
            if iter>burnin && mod(iter,CollectionStep)==0
                Phi = dirrnd([WSZS,zeros(V,K_star)] + eta);
                Theta = bsxfun(@times,randg([DSZS';zeros(K_star,N)] + r_i(ones(1,K+K_star),:)), [p_k;1-exp(-p_star/K_star)*ones(K_star,1)]);
            end
    end
    
    ave.K(iter) = nnz(sum(WSZS,1));
    ave.gamma0(iter) = gamma0;
    
    if iter>burnin && mod(iter,CollectionStep)==0 && percentage<1
        X1 = Mult_Sparse(Xmask,Phi,Theta);
        X1sum = sum(Theta,1);
        ave.PhiTheta = ave.PhiTheta + X1;
        ave.PhiThetaSum = ave.PhiThetaSum + X1sum;
        ave.Count = ave.Count+1;
        X1 = bsxfun(@rdivide, X1,X1sum);
        loglike(end+1)=sum(Xtest(Yflagtest).*log(X1(Yflagtest)))/sum(Xtest(:));
        loglikeTrain(end+1)=sum(Xtrain(Yflagtrain).*log(X1(Yflagtrain)))/sum(Xtrain(:));
        
        X1 = ave.PhiTheta/ave.Count;
        X1sum = ave.PhiThetaSum/ave.Count;
        X1= bsxfun(@rdivide, X1,X1sum);
        ave.loglike(end+1) = sum(Xtest(Yflagtest).*log(X1(Yflagtest)))/sum(Xtest(:));
        
        clear X1;
    end
    %
    %     if mod(iter,10)==0
    %         toc
    %         if iter>burnin
    %             disp(full([iter/100,loglikeTrain(end),loglike(end),ave.K(end),ave.loglike(end)]));
    %         else
    %             disp(full(iter/100));
    %         end
    %         tic;
    %     end
    
    if IsPlot && mod(iter,10)==0
        [temp, Thetadex] = sort(sum(WSZS,1),'descend');
        %[temp, Thetadex] = sort(n_dot_k,'descend');
        switch sampler
            case 'Beta_NB_collapsed'
                subplot(2,2,1);plot((p_k(Thetadex)),'.'); title('p_k')
                subplot(2,2,2);plot(r_i);title('r_i')
                subplot(2,2,3);plot(ave.K(1:iter)); title('K')
                subplot(2,2,4);plot(ave.gamma0(1:iter)); %title('gamma0')
                title(num2str([c,eta]));
            otherwise
                subplot(2,2,1);plot((r_k(Thetadex)),'.');title('r_k')
                subplot(2,2,2);plot(p_i);title('p_i')
                subplot(2,2,3);plot(ave.K(1:iter));title('K')
                subplot(2,2,4);plot(ave.gamma0(1:iter));
                title(num2str([c,eta]));
        end
        drawnow
    end
    fprintf(repmat('\b',1,numel(text)));
    text = sprintf('Train Iter: %d',iter); fprintf(text, iter);
end
output.r_k = r_k;
output.r_star = r_star;
output.p_i=p_i;
output.WSZS=WSZS;
output.ZS=ZS;
output.eta=eta;
if strcmp(sampler,'Beta_NB_collapsed')
    output.r_k = p_k;
    output.r_star = p_star;
end
ave.LogLikeTrain = loglikeTrain;
ave.LogLikeTest = loglike;
% LoglikeFinal = ave.loglike(end);



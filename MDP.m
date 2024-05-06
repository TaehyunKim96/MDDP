function [uMDP, uMDPu, uX, Xc, M, MuHat] = MDP(X,Y,option)
% Maximal Data Piling directions. X is n x p matrix of data ; Y is n x 1
% matrice of labels (1,...,K).
%   [uMDP, uMDPu, uX, Xc, M] = MDP(X,Y)
%   [uMDP, uMDPu, uX, Xc, M] = MDP(X,Y,option) with option given either by
%   'svd' (default) 'Xinv' ,'STinv'.
%
% returns:
% uX: the basis of n-dimensional subspace on which all data lie
% uMDPu: MDP on span(uX)
% uMDP : (= uX * uMDPu). MDP vectors in the p-space.
% Xc, M and MuHat on span(uX)

opt = 'svd';
if nargin > 2
    opt = option;
end

[n, p]=size(X);
K = max(Y);

% initial dimension reduction
[uX, ~, ~] = svd(X',0); 
Xu = X * uX; 
[~,q] = size(uX);

Xc = Xu;
MuHat = zeros(K,q);
nivec = zeros(K,1);
for iK = 1:K
    data = Xu(Y == iK,:);
    ni = sum(Y == iK);
    nivec(iK) = ni;
    MuHat(iK,:) = mean(data);
    Xc(Y == iK,:) = data - repmat(mean(data),ni,1);
end

M = (MuHat - repmat(mean(Xu),K,1))' * diag(sqrt(nivec)); 

switch opt
    case 'Xinv'
        [uMDPu,~]=svds(M - Xc' * pinv(Xc') * M,K-1);       
    case 'STinv' 
        SW = Xc' * Xc;
        SB = M * M';
        [uMDPu,~] = svds(pinv(SB + SW) * M,K-1); 
    otherwise  % 'svd'
        [Vxc, ~]= svds(Xc',n-K);
        [uMDPu,~]=svds(M - Vxc * Vxc' * M,K-1); 
end

uMDP = uX * uMDPu; 
function [uPRS, uPRSu, uX, alphahat] = PRS(X, Y, ngroup, ncomp)
% Projected Ridge Subspace. X is n x p matrix of data ; Y is n x 1
% matrice of labels (1,...,K).
%   [uPRS, uPRSu] = PRS(X, Y, ngroup, ncomp, alphavec)
%
% returns:
% alphahat: consistent estimator of optimal ridge parameter 
% uX: the basis of n-dimensional subspace on which all data lie
% uPRSu: PRS on span(uX)
% uPRS : (= uX * uMDPu). PRS vectors in the p-space.

[n, p]=size(X);

% initial dimension reduction
[uX, ~, ~] = svd(X',0); 
Xu = X * uX;
[~,q] = size(uX);

Xc = Xu;
MuHat = zeros(ngroup,q);
nivec = zeros(ngroup,1);
for iK = 1:ngroup
    data = Xu(Y == iK,:);
    ni = sum(Y == iK);
    nivec(iK) = ni;
    MuHat(iK,:) = mean(data);
    Xc(Y == iK,:) = data - repmat(mean(data),ni,1);
end

eigenvalues = (svd(Xc)).^2;
alphahat = -mean(eigenvalues((ncomp+1):(n-ngroup))) / p;

[u,~] = svds(Xc', n-ngroup, 'largest');
    
D = zeros(n, ngroup-1);
for i = 1:(ngroup-1)
    D(:,i) = MuHat(i,:)' - MuHat(ngroup,:)';
end

uPRSu =  p^(-1/2) * (u(:,1:ncomp) * diag(alphahat ./ ((eigenvalues(1:ncomp) / p) + alphahat)) * u(:,1:ncomp)' * D + ...
        (eye(n) - u(:,1:(n-ngroup)) * u(:,1:(n-ngroup))') * D);
[uPRSu,~] = svds(uPRSu, ngroup-1, 'largest');
uPRS = uX * uPRSu;        
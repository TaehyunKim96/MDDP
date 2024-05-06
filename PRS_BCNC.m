function err = PRS_BCNC(X, label_X, Y, label_Y, ngroup, ncomp, alphavec)
% Projected Ridge Subspace Bias-Corrected Nearest Centroid classification rule. 
% X is n x p matrix of training data; label_X is n x 1 matrix of labels
% (1,...,K)
% Y is test_n x p matrix of independent test data; label_Y is test_n x 1
% matrix of labels (1,...,K)
% ngroup = the number of groups, ncomp = the number of strong spikes
% 
% err = PRS_BCNC(X, label_X, Y, label_Y, ngroup, ncomp, alphavec)
% returns:
% err: Test error

[n,p] = size(X);

% initial dimension reduction
[uX, ~, ~] = svd(X',0); 
Xu = X * uX;
[~,q] = size(uX);

Xc = Xu;
MuHat = zeros(ngroup,q);
nivec = zeros(ngroup,1);
for iK = 1:ngroup
    data = Xu(label_X == iK,:);
    ni = sum(label_X == iK);
    nivec(iK) = ni;
    MuHat(iK,:) = mean(data);
    Xc(label_X == iK,:) = data - repmat(mean(data),ni,1);
end

eigenvalues = (svd(Xc)).^2;
if nargin < 7
    alphahat = -mean(eigenvalues((ncomp+1):(n-ngroup))) / p;
    alphavec = alphahat;
end

[u,~] = svds(Xc', n-ngroup, 'largest');
    
D = zeros(n, ngroup-1);
for i = 1:(ngroup-1)
    D(:,i) = MuHat(i,:)' - MuHat(ngroup,:)';
end

err = zeros(1, length(alphavec));

for alphaidx = 1:length(alphavec)
    alpha = alphavec(alphaidx);
    valphau =  p^(-1/2) * (u(:,1:ncomp) * diag(alpha ./ ((eigenvalues(1:ncomp) / p) + alpha)) * u(:,1:ncomp)' * D + ...
                (eye(n) - u(:,1:(n-ngroup)) * u(:,1:(n-ngroup))') * D);
    valpha = uX * valphau;
    meanX = uX * MuHat';

    gmat = zeros(ngroup-1, ngroup, length(label_Y));
    bmat = [diag(-alpha./nivec(1:ngroup-1)), ones(ngroup-1, 1)*alpha/nivec(ngroup)]; 
    for i = 1:length(label_Y)
        gmat(:,:,i) = p^(-1/2) * valpha' * (Y(i,:)' - meanX) + bmat;
    end
    [~, idx] = min(sum(gmat.^2, 1));
    err(alphaidx) = mean(label_Y ~= reshape(idx, length(label_Y), 1));
end
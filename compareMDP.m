%%
clear;clc;close all;
K = 3;
p = 5000;

rng(2023)
n = 20;
test_n = 500;

tau = 20; sigma = 1; m = 1;
u1 = ones(1, p);
mu1 = [ones(1, p/2), sqrt(3) * ones(1, p/2)];
mu2 = [sqrt(3) * ones(1, p/2), -ones(1, p/2)];
mu3 = zeros(1, p); 
Sigma = tau * eye(p) + sigma * u1.' * u1;

X1 = mvnrnd(mu1, Sigma, n);
X2 = mvnrnd(mu2, Sigma, n);
X3 = mvnrnd(mu3, Sigma, n);
X = [X1; X2; X3]; label_X = [ones(n, 1); 2*ones(n, 1); 3*ones(n, 1)];

%% Pairwise MDP vectors
uMDPv = [];
for i = 1:K
    for j = (i+1):K
        Xij = X(label_X == i | label_X == j, :);
        label_Xij = [ones(n, 1); 2*ones(n,1)];
        uMDPij = MDP(Xij, label_Xij);
        uMDPv = [uMDPv, uMDPij];
    end
end

[uMDPv, ~] = svds(uMDPv, 3);
projMDPv = uMDPv' * (X - mean(X))';
col_test = [repmat([1 0 0], n, 1); repmat([0 1 0], n, 1); repmat([0 0 1], n, 1)];
s = scatter3(projMDPv(1,:), projMDPv(2,:), projMDPv(3,:), label_X, col_test);
s.SizeData = 30;
s.Marker = 'o';
hold on 
set(gcf, 'position', [0, 0, 450, 450])
xlim([-100, 100]);
ylim([-100, 100]);
zlim([-100, 100]);
view([-75, 30])
print('-dpng', ['images/MDP_vectors.png'])

%% MDP subspace
uMDPs = MDP(X, label_X);
projMDPs = uMDPs' * (X - mean(X))';
gscatter(projMDPs(1,:), projMDPs(2,:), label_X, 'rgb', 'ooo', 6, 'off')
set(gcf, 'position', [0, 0, 450, 450])
xlim([-100, 100]);
ylim([-100, 100]);
print('-dpng', ['images/MDP_subspace.png'])
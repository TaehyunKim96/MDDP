%% 
clear;clc;close all;
K = 3;
vp = [200, 800, 2000, 5000];
alphavec = -45:15;

err_mean_matrix = zeros(4, length(alphavec));
err_se_matrix = zeros(4, length(alphavec));
rng(2023);

%%
for ip = 1:4
    %% Setting
    p = vp(ip); 
    n1 = 20; n2 = 20; n3 = 20;
    test_n1 = 500; test_n2 = 500; test_n3 = 500;

    tau = 20; sigma = 1; m = 1;
    u1 = ones(1, p);
    mu1 = [ones(1, p/2), sqrt(3) * ones(1, p/2)];
    mu2 = [sqrt(3) * ones(1, p/2), -ones(1, p/2)];
    mu3 = zeros(1, p); 
    Sigma = tau * eye(p) + sigma * u1.' * u1;

    nrep = 100;
    err_matrix = zeros(nrep, length(alphavec));

    %%
    for irep = 1:nrep
        %% Generating data
        X1 = mvnrnd(mu1, Sigma, n1);
        X2 = mvnrnd(mu2, Sigma, n2);
        X3 = mvnrnd(mu3, Sigma, n3);

        Y1 = mvnrnd(mu1, Sigma, test_n1);
        Y2 = mvnrnd(mu2, Sigma, test_n2);
        Y3 = mvnrnd(mu3, Sigma, test_n3);

        X = [X1; X2; X3]; label_X = [ones(n1, 1); 2*ones(n2, 1); 3*ones(n3, 1)];
        Y = [Y1; Y2; Y3]; label_Y = [ones(test_n1, 1); 2*ones(test_n2, 1); 3*ones(test_n3, 1)];
        meanX = [mean(X1); mean(X2); mean(X3)];

        %% PRS-BCNC (Projected Ridge Subspace Bias-Corrected Nearest Centroid)
        err_matrix(irep,:) = PRS_BCNC(X, label_X, Y, label_Y, K, m, alphavec);
        disp([p, irep])
    end  
    err_mean_matrix(ip,:) = mean(err_matrix, 1);
    err_se_matrix(ip,:) = std(err_matrix, 1);
end

%%
plot(alphavec, err_mean_matrix(1,:), ':', Color = 'k');
hold on
plot(alphavec, err_mean_matrix(2,:), '-.', Color = 'k');
hold on
plot(alphavec, err_mean_matrix(3,:), '--', Color = 'k');
hold on
plot(alphavec, err_mean_matrix(4,:), '-', Color = 'k');
hold on
xline(-tau, ':');
xline(0, ':');
xlim([min(alphavec)-1, max(alphavec)+1])
ylim([0, 0.7])
xlabel('$\alpha$','Interpreter','latex')
ylabel('Test error')
lgd = legend('p = 200', 'p = 800', 'p = 2000', 'p = 5000', 'Location', 'northeast');
lgd.FontSize = 10;
set(gcf, 'position', [0, 0, 450, 450])
print('-dpng', ['images/simulation_alphaplot.png'])
hold off

%% 
clear;clc;close all;
K = 3;
vp = [200, 800, 2000, 5000, 10000];

err_mean_matrix = zeros(5, 4);
err_se_matrix = zeros(5, 4);
rng(2023)

%%
for ip = 1:5
    %% Setting (Heterogeneous Spiked Covariance Models with equal tail eigenvalues)
    p = vp(ip); 
    n1 = 20; n2 = 20; n3 = 20; 
    test_n1 = 500; test_n2 = 500; test_n3 = 500;

    tau = 20; sigma1 = 3; sigma2 = 2; sigma3 = 1; m = 3;
    u11 = ones(1, p);
    u21 = randn(1,p); u21 = sqrt(p) * u21./sqrt(u21*u21');
    u31 = randn(1,p); u31 = sqrt(p) * u31./sqrt(u31*u31');
    
    mu1 = [ones(1, p/2), sqrt(3) * ones(1, p/2)];
    mu2 = [sqrt(3) * ones(1, p/2), -ones(1, p/2)];
    mu3 = zeros(1, p); 
    
    Sigma1 = tau * eye(p) + sigma1 * u11.' * u11;
    Sigma2 = tau * eye(p) + sigma2 * u21.' * u21;
    Sigma3 = tau * eye(p) + sigma3 * u31.' * u31;

    nrep = 100;
    err_matrix = zeros(nrep, 4);

    %%
    for irep = 1:nrep
        %% Generating data
        X1 = mvnrnd(mu1, Sigma1, n1);
        X2 = mvnrnd(mu2, Sigma2, n2);
        X3 = mvnrnd(mu3, Sigma3, n3);

        Y1 = mvnrnd(mu1, Sigma1, test_n1);
        Y2 = mvnrnd(mu2, Sigma2, test_n2);
        Y3 = mvnrnd(mu3, Sigma3, test_n3);
        
        X = [X1; X2; X3]; label_X = [ones(n1, 1); 2*ones(n2, 1); 3*ones(n3, 1)];
        Y = [Y1; Y2; Y3]; label_Y = [ones(test_n1, 1); 2*ones(test_n2, 1); 3*ones(test_n3, 1)];
        meanX = [mean(X1); mean(X2); mean(X3)];

        %% 1. NC (Nearest Centroid)
        label_pred_NC = knnsearch(meanX, Y);
        err_matrix(irep,1) = mean(label_pred_NC ~= label_Y);

        %% 2. PRS-NC (Projected Ridge Subspace Nearest Centroid)
        uPRS = PRS(X, label_X, K, m);
        projPRD_Y = uPRS' * Y';
        projPRD_meanX = uPRS' * meanX';

        label_pred_PRD_NC = knnsearch(projPRD_meanX', projPRD_Y');
        err_matrix(irep,2) = mean(label_pred_PRD_NC ~= label_Y);

        %% 3. One-vs-Rest PRD (Projected Ridge Direction)
        uPRD = zeros(p, K);
        label_newX1 = [ones(n1, 1); 2*ones(n2,1); 2*ones(n3,1)];
        uPRD(:,1) = PRS(X, label_newX1, 2, m);
        label_newX2 = [2*ones(n1, 1); ones(n2,1); 2*ones(n3,1)];
        uPRD(:,2) = PRS(X, label_newX2, 2, m);
        label_newX3 = [2*ones(n1, 1); 2*ones(n2,1); ones(n3,1)];
        uPRD(:,3) = PRS(X, label_newX3, 2, m);
        [~,label_pred_PRDs] = max(uPRD' * (Y - mean(X))'); 
        err_matrix(irep,3) = mean(label_pred_PRDs' ~= label_Y);

        %% 4. PRS-BCNC (Projected Ridge Subspace Bias-Corrected Nearest Centroid)
        err_matrix(irep,4) =  PRS_BCNC(X, label_X, Y, label_Y, K, m);
        disp([p, irep])
    end  
    err_mean_matrix(ip, :, :) = mean(err_matrix, 1);
    err_se_matrix(ip, :, :) = std(err_matrix, 1);
end

%%
err_mean_NC = err_mean_matrix(:,1);
err_mean_PRS_NC = err_mean_matrix(:,2);
err_mean_OVR_PRD = err_mean_matrix(:,3);
err_mean_PRS_BCNC = err_mean_matrix(:,4);

plot(vp, err_mean_NC, '-or');
hold on
plot(vp, err_mean_PRS_NC, '-+g');
hold on
plot(vp, err_mean_OVR_PRD, '-*b');
hold on
plot(vp, err_mean_PRS_BCNC, '-xm');
ylim([0, 0.6]);
xticks(vp);
xline(vp, ':');
xlabel('$p$','Interpreter','latex')
ylabel('Test error')

lgd = legend('NC', 'PRS-NC', 'PRD-OVR', 'PRS-BCNC', 'Location', 'northeast');
lgd.FontSize = 10;
set(gcf, 'position', [0, 0, 450, 450])
print('-dpng', ['images/simulation_pplot_v2.png'])
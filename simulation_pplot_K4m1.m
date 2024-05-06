%% 
clear;clc;close all;
K = 4;
vp = [200, 800, 2000, 5000, 10000];

err_mean_matrix = zeros(5, 4);
err_se_matrix = zeros(5, 4);
rng(2023)

%%
for ip = 1:5
    %% Setting
    p = vp(ip); 
    n1 = 20; n2 = 20; n3 = 20; n4 = 20;
    test_n1 = 500; test_n2 = 500; test_n3 = 500; test_n4 = 500;

    tau = 20; sigma = 1; m = 1;
    u1 = ones(1, p);
    mu1 = [ones(1, p/2), sqrt(3) * ones(1, p/2)];
    mu2 = [sqrt(3) * ones(1, p/2), -ones(1, p/2)];
    mu3 = [sqrt(3) * ones(1, p/2), ones(1, p/2)];
    mu4 = zeros(1, p); 
    Sigma = tau * eye(p) + sigma * u1.' * u1;

    nrep = 100;
    err_matrix = zeros(nrep, 4);

    %%
    for irep = 1:nrep
        %% Generating data
        X1 = mvnrnd(mu1, Sigma, n1);
        X2 = mvnrnd(mu2, Sigma, n2);
        X3 = mvnrnd(mu3, Sigma, n3);
        X4 = mvnrnd(mu4, Sigma, n4);

        Y1 = mvnrnd(mu1, Sigma, test_n1);
        Y2 = mvnrnd(mu2, Sigma, test_n2);
        Y3 = mvnrnd(mu3, Sigma, test_n3);
        Y4 = mvnrnd(mu4, Sigma, test_n4);

        X = [X1; X2; X3; X4]; label_X = [ones(n1, 1); 2*ones(n2, 1); 3*ones(n3, 1); 4*ones(n4, 1)];
        Y = [Y1; Y2; Y3; Y4]; label_Y = [ones(test_n1, 1); 2*ones(test_n2, 1); 3*ones(test_n3, 1); 4*ones(test_n4, 1)];
        meanX = [mean(X1); mean(X2); mean(X3); mean(X4)];

        %% 1. NC (Nearest Centroid)
        label_pred_NC = knnsearch(meanX, Y);
        err_matrix(irep,1) = mean(label_pred_NC ~= label_Y);

        %% 2. MDP-NC (Maximal Data Piling Nearest Centroid)
        uMDP = MDP(X, label_X);
        projMDP_Y = uMDP' * Y';
        projMDP_meanX = uMDP' * meanX';

        label_pred_MDP_NC = knnsearch(projMDP_meanX', projMDP_Y');
        err_matrix(irep,2) = mean(label_pred_MDP_NC ~= label_Y);

        %% 3. PRS-NC (Projected Ridge Subspace Nearest Centroid)
        uPRS = PRS(X, label_X, K, m);
        projPRD_Y = uPRS' * Y';
        projPRD_meanX = uPRS' * meanX';

        label_pred_PRD_NC = knnsearch(projPRD_meanX', projPRD_Y');
        err_matrix(irep,3) = mean(label_pred_PRD_NC ~= label_Y);

        %% 4. PRS-BCNC (Projected Ridge Subspace Bias-Corrected Nearest Centroid)
        err_matrix(irep,4) =  PRS_BCNC(X, label_X, Y, label_Y, K, m);
        disp([p, irep])
    end  
    err_mean_matrix(ip, :, :) = mean(err_matrix, 1);
    err_se_matrix(ip, :, :) = std(err_matrix, 1);
end

%%
err_mean_matrix(:,1) = [0.3985 0.3164 0.2819 0.2708 0.2579];
err_mean_matrix(:,2) = [0.4714 0.2786 0.2040 0.1658 0.1531];
err_mean_matrix(:,3) = [0.5634 0.3933 0.3182 0.2515 0.2106];
err_mean_matrix(:,4) = [0.4978 0.2411 0.1062 0.0224 0.0020];

%%
err_mean_NC = err_mean_matrix(:,1);
err_mean_MDP_NC = err_mean_matrix(:,2);
err_mean_PRD_NC = err_mean_matrix(:,3);
err_mean_PRS_BCNC = err_mean_matrix(:,4);

plot(vp, err_mean_NC, '-or');
hold on
plot(vp, err_mean_MDP_NC, '-+g');
hold on
plot(vp, err_mean_PRD_NC, '-*b');
hold on
plot(vp, err_mean_PRS_BCNC, '-xm');
ylim([0, 0.4]);
xticks(vp);
xline(vp, ':');
xlabel('$p$','Interpreter','latex')
ylabel('Test error')

lgd = legend('NC', 'MDP-NC', 'PRS-NC', 'PRS-BCNC', 'Location', 'northeast');
lgd.FontSize = 10;
set(gcf, 'position', [0, 0, 450, 450])
print('-dpng', ['images/simulation_pplot_revision_K4m1.png'])
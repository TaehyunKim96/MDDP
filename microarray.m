clear;clc;close all;
rng(2024)

item = 1;
if item == 1
    % MLL dataset
    MLL = readmatrix("MLL.csv");
    label = MLL(:,1);
    data =  MLL(:,2:end);
elseif item == 2
    % GSE2685 dataset
    GSE2685 = readmatrix("GSE2685.csv");
    label = GSE2685(:,end);
    data =  GSE2685(:,1:end-1);
end

[n,p] = size(data);
K = max(label);
m = 1;
err_matrix = zeros(n, 5);

%%
for testid = 1:length(label)
    trainid = [1:(testid-1), (testid+1):n];
    X = data(trainid,:); label_X = label(trainid); train_n = n-1; 
    Y = data(testid,:); label_Y = label(testid); test_n = 1;   
    meanX = [];
    for i = 1:K
        meanX = [meanX; mean(X(label_X == i,:),1)];
    end
    
    %% 1. NC (Nearest Centroid)
    label_pred_NC = knnsearch(meanX, Y);
    err_matrix(testid,1) = mean(label_pred_NC ~= label_Y);

    %% 2. MDP-NC (Maximal Data Piling Nearest Centroid)
    uMDP = MDP(X, label_X);
    projMDP_Y = uMDP' * Y';
    projMDP_meanX = uMDP' * meanX';

    label_pred_MDP_NC = knnsearch(projMDP_meanX', projMDP_Y');
    err_matrix(testid,2) = mean(label_pred_MDP_NC ~= label_Y);

    %% 3. PRS-NC (Projected Ridge Subspace Nearest Centroid)
    uPRS = PRS(X, label_X, K, m);
    projPRD_Y = uPRS' * Y';
    projPRD_meanX = uPRS' * meanX';

    label_pred_PRD_NC = knnsearch(projPRD_meanX', projPRD_Y');
    err_matrix(testid,3) = mean(label_pred_PRD_NC ~= label_Y);

    %% 4. One-vs-Rest PRD (Projected Ridge Direction)
    uPRD = zeros(p, K);
    for i = 1:K
        label_newX = (label_X ~= i) + 1;
        uPRD(:,i) = PRS(X, label_newX, 2, m);
    end
    [~,label_pred_PRDs] = max(uPRD' * (Y - mean(X))'); 
    err_matrix(testid,4) = mean(label_pred_PRDs' ~= label_Y);

    %% 5. PRS-BCNC (Projected Ridge Subspace Bias-Corrected Nearest Centroid)
    err_matrix(testid,5) = PRS_BCNC(X, label_X, Y, label_Y, K, m);

    disp([testid])
end
%%
err_mean_matrix = mean(err_matrix, 1);
err_mean_matrix
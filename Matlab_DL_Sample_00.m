%% Housing Price Prediction: End-to-End (Simulation -> DL -> Diagnostics)
% Purpose: Predict simulated housing prices with a neural network (MLP), plus
% diagnostics, OLS baseline, permutation importance, and model saving.
%
% Requires:
%   - Deep Learning Toolbox (for trainNetwork, layers)
% Optional:
%   - Statistics and Machine Learning Toolbox (for fitlm; script has a fallback)

clear; clc; close all;
rng(42);

%% 1) Simulate dataset (n = 500)
n = 500;
Size_sqft        = round(max(400, 1500 + 500*randn(n,1)));
Bedrooms         = randi([1,5], n, 1);
Age_years        = randi([0,50], n, 1);
Location_Quality = randi([1,10], n, 1);

% True price function + noise
Price = 50*Size_sqft + 10000*Bedrooms - 500*Age_years + ...
        20000*Location_Quality + 20000*randn(n,1);

X = [Size_sqft Bedrooms Age_years Location_Quality];
Y = Price;

featureNames = {'Size_sqft','Bedrooms','Age_years','Location_Quality'};

%% 2) Split data (60/20/20)
idx = randperm(n);
nTrain = round(0.6*n); nVal = round(0.2*n); nTest = n - nTrain - nVal;

idxTrain = idx(1:nTrain);
idxVal   = idx(nTrain+1 : nTrain+nVal);
idxTest  = idx(nTrain+nVal+1 : end);

XTrain = X(idxTrain,:);  YTrain = Y(idxTrain);
XVal   = X(idxVal,:);    YVal   = Y(idxVal);
XTest  = X(idxTest,:);   YTest  = Y(idxTest);

%% 3) SCALE THE TARGET (critical for stable learning)
muY    = mean(YTrain);
sigmaY = std(YTrain);
YTrainScaled = (YTrain - muY) / sigmaY;
YValScaled   = (YVal   - muY) / sigmaY;

%% 4) Define neural net (compact MLP; inputs auto z-scored)
layers = [
    featureInputLayer(size(X,2), "Normalization","zscore")
    fullyConnectedLayer(128)
    reluLayer
    dropoutLayer(0.2)
    fullyConnectedLayer(64)
    reluLayer
    fullyConnectedLayer(32)
    reluLayer
    fullyConnectedLayer(1)   % predicts scaled target
    regressionLayer
];

opts = trainingOptions('adam', ...
    'MaxEpochs',120, ...
    'MiniBatchSize',64, ...
    'InitialLearnRate',1e-3, ...
    'Shuffle','every-epoch', ...
    'ValidationData',{XVal, YValScaled}, ...
    'ValidationFrequency',max(1,floor(nTrain/64)), ...
    'Verbose',false, ...
    'Plots','training-progress', ...
    'ExecutionEnvironment','auto');

%% 5) Train (on scaled targets)
net = trainNetwork(XTrain, YTrainScaled, layers, opts);

%% 6) Predict on test (unscale back)
YPredScaled = predict(net, XTest);
YPred = YPredScaled * sigmaY + muY;

%% 7) Metrics
rmse = sqrt(mean((YPred - YTest).^2));
R2   = 1 - sum((YTest - YPred).^2) / sum((YTest - mean(YTest)).^2);
disp("DL  Test RMSE: " + round(rmse,2));
disp("DL  Test R^2 : " + round(R2,4));

%% 8) Plot: Predicted vs Actual
figure;
scatter(YTest, YPred, 18, 'filled'); grid on; hold on;
xlabel('Actual Price'); ylabel('Predicted Price');
title('Housing Price Prediction (Test Set)');
lims = [min([YTest; YPred]) max([YTest; YPred])];
plot(lims, lims, 'k--', 'LineWidth',1.2); xlim(lims); ylim(lims);

%% 9) Residual diagnostics
res = YTest - YPred;

% Histogram
figure; histogram(res, 30); grid on;
title('Residuals Histogram'); xlabel('Residual'); ylabel('Count');

% Residuals vs Fitted
figure; scatter(YPred, res, 14, 'filled'); grid on; yline(0,'k--');
xlabel('Predicted Price'); ylabel('Residual');
title('Residuals vs Fitted');

% Q-Q plot (toolbox-free version)
res_sorted = sort(res);
nq = numel(res_sorted);
theoretical = norminv(((1:nq)' - 0.5)/nq, 0, std(res)); % zero-mean, sd=std(res)
figure; scatter(theoretical, res_sorted, 14, 'filled'); grid on; hold on;
plot([theoretical(1) theoretical(end)], [theoretical(1) theoretical(end)], 'k--', 'LineWidth',1.2);
xlabel('Normal Quantiles'); ylabel('Residual Quantiles');
title('Q-Q Plot of Residuals (manual)');

%% 10) Baseline benchmark: OLS (with fallback if fitlm unavailable)
useFitlm = exist('fitlm','file')==2;
if useFitlm
    tblTrain = array2table(XTrain, 'VariableNames',featureNames);
    tblTrain.Price = YTrain;
    mdl = fitlm(tblTrain, 'Price ~ Size_sqft + Bedrooms + Age_years + Location_Quality');

    tblTest = array2table(XTest, 'VariableNames',featureNames);
    yhat_lin = predict(mdl, tblTest);
else
    % Toolbox-free OLS: add intercept; beta = (X'X)^(-1)X'y
    Xtr = [ones(size(XTrain,1),1) XTrain];
    beta = Xtr \ YTrain;
    Xts = [ones(size(XTest,1),1) XTest];
    yhat_lin = Xts * beta;
end

rmse_lin = sqrt(mean((yhat_lin - YTest).^2));
R2_lin   = 1 - sum((YTest - yhat_lin).^2) / sum((YTest - mean(YTest)).^2);
disp("OLS Test RMSE: " + round(rmse_lin,2));
disp("OLS Test R^2 : " + round(R2_lin,4));

%% 11) Permutation feature importance (for the DL model)
base_rmse = sqrt(mean((YPred - YTest).^2));
imp = zeros(1, numel(featureNames));
XTest_perm = XTest;

for j = 1:numel(featureNames)
    col = XTest(:,j);
    XTest_perm(:,j) = col(randperm(numel(col)));   % permute one feature
    yperm_scaled = predict(net, XTest_perm);
    yperm = yperm_scaled * sigmaY + muY;           % unscale
    rmse_perm = sqrt(mean((yperm - YTest).^2));
    imp(j) = rmse_perm - base_rmse;                % RMSE increase
    XTest_perm(:,j) = XTest(:,j);                  % restore
end

[imp_sorted, idxImp] = sort(imp, 'descend');
disp('Permutation importance (RMSE increase):');
for k = 1:numel(idxImp)
    fprintf('%-18s  %+8.2f\n', featureNames{idxImp(k)}, imp_sorted(k));
end

figure; bar(categorical(featureNames(idxImp)), imp_sorted); grid on;
ylabel('RMSE Increase'); title('Permutation Feature Importance (DL MLP)');

%% 12) Save trained network (with target scaling params)
% save housing_mlp.mat net muY sigmaY featureNames;


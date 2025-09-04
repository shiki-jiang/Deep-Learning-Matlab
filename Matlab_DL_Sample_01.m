%% Digit Classification with a Small CNN
% Purpose: End-to-end example—load images, define CNN, train, evaluate.
% Requires: Deep Learning Toolbox
% Optional: Image Processing Toolbox (preprocessing), Parallel Computing Toolbox (GPU)

%% 1) Data: built-in digit image set (10 classes, 28x28)
digitDatasetPath = fullfile(matlabroot,'toolbox','nnet','nndemos','nndatasets','DigitDataset');
imds = imageDatastore(digitDatasetPath, ...
    'IncludeSubfolders',true, 'LabelSource','foldernames');

% Split train/validation
[imdsTrain, imdsVal] = splitEachLabel(imds, 0.8, 'randomized');

%% 2) (Optional) Preprocessing/augmentation
inputSize = [28 28];
augmenter = imageDataAugmenter( ...
    'RandRotation',[-10 10], ...
    'RandXTranslation',[-2 2], ...
    'RandYTranslation',[-2 2]);

augTrain = augmentedImageDatastore(inputSize, imdsTrain, 'ColorPreprocessing','gray2gray', ...
    'DataAugmentation',augmenter);
augVal   = augmentedImageDatastore(inputSize, imdsVal,   'ColorPreprocessing','gray2gray');

%% 3) Model: a compact CNN
layers = [
    imageInputLayer([28 28 1])

    convolution2dLayer(3,16,'Padding','same')
    batchNormalizationLayer
    reluLayer
    maxPooling2dLayer(2,'Stride',2)

    convolution2dLayer(3,32,'Padding','same')
    batchNormalizationLayer
    reluLayer
    maxPooling2dLayer(2,'Stride',2)

    convolution2dLayer(3,64,'Padding','same')
    batchNormalizationLayer
    reluLayer

    fullyConnectedLayer(10)     % 10 digit classes
    softmaxLayer
    classificationLayer
];

%% 4) Training options
opts = trainingOptions('adam', ...
    'MaxEpochs',8, ...
    'MiniBatchSize',128, ...
    'Shuffle','every-epoch', ...
    'ValidationData',augVal, ...
    'ValidationFrequency',floor(numel(imdsTrain.Files)/128), ...
    'Verbose',false, ...
    'Plots','training-progress');   % shows live training curve

%% 5) Train
net = trainNetwork(augTrain, layers, opts);

%% 6) Evaluate
YPred = classify(net, augVal);
acc = mean(YPred == imdsVal.Labels);
disp("Validation accuracy: " + string(round(acc*100,2)) + "%");

% Confusion chart (requires Deep Learning Toolbox)
figure; confusionchart(imdsVal.Labels, YPred);
title('Confusion Matrix: Validation Set');
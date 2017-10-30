function [trained_thetas J] = trainNeuralNet(X, y, lambda, layer_dims, maxIters)
% Train the Neural Network, of size specified by layer_dims, on X and y using regularization
% parameter lambda. The network is trained using fmincg with at most maxIters epochs.

thetas = randInitThetas(layer_dims);

options = optimset('MaxIter', maxIters);
costFunc = @(p) CostGrad(X, y, lambda, p, layer_dims);
[trained_thetas J] = fmincg(costFunc, thetas, options);

end
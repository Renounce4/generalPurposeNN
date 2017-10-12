function execute(lambda, layer_dims)

data = load('-ascii','testData.txt');

X = data(:,2:end);
y = data(:,1);

r = randperm(size(X,1));
X = X(r,:);
y = y(r,:);

thetas = randInitThetas(layer_dims);

options = optimset('MaxIter', 1000000);
costFunc = @(p) CostGrad(X, y, lambda, p, layer_dims);

[trained_thetas J] = fmincg(costFunc, thetas, options);

pred = predict(X,trained_thetas,layer_dims);

fprintf('\nTraining Set Accuracy: %f\n', mean(double(pred == y)) * 100);

end
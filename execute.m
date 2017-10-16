function execute(layer_dims, maxIters, lambda)
% USAGE: function execute(layer_dims, maxIters, lambda)

% Load data, normalize it, randomize it and split it into
% a training set, cross validation set and test set.
data = load('-ascii','testData.txt');

X = data(:,2:end);
y = data(:,1);

[X_norm, mu, sig] = featureNormalize(X);
[X_train, y_train, X_cval, y_cval, X_test, y_test] = randSplitData(X_norm, y, .2, .2);

[m n] = size(X_train);


if ~exist('lambda', 'var') || isempty(lambda)
	% Use Cross Validation Curves to optimize lambda
	fprintf('\nRunning Cross Validation over lambda\n');

	lambda_vec = [0 3.^[-7:2]];
	cv_error_train = zeros(length(lambda_vec),1);
	cv_error_cval = zeros(length(lambda_vec),1);

	for i = 1:length(lambda_vec)
		trained_thetas = trainNeuralNet(X_train, y_train, lambda_vec(i), layer_dims, maxIters);
		cv_error_train(i) = CostGrad(X_train, y_train, 0, trained_thetas, layer_dims);
		cv_error_cval(i) = CostGrad(X_cval, y_cval, 0, trained_thetas, layer_dims);
	end

	figure(1);
	plot(lambda_vec, cv_error_train, lambda_vec, cv_error_cval);
	title(sprintf('Neural Network Cross Validation Curves (lambda)'));
	xlabel('lambda');
	ylabel('Error');
	axis([0 max(lambda_vec) 0 n]);
	legend('Train', 'Cross Validation');

	[dummy lambda_idx] = min(cv_error_cval);
	lambda = lambda_vec(lambda_idx);
end


% Train the network and plot learning curves
fprintf('\nCreating Learning Curves with lambda = %f\n', lambda);

error_train = zeros(m,1);
error_cval = zeros(m,1);

for i = 1:m
	trained_thetas = trainNeuralNet(X_train(1:i,:), y_train(1:i), lambda, layer_dims, maxIters);
	error_train(i) = CostGrad(X_train(1:i,:), y_train(1:i), 0, trained_thetas, layer_dims);
	error_cval(i) = CostGrad(X_cval, y_cval, 0, trained_thetas, layer_dims);
end

figure(2);
plot(1:m, error_train, 1:m, error_cval);
title(sprintf('Neural Network Learning Curves (lambda = %f', lambda));
xlabel('Number of training examples');
ylabel('Error');
axis([0 m 0 n]);
legend('Train', 'Cross Validation');

pred = predict(X_train, trained_thetas, layer_dims);
fprintf('\nTraining Set Accuracy: %f\n', mean(double(pred == y_train)) * 100);


% Final test set accuracy
pred = predict(X_test, trained_thetas, layer_dims);
fprintf('Test Set Accuracy: %f\n\n', mean(double(pred == y_test)) * 100);

end
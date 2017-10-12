function p = predict(X, thetas, layer_dims)

[m n] = size(X);				% Let: m = number of training examples & n = number of features
L = length(layer_dims);			% Let: L = number of layers
K = layer_dims(L);				% Let: K = number of classes (output neurons)

% 	X - (m x n) matrix of all training examples
%	thetas - (? x 1) vector of all theta values to be used in the Network
%	layer_dims - (L x 1) matrix defining the size of each layer and the number of layers


% ========== Calculate Activations ========== %

strt = 1;
a = [ones(m,1), X]';
reg_thetas = [];

for l = 1:(L-1)

	s1 = layer_dims(l+1);
	s2 = layer_dims(l)+1;
	theta = reshape(thetas(strt:strt + (s1*s2) - 1), s1, s2);
	reg_thetas = [reg_thetas; [zeros(s1,1), theta(:,2:end)](:)];
	a = [a; [ones(1,m); sigmoid(theta * a([sum(layer_dims(1:l-1)+1) + 1:sum(layer_dims(1:l)+1)],:))]];
	strt += s1*s2;

end


% ========== Extract Hypothesis ========== %

h = a([sum(layer_dims(1:L-1) + 1) + 2:end],:)';
[dummy p] = max(h, [], 2);

end
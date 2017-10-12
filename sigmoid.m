function g = sigmoid(z)
%SIGMOID Comput sigmoid function
%	z - matrix of any size

g = 1.0 ./ (1.0 + exp(-z));
end
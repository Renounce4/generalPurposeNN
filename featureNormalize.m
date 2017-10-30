function [X_norm, mu, sigma] = featureNormalize(X, mu, sigma)
% USAGES: function [X_norm, mu, sigma] = featureNormalize(X)
%         function [X_norm, mu, sigma] = featureNormalize(X, mu)
%         function [X_norm, mu, sigma] = featureNormalize(X, sigma)
%         function [X_norm, mu, sigma] = featureNormalize(X, mu, sigma)
%
%	Retuns a normalized version of X defined as (X - mu)/sigma.
%	If mu or sigma are supplied, it will use those instead of calculating them over X.
%
%	To perform min-max normalization use: featureNormalize(X, min(X), max(X)-min(X))
%
%	Parameters:
%		X - num_examples x num_features matrix.
%		mu - scalar or 1 x num_features matrix to be subtracted from X.
%		sigma - scalar or 1 x num_features matrix (X - mu) is devided over.
%
%	Returns:
%		X_norm - normalized version of X.
%		mu - scalar or 1 x num_features matrix subtracted from X.
%		sigma - scalar or 1 x num_features matrix (X - mu) is devided over.

if ~exist('mu', 'var') || isempty(mu)
	mu = mean(X);
end

X_norm = bsxfun(@minus, X, mu);

if ~exist('sigma', 'var') || isempty(sigma)
	sigma = std(X_norm);
end

X_norm = bsxfun(@rdivide, X_norm, sigma);

end
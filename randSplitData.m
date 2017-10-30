function [X_batch, y_batch, X_cut, y_cut] = randSplitData(X, y, cut_proportion)
% USAGES: function [X_batch, y_batch, X_cut, y_cut] = randSplitData(X, y, cut_proportion)
%         function [X_batch, y_batch, X_cut, y_cut] = randSplitData(X, y)
%
%	Randomizes the order of the training examples and then splits them so
%	X_cut and y_cut contain cut_proportion of the data and the rest is in
%	X_batch and y_batch. If cut_proportion is not specified, assumes 50-50 split.
%
%	Parameters:
%		X - num_examples x num_features matrix.
%		y - num_examples x [num_labels | 1] matrix.
%		cut_proportion - scalar [0, 1] proportion of X and y to cut to X_cut and y_cut.
%			Batch wins with odd num_examples (i.e. 0.5 cut of 5 examples gives 3 to batch).
%
%	Returns:
%		X_batch - randomized selection of 1-cut_proportion examples from X.
%		y_batch - randomized selection of 1-cut_proportion examples from y.
%		X_cut - randomized selection of cut_proportion examples from X.
%		y_cut - randomized selection of cut_proportion examples from y.

if ~exist('cut_proportion', 'var') || isempty(cut_proportion)
	cut_proportion = 0.5;
end

% Ensure cut_proportion is in range
cut_proportion = max(cut_proportion, 0);
cut_proportion = min(cut_proportion, 1);

m = size(X,1);

% ===== Randomize ===== %
r = randperm(m);
X_rand = X(r,:);
y_rand = y(r,:);


% ===== Split ===== %
m_cut = floor(m * cut_proportion);
m_batch = m - m_cut;

X_cut = X_rand(1:m_cut,:);
y_cut = y_rand(1:m_cut,:);

X_batch = X_rand(m_cut+1:m_cut+m_batch,:);
y_batch = y_rand(m_cut+1:m_cut+m_batch,:);

end
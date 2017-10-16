function idxs = underSample(y, K)
% USAGE: function idxs = underSample(y, K)
% Returns a (K*m) vector idxs of the indexes within y to use for each class in K.
% m is defined as the number of examples of the minority (under-represented) class.
% Indexes are randomized for each class, and returned in increasing order by class.
% Warning: function will skip over classes with no examples provided in y.

labels = y(:)==(ones(length(y),1) * [1:K]);

m = min(sum(labels));

idxs = [];

for k = 1:K
	if ~isempty(idx = find(labels(:,k)))
		idxs = [idxs;idx(randperm(length(idx),m))];
	end
end

end
function idxs = overSample(y, K)
% USAGE: function idxs = overSample(y, K)
% Returns a (K*m) vector idxs of the indexes within y to use for each class in K.
% m is defined as the number of examples of the majority (over-represented) class.
% Under-represented examples are randomly duplicated, and returned in increasing order by class.
% Warning: function will skip over classes with no examples provided in y.

labels = y(:)==(ones(length(y),1) * [1:K]);

m = max(sum(labels));

idxs = [];

for k = 1:K
	if ~isempty(idx = find(labels(:,k)))
		idxs = [idxs;idx(randi(length(idx),[m,1]))];
	end
end

end
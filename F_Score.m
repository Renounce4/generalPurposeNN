function F = F_Score(y, p, b)
% USAGE: function F = F_Score(y, p, b=1)
% Calculates the F score of your predictions (p) given ground truth labels (y), and using bias (b).
% F = the effectiveness of retrieval with b times as much importance given to recall as precision.
% Assumes y and p are binary vectors of the same length.

if ~exist('b', 'var') || isempty(b)
	b = 1;
end

tp = sum(p==1 & y==1);
fp = sum(p==1 & y==0);
fn = sum(p==0 & y==1);

prec = 0;
recall = 0;
F1 = 0;

if(tp~=0 || fp ~= 0)
	prec = tp / (tp+fp);
end

if(tp~=0 || fn ~= 0)
	recall = tp / (tp+fn);
end

if(prec~=0 || recall~=0)
	F1 = (1 + b^2) * (prec * recall) / (((b^2) * prec) + recall);
end

end
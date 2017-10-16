function F1 = F1_Score(y, p)
% USAGE: function F1 = F1_Score(y, p)
% Calculates the F1 score of your predictions (p) given ground truth labels (y).
% Assumes y and p are of same length.

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
	F1 = (2 * prec * recall) / (prec + recall);
end

end
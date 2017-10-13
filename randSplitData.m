function [X, y, X_cval, y_cval, X_test, y_test] = randSplitData(X, y, cval_prcnt, test_prcnt)
% Randomizes the order of the training examples and then splits them up
% where cval_prcnt is the percentage cut to the cross validation set
% and test_prcnt is the percentage cut to the test set
% and the remaining is returned in X, y.

m = size(X,1);

% ===== Randomize ===== %
r = randperm(m);
X_rand = X(r,:);
y_rand = y(r,:);


% ===== Split ===== %
m_cval = floor(m * cval_prcnt);
m_test = floor(m * test_prcnt);
m_train = m - m_test - m_cval;

X = X_rand(1:m_train,:);
y = y_rand(1:m_train,:);

X_cval = X_rand(m_train+1:m_train+m_cval,:);
y_cval = y_rand(m_train+1:m_train+m_cval,:);

X_test = X_rand(m_train+m_cval+1:m_train+m_cval+m_test,:);
y_test = y_rand(m_train+m_cval+1:m_train+m_cval+m_test,:);

end
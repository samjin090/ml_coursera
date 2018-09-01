% Plotting learning curves with randomly selected examples
% 
% In practice, especially for small training sets, when you plot learning 
% curves to debug your algorithms, it is often helpful to average across 
% multiple sets of randomly selected examples to determine the training 
% error and cross validation error.
% Concretely, to determine the training error and cross validation error 
% for i examples, you should first randomly select i examples from the 
% training set and i examples from the cross validation set. You will 
% then learn the param- eters θ using the randomly chosen training set 
% and evaluate the parameters θ on the randomly chosen training set and
% cross validation set. The above steps should then be repeated multiple
% times (say 50) and the averaged error should be used to determine the
% training error and cross validation error for i examples.
%For this optional (ungraded) exercise, you should implement the above
% strategy for computing the learning curves. For reference, figure 10
% shows the learning curve we obtained for polynomial regression with 
% λ = 0.01. Your figure may differ slightly due to the random selection 
% of examples.

function [error_train, error_val] = ...
    learningCurve(X, y, Xval, yval, lambda)

% Number of training examples
m = size(X, 1);

% You need to return these values correctly
error_train = zeros(m, 50);
error_val   = zeros(m, 50);

for i = 1:m
	for j = 1:50
		sel = randperm(size(X,1));
		sel = sel(1:i);
		tempX = X(sel, :);
		tempy = y(sel, :);
		tempXval = Xval(sel, :);
		tempyval = yval(sel, :);
		[theta] = trainLinearReg(tempX, tempy, lambda);
		[error_train(i,j), grad1] = linearRegCostFunction(tempX, tempy, theta, 0);
		[error_val(i,j), grad2] = linearRegCostFunction(Xval, yval, theta, 0);
	end
end
error_train = mean(error_train, 2);
error_val = mean(error_val, 2);

end
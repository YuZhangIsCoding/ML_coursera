function [error_train, error_val] = ...
    learningCurve_random(X, y, Xval, yval, lambda)
%LEARNINGCURVE Generates the train and cross validation set errors needed 
%to plot a learning curve
%   [error_train, error_val] = ...
%       LEARNINGCURVE(X, y, Xval, yval, lambda) returns the train and
%       cross validation set errors for a learning curve. In particular, 
%       it returns two vectors of the same length - error_train and 
%       error_val. Then, error_train(i) contains the training error for
%       i examples (and similarly for error_val(i)).
%       
%       For small training sets, it's helpful to average across multiple
%       sets of randomly selected samples to determine the training error
%       and cross validation error.

% Number of training examples
m = size(X, 1);


% You need to return these values correctly
error_train = zeros(m, 1);
error_val   = zeros(m, 1);

% ---------------------- Sample Solution ----------------------
repeat = 50;
val_rate = 0.6
for i = 1:m
    train_sum = 0;
    val_sum = 0;
    for j =  1:repeat
        inds = randperm(m);
        [theta] = trainLinearReg(X(inds(1:i), :), y(inds(1:i)), lambda);
        train_sum = train_sum+linearRegCostFunction(X(inds(1:i), :), y(inds(1:i)), theta, 0);
        inds = randperm(length(Xval));
        val_sum = val_sum+linearRegCostFunction(Xval(inds(1:val_rate*length(Xval)), :), yval(inds(1:val_rate*length(Xval))), theta, 0);
    end
    error_train(i) = train_sum/repeat;
    error_val(i) = val_sum/repeat;
end
end

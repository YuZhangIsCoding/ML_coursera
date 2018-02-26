function plotData(X, y)
%PLOTDATA Plots the data points X and y into a new figure 
%   PLOTDATA(x,y) plots the data points with + for the positive examples
%   and o for the negative examples. X is assumed to be a Mx2 matrix.

% Create New Figure
figure; hold on;

% ====================== YOUR CODE HERE ======================
% Instructions: Plot the positive and negative examples on a
%               2D plot, using the option 'k+' for the positive
%               examples and 'ko' for the negative examples.
%

%the following is a quick way to visualize the data
%gscatter(X(:,1), X(:,2), y, 'yk', 'o+', [12,12]);

ad = find(y == 1);
notad  = find(y == 0);
plot(X(ad, 1), X(ad, 2), 'k+', 'LineWidth', 2, 'MarkerSize', 7);
plot(X(notad, 1), X(notad, 2), 'yo', 'MarkerFaceColor', 'y', 'MarkerSize', 7);

% =========================================================================



hold off;

end

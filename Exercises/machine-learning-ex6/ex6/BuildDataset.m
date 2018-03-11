function [X, y, Xval, yval, Xtest, ytest] = BuildDataset()
    info1 = dir('spam_2/0*');
    info2 = dir('easy_ham_2/0*');
    l1 = length(info1);
    l2 = length(info2);
    results = [ones(l1, 1); zeros(l2, 1)];
    n = 1899; % feature size
    features = zeros(l1+l2, n);
    for i = 1:(l1+l2)
        if i <= l1
            filename = strcat('spam_2/', info1(i).name);
        else
            filename = strcat('easy_ham_2/', info2(i-l1).name);
        email_contents = readFile(filename);
        word_indices = processEmail(email_contents);
        features(i, :) = emailFeatures(word_indices)';
    end
    inds = randperm(l1+l2);
    i1 = floor((l1+l2)*0.6);
    i2 = floor((l1+l2)*0.8);
    X = features(1:inds(1:i1), :);
    y = results(1:inds(1:i1), :);
    Xval = features(inds(l1+1:l2), :);
    yval = results(inds(l1+1:l2), :);
    Xtest = features(inds(l2+1:end), :);
    ytest = results(inds(l2+1:end), :);
end

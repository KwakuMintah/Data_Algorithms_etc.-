clear all;
load("dataset_1.mat");

%This splits the data into Train and Test
ind = randperm(1000);
w3XTestOne = x(ind(1:300),:);
w3XTrainOne = x(ind(301:end),:);
w3YTestOne = y(ind(1:300),:);
w3YTrainOne = y(ind(301:end),:);

D_siz = size(coeffs);
D = D_siz(1);
%Started off with 5 as that was in the docs
k = 5;

coefL0 = fminsearch(@(A) lZeroNorm(A,w3XTestOne,w3YTestOne),rand(D,1));
coefL1 = fminsearch(@(A) lOneNorm(A,w3XTestOne,w3YTestOne),rand(D,1));
coefL2 = fminsearch(@(A) lTwoNorm(A,w3XTestOne,w3YTestOne),rand(D,1));
%coefLasso = fminsearch(@(A) lasso(w3XTestOne,w3YTestOne),rand(D,1));
%Ridge works very well. Lasso not as much.
coefRidge = fminsearch(@(A) ridge(w3YTestOne,w3XTestOne,k),rand(D,1));

yHatL0TestOne = polyval(coefL0, w3XTestOne);
yHatL1TestOne = polyval(coefL1, w3XTestOne);
yHatL2TestOne = polyval(coefL2, w3XTestOne);
yHatRidgeTestOne = polyval(coefRidge, w3XTestOne);

yHatL0TrainOne = polyval(coefL0, w3XTrainOne);
yHatL1TrainOne = polyval(coefL1, w3XTrainOne);
yHatL2TrainOne = polyval(coefL2, w3XTrainOne);
yHatRidgeTrainOne = polyval(coefRidge, w3XTrainOne);

%Started to try Vandermonde
VanderXTestOne = vander(w3XTestOne);

plot(coeffs,'-');
hold all;
plot(coefL2,'-');

function yL0 = lZeroNorm(A,x,y)
    yHatL0 = polyval(A,x);
    yL0 = max(abs(yHatL0 - y));
end

function yL1 = lOneNorm(A,x,y)
    yHatL1 = polyval(A,x);
    yL1 = mean(abs(yHatL1 - y));
end

function yL2 = lTwoNorm(A,x,y)
    yHatL2 = polyval(A,x);
    yL2 = mean(abs(yHatL2 - y).^2);
end
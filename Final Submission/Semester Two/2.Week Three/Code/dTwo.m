clear all;
load("dataset_2.mat");

ind = randperm(1000);
w3XTestTwo = x(ind(1:300),:);
w3XTrainTwo = x(ind(301:end),:);
w3YTestTwo = y(ind(1:300),:);
w3YTrainTwo = y(ind(301:end),:);

D_siz = size(coeffs);
D = D_siz(1);

coefL0 = fminsearch(@(A) normL0(A,w3XTrainTwo,w3YTrainTwo),rand(D,1));
coefL1 = fminsearch(@(A) normL1(A,w3XTrainTwo,w3YTrainTwo),rand(D,1));
coefL2 = fminsearch(@(A) normL2(A,w3XTrainTwo,w3YTrainTwo),rand(D,1));

yHatL0TrainTwo = polyval(coefL0, w3XTrainTwo);
yHatL1TrainTwo = polyval(coefL1, w3XTrainTwo);
yHatL2TrainTwo = polyval(coefL2, w3XTrainTwo);

yHatL0TestTwo = polyval(coefL0, w3XTestTwo);
yHatL1TestTwo = polyval(coefL1, w3XTestTwo);
yHatL2TestTwo = polyval(coefL2, w3XTestTwo);

plot(coeffs,'-');
hold all;
plot(coefL2,'-');

function yL0 = normL0(A,x,y)
    yHatL0 = polyval(A,x);
    yL0 = max(abs(yHatL0 - y));
end

function yL1 = normL1(A,x,y)
    yHatL1 = polyval(A,x);
    yL1 = mean(abs(yHatL1 - y));
end

function yL2 = normL2(A,x,y)
    yHatL2 = polyval(A,x);
    yL2 = mean(abs(yHatL2 - y).^2);
end
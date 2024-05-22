clear all;
load("dataset_3.mat");

ind = randperm(1000);
w3XTestThree = x(ind(1:300),:);
w3XTrainThree = x(ind(301:end),:);
w3YTestThree = y(ind(1:300),:);
w3YTrainThree = y(ind(301:end),:);

D = 10;
coeffs = D * rand(D,1);

coefL0 = fminsearch(@(A) normL0(A,w3XTrainThree,w3YTrainThree),rand(D,1));
coefL1 = fminsearch(@(A) normL1(A,w3XTrainThree,w3YTrainThree),rand(D,1));
coefL2 = fminsearch(@(A) normL2(A,w3XTrainThree,w3YTrainThree),rand(D,1));

yHatL0TrainTwo = polyval(coefL0, w3XTrainThree);
yHatL1TrainTwo = polyval(coefL1, w3XTrainThree);
yHatL2TrainTwo = polyval(coefL2, w3XTrainThree);

yHatL0TestTwo = polyval(coefL0, w3XTestThree);
yHatL1TestTwo = polyval(coefL1, w3XTestThree);
yHatL2TestTwo = polyval(coefL2, w3XTestThree);

plot(w3XTestThree,w3YTestThree,'x');
hold all;
plot(w3XTestThree,yHatL2TestTwo,'o');

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
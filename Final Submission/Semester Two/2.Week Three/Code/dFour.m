clear all;
load("dataset_4.mat");

ind = randperm(1000);
w3XTestFour = x(ind(1:300),:);
w3XTrainFour = x(ind(301:end),:);
w3YTestFour = y(ind(1:300),:);
w3YTrainFour = y(ind(301:end),:);

D = 10;

coefL0 = fminsearch(@(A) normL0(A,w3XTrainFour,w3YTrainFour),rand(D,1));
coefL1 = fminsearch(@(A) normL1(A,w3XTrainFour,w3YTrainFour),rand(D,1));
coefL2 = fminsearch(@(A) normL2(A,w3XTrainFour,w3YTrainFour),rand(D,1));

yHatL0TrainTwo = polyval(coefL0, w3XTrainFour);
yHatL1TrainTwo = polyval(coefL1, w3XTrainFour);
yHatL2TrainTwo = polyval(coefL2, w3XTrainFour);

yHatL0TestTwo = polyval(coefL0, w3XTestFour);
yHatL1TestTwo = polyval(coefL1, w3XTestFour);
yHatL2TestTwo = polyval(coefL2, w3XTestFour);

plot(w3XTestFour,w3YTestFour,'x');
hold all;
plot(w3XTestFour,yHatL2TestTwo,'o');

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
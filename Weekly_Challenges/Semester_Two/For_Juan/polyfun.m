clear all;

load("dataset_1.mat");
D = 10;

ind = randperm(1000);
w3XTrainOne = x(ind(1:300),:);
w3XTestOne = x(ind(301:end),:);
w3YTrainOne = x(ind(1:300),:);
w3YTestOne = x(ind(301:end),:);

coefL0 = fminsearch(@(A) l0Norm(A,w3XTrainOne,w3YTrainOne),rand(D,1));
coefL1 = fminsearch(@(A) l1Norm(A,w3XTrainOne,w3YTrainOne),rand(D,1));
coefL2 = fminsearch(@(A) l2Norm(A,w3XTrainOne,w3YTrainOne),rand(D,1));

w3YTrainOneL0 = polyval(coefL0, w3XTrainOne);
w3YTrainOneL1 = polyval(coefL1, w3XTrainOne);
w3YTrainOneL2 = polyval(coefL2, w3XTrainOne);
w3YTestOneL0 = polyval(coefL0, w3XTestOne);
w3YTestOneL1 = polyval(coefL1, w3XTestOne);
w3YTestOneL2 = polyval(coefL2, w3XTestOne);
w3YModelL0 = polyval(coefL0, x);
w3YModelL1 = polyval(coefL1, x);
w3YModelL2 = polyval(coefL2, x);

function yL0 = l0Norm(A,x,y)
    yAlt = polyval(A,x);
    yL0 = max(abs(yAlt - y));
end

function yL1 = l1Norm(A,x,y)
    yAlt = polyval(A,x);
    yL1 = mean(abs(yAlt - y));
end

function yL2 = l2Norm(A,x,y)
    yAlt = polyval(A,x);
    yL2 = mean(abs(yAlt - y).^2);
end
clear all;
load("mnist.mat");

%Taken from Week One
XTrainReshape = reshape(XTrain,28,28,60000);
XTrainReshapeForDiv = reshape(XTrainReshape, [], 60000);
XTrainReshapeTrans = transpose(XTrainReshapeForDiv);
YTrainTrans = transpose(YTrain);
C = mrdivide(XTrainReshapeForDiv,YTrainTrans);
CTrans = transpose(C);
YHatTrain = (XTrainReshapeTrans * C);
YHatTrainTrans = transpose(YHatTrain);

d = YTrainTrans - (CTrans * XTrainReshapeForDiv);
nFull = size(YTrainTrans);
n = nFull(1,2);

mse = mean(abs(YHatTrain - YTrain));

maxError = abs(YHatTrain - YTrain);
l_two = lTwoNorm(YHatTrain,YTrain,n);
l_one = lOneNorm(YHatTrain,YTrain,n);
mnistRidge = ridge(YHatTrain,YTrain,5);
mnistLasso = lasso(YHatTrain,YTrain);
L = genLoss(C,d,YTrain,YHatTrain,n,1,0);

x0 = [1,1];
fminGL = fminsearch(@(YHatTrain) genLoss(C,d,YTrain,YHatTrain,n,1,2),x0);
fminLT = fminsearch(@(YHatTrain) lOneNorm(YHatTrain,YTrain,n),x0);
fminLO = fminsearch(@(YHatTrain) lTwoNorm(YHatTrain,YTrain,n),x0);

vecMat = ones(784,1);
vecTrain = ones(60000,1);
vecTest = ones(10000,1);

CTrainGL = (fminGL(1,1) * vecMat);
CTrainLT = (fminLT(1,1) * vecMat);
CTrainLO = (fminLO(1,1) * vecMat);
dTrainGL = (fminGL(1,2) * vecTrain);
dTrainLT = (fminLT(1,2) * vecTrain);
dTrainLO = (fminLO(1,2) * vecTrain);

ykGL = (XTrainReshapeTrans * CTrainGL) + dTrainGL;

ykLT = (XTrainReshapeTrans * CTrainLT) + dTrainLT;

ykO = (XTrainReshapeTrans * CTrainLO) + dTrainLO;

yCheck = ones(60000,3);
yCheck(:,1) = ykGL;
yCheck(:,2) = ykLT;
yCheck(:,3) = ykO;
%imagesc(yCheck);

yCheckSmaller = ones(60000,2);
yCheck(:,1) = ykGL;
yCheck(:,2) = ykO;
%imagesc(yCheckSmaller);

plot(ykGL);
hold all;
plot(ykO);

%One Hot Encode
digits = ["One","Two","Three","Four","Five","Six","Seven","Eight","Nine","Ten"];
digits = categorical(digits);
categories(digits);
digits = onehotencode(digits,1);
XTrainReshapeOHE = reshape(XTrainReshape, [], 10);
XTrainReshapeOHETrans = transpose(XTrainReshapeOHE);
digitsVec = reshape(digits,[],1);
cOHE = mrdivide(XTrainReshapeOHE,digits);
YHatOHE = (XTrainReshapeOHETrans * cOHE);
YHatOHEVec = reshape(YHatOHE,[],1);

nFullOHE = size(digits);
nOHE = nFullOHE(1,2);
mseOHE = mean(abs(YHatOHE - digits));
maxErrorOHE = abs(YHatOHE - digits);
l_twoOHE = lTwoNorm(YHatOHE,digits,nOHE);
l_oneOHE = lOneNorm(YHatOHE,digits,nOHE);
mnistRidgeOHE = ridge(YHatOHEVec,digitsVec,5);

function l_two = lTwoNorm(y_k,YTestTrans,n)
  coef = 1/n;
  brac = abs(y_k - YTestTrans);
  bracsq = brac.^2;
  bracsum = sum(bracsq,"all");
  rootsum = sqrt(bracsum);
  l_two = coef * rootsum;
end

function l_one = lOneNorm(y_k,YTestTrans,n)
  coef = 1/n;
  brac = y_k - YTestTrans;
  bracsum = sum(brac,"all");
  l_one = coef * bracsum;
end

function L = genLoss(x,d,Y,YHat,n,lambdaOne,lambdaTwo)
    problemLTwo = lTwoNorm(YHat,Y,n);
    lOne = lOneNorm(d,x,n);
    lTwo = lTwoNorm(d,x,n);
    L = problemLTwo + (lambdaOne * lOne) + (lambdaTwo * lTwo);
end
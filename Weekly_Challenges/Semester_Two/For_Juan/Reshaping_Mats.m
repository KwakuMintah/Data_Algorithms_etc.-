
function [XReshape, XReshapeForDiv] = Reshaping_Mats(XTest)
    XReshape = reshape(XTest,28,28,10000);
    XReshapeForDiv = reshape(XReshape, [], 10000);
end
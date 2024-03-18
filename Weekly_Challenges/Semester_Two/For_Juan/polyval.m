function polynom = polyval(C,x)
    D = 10;
    fPolOne = zeros(D,1);
    for i = 1:D
        xPowOne = x(i,1) ^ i;
        cJOne = C(i,1);
        polValOne = xPowOne * cJOne;
        fPolOne(i,1) = polValOne;
    end
    polynom = fPolOne;
end
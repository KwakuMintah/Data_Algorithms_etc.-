[f_pol] = polyFunc(coeffs, x)
    D = 10;
    f_pol = zeros(D,1);
    for i = 0:D
        x_pow = x(i,1) ^ i;
        C_j = coef(i,1);
        pol_val = x_pow * C_j;
        f_pol(i,1) = pol_val;
    end
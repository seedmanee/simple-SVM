function sim = gaussian(x1, x2, param)
x1 = x1(:); x2 = x2(:);
sim = 0;

xx = x1 - x2;
sim = exp(-(xx'*xx)*param);
    
end

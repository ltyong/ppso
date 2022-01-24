% Griewank
function f=Griewank(x)
[ps,D]=size(x);
xs = x.^2;
f=sum(xs,2)/4000-prod(cos(x./sqrt([1:D])),2)+1;
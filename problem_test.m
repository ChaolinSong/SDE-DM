function obj = problem_test(x,problem_number)
switch problem_number
    case 1
    %function one Ellipsoid method
    x=x.^2; [m,n]=size(x); I=ones(1,n);
    for i=1:n, I(i)=i; end; I=repmat(I,m,1);
    x=x.*I;
    obj = sum(x,2);

    case 2
    %function two Rosenbrock Problem 2.048
    x1=x; x2=x; if size(x,1)~=0, x1(:,1)=[]; x2(:,size(x,2))=[]; end
    x3=sum(100*(x1-x2.^2).^2,2);
    x4=sum((1-x2).^2,2);
    obj = x3+x4;

    case 3
    %function_three Ackley Problem 32.768
    dim=size(x,2);% dim=2;
    x1=-0.2*(sum(x.^2,2)/dim).^0.5;
    x2=sum(cos(2*pi*x),2)/dim;
    obj=-20*exp(x1)-exp(x2)+20+exp(1);

    case 4
    %function_four Griewank Problem 600
    x1=sum(x.^2./4000,2); [m,n]=size(x); I=ones(1,n);
    for i=1:n, I(i)=i^0.5; end; I=repmat(I,m,1);
    x2=cos(x./I); x2=prod(x2,2);
    obj = 1+ x1 -x2;

    case 5
    %function five Rastrigin Problem
    h=10; l=2;
    x1=h*(1-cos(l*pi*x));
    x2=(x).^2; 
    obj = sum(x1+x2,2);

%function six Schwefel Problem
% x= x /scalex;
% obj = 418.8929 * size(x,2)- sum(x .* sin(abs(x).^0.5), 2);
%obj = obj/scaley;

%function seven Schwefel Problem
% x=abs(x);
% obj = max(x,[],2);
end
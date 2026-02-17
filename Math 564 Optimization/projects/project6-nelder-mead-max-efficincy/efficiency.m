function f=efficiency(x)

x=(x-[0;1;1])./[5;2;3];

%%% Set up hyperparameters
N = 10;
a = 0.3;
w = 1.5;
normconst=1.14;

% read coefficients
A = csvread('pdata.csv');
p = reshape(A,N+1,N+1,N+1);

% compute raw f
f=0;
for k1=0:N
    for k2=0:N
        for k3=0:N
            if k1+k2+k3<=N
                f=f+p(k1+1,k2+1,k3+1)*(x(1)^k1)*(x(2)^k2)*(x(3)^k3);
            end
        end
    end
end

% adjust f to be 0<f<=100
f = a/(a+f^2);
r = norm(x)^2/(w^2);
f = f*exp(-r);
f = f*normconst;
f = f*100;

return

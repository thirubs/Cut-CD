%% This demo file will help you to understand the overall process.
clear;
nnz = 0.00001; % Sparsity
SI = 512; % Mode length 
X = sptenrand([SI SI SI],nnz);%<-- Create a sparse tensor with nonzeros.
Y = sptenrand([SI SI],nnz); % Input matrix
useruser = double(Y);
N = ndims(X);
maxiter = 30; % Maximum number of iterations
J = 50;  % no of reduced dimension factor of tensor and matrix
Uinit = cell(N,1);    
for n = 1:N 
	Uinit{n} = normalize_factor(rand(size(X,n),J),2);        
end
U=arrange(ktensor(Uinit));
N = 2;
UinitM = cell(N,1);
for n = 1:N
    if n == 1
       UinitM{n} = Uinit{n};
    else
        UinitM{n} = normalize_factor(rand(size(useruser,n),J),2);
    end
end
UM = arrange(ktensor(UinitM));
[UVW_CutCD] = cutcd(X,J,U,useruser,UM,maxiter);


function [U,UM] = cutcd(X,J,U,Y,UM,maxiters)
%%N-CMTF. Nonnegative Coupled Matrix Tensor Factorization using Cut-off Coordinate Descent (Cut-CD) algorithm
% This implementation can be used for only 3-order tensor with mode 1
% shared between the tensor and matrix.

%%
% [Input]
% X           data tensor                        (sptensor)
% Y           data matrix                        (double)
% J           Rank of the tensor and matrix
% U,UM        Randonly initialized factor matrices
% maxiters    the numer of iterations   

% [Output]
% U        the factorization result             (ktensor)
%% Title
fprintf('*************************\n')
fprintf('N-CMTF - CutCD\n');
fprintf('*************************\n')

%% Initialization for parameters 
%tic
N = ndims(X);
epsilon=1e-12;
tol=1.0e-4;
nor = 2; % Normalization (1 for norm 1 and 2 for norm 2)
%% Error Cheacking
% None check by yourself 
%% compute Second-order derivatives for tensor factors
Hmat=sparse(ones(J,J));
for n=1:N
    Hmat= Hmat .* ((U{n,1}' *U{n,1})+epsilon);
end

%% iterations
%tic
for i = 1:maxiters
    for n= 1:N  % N is number of factor matrices in the tensor
        rows_and_cols = size(U{n});
        rows = rows_and_cols(1);
        cols = rows_and_cols(2);
        
        %Update second-order derivative for nth mode
        Hmat = Hmat ./ ((U{n,1}' *U{n,1})+epsilon);
        
        % MTTKRP calculation
        tmpmat=mttkrp(X,U,n);
      
        % Gradient calculation
        grad= -(tmpmat-(U{n,1} *Hmat));
        
        % compute second-order derivative and gradient for additional matrix     
        HHT=sparse(ones(J,J));
        type = 0;
        if n == 1
           HHT = (UM{2,1}' *UM{2,1})+epsilon;
           F2_grad = -((Y*UM{2,1})-(U{1,1} *HHT)); %gradient of input matrix
           F2 = F2_grad;
           grad = grad + F2_grad;
           Hmat_a = Hmat+HHT;      
           [Unew]=fac_update(grad,Hmat_a,U{n,1},rows,cols);
        else
           [Unew]=fac_update(grad,Hmat,U{n,1},rows,cols);
        end
        U{n,1}=  Unew;
        if n == 1
            UM{1,1} = Unew;
            %Check nonnegativity 
            UM{1,1}(UM{1,1}<=epsilon)=epsilon;

            UM{1,1}=normalize_factor(UM{1,1},nor);

        end
        %Check nonnegativity 
        U{n,1}(U{n,1}<=epsilon)=epsilon;
        %Normalization (if you need)
        if (n~=N)
            U{n,1}=normalize_factor(U{n,1},nor);
        end
        %Update second-order derivative with updated matrix			
        Hmat = Hmat .* ((U{n,1}' *U{n,1})+epsilon);
    end 
    % Update H of input matrix's factor
    ATA = (U{1,1}' *U{1,1})+epsilon;
    grad_H = (UM{2,1} *ATA)-(Y'*U{1,1});  
    rows_and_cols = size(UM{2});
    rows = rows_and_cols(1);
    cols = rows_and_cols(2);
    [UMnew]=fac_update(grad_H,ATA,UM{2,1},rows,cols);
    UM{2,1} = UMnew;
    %Check nonnegativity 
	UM{2,1}(UM{2,1}<=epsilon)=epsilon;
	%Normalization (if you need)
	if(2~=N)
		UM{2,1}=normalize_factor(UM{2,1},nor);
    end
end

function[U,UM] = cutcd_sc(X,J,U,Y,UM,maxiters,la)
%%N-CMTF. Nonnegative Coupled Matrix Tensor Factorization with sparsity constraint using Cut-off Coordinate Descent (Cut-CD-SC) algorithm
% This implementation can be used for only 3-order tensor with mode 1
% shared between the tensor and matrix.

%%
% [Input]
% X           data tensor                        (sptensor)
% Y           data matrix
% J           the number of Factors              (double)
% maxiters     the numer of iterations            
% [Output]
% U        the factorization result             (ktensor)
%% Title
fprintf('*************************\n')
fprintf('N-CMTF - Cut-CD-SC\n');
fprintf('*************************\n')
%% Initialization for parameters 
N = ndims(X);
JM = J;
epsilon=1e-12;
tol=1.0e-4;
%% Error Cheacking
% None check by yourself 
%% compute Second-order derivatives for tensor factors
Hmat=sparse(ones(J,J));
for n=1:N
    Hmat= Hmat .* ((U{n,1}' *U{n,1})+epsilon);
end

%% iterations
for i = 1:maxiters
    tic
    for n= 1:N
       
    % Calculate the diagonal matrix 
    rows_and_cols = size(U{n});
    rows = rows_and_cols(1);
    cols = rows_and_cols(2);
    Q_u = eye(rows);
    for qu = 1:rows
        Q_u(qu,qu) = 1/(2*(norm(U{n}(qu,:))));
    end
    
    %%Update second-order derivative for nth mode
	Hmat = Hmat ./ ((U{n,1}' *U{n,1})+epsilon);
    
    % MTTKRP calculation
    tmpmat=mttkrp(X,U,n);
    
    ads = la*(Q_u*U{n,1});
    
    % Gradient calculation
	grad= -(tmpmat-(U{n,1} *Hmat)) + ads;
    Hmat = Hmat ;
    
    %F2_grad = zeros(rows,JM);
    HHT=sparse(ones(J,J));
    type = 0;
    if n == 1
        HHT = (UM{2,1}' *UM{2,1})+epsilon;
        F2_grad = -((Y*UM{2,1})-(U{1,1} *HHT)); %gradient of input matrix
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
        
		UM{1,1}=normalize_factor(UM{1,1},1);
	
    end
    U{n,1}(U{n,1}<=epsilon)=epsilon;
	%Normalization (if you need)
	if (n~=N)
		U{n,1}=normalize_factor(U{n,1},1);
	end
	%Update Hadamard matrix with updated matrix			
	Hmat = Hmat .* ((U{n,1}' *U{n,1})+epsilon);
    end 
   
   % Update H of input matrix's factor
   Q_u = eye(J);
    for qu = 1:J
        Q_u(qu,qu) = 1/(2*(norm(UM{2,1}(:,qu))));
    end
   
   ATA = (U{1,1}' *U{1,1})+epsilon;
   grad_H = (UM{2,1} *ATA)-(Y'*U{1,1}) + (la*Q_u * UM{2,1}')';  
   rows_and_cols = size(UM{2});
   rows = rows_and_cols(1);
   cols = rows_and_cols(2);
   [UMnew]=goWiter_NNCMTD_new(grad_H,ATA,UM{2,1},tol,rows,cols,J^2);
   %UMnew = trace(UMnew'*Q_u*UMnew);
   UM{2,1} = UMnew;
   %Check nonnegativity 
	UM{2,1}(UM{2,1}<=epsilon)=epsilon;
	%Normalization (if you need)
	if (2~=N)
		UM{2,1}=normalize_factor(UM{2,1},1);
    end
end

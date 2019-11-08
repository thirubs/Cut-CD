function [Wout] = fac_update(GW,HH,W,n,k)
%% Input Parameters
% GW - Gradient
% HH - Second-order derivative
% W - Factor Matrix to update
% n - number of rows in W
% k - number of columns in W
%% Version 1 - Column-wise (Very Fast)
for p=1:k
		for i = 1:n          
			s = GW(i,p)/HH(p,p);
			s = W(i,p)-s;
			if ( s< 0)
               s=0;
            end			
            s = s-W(i,p);
			ss(i,p) = s; 
            diffobj = (-1)*s*GW(i,p)-0.5*HH(p,p)*s*s;
           diff_all(i,p) = diffobj;
		end   
		%% Normalization of element importance
		diff_all_in(:,p) = diff_all(:,p);
		diff_all(:,p) = (diff_all_in(:,p)-min(diff_all_in(:,p)))/(max(diff_all_in(:,p))-min(diff_all_in(:,p)));
		threshold = mean(diff_all(:,p));   
		if isnan(threshold)
			threshold = 0;
        end
        for i=1:n
			diffobj = diff_all(i,p);
            if (diffobj>=threshold) 
				W(i,p) = W(i,p)+ss(i,p);
				GW(i,p) = GW(i,p)+( ss(i,p)*HH(p,p));
            end
        end
end
%% Output Arguments
Wout = W;


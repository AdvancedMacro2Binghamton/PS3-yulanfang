clear all; close all; clc;
% PARAMETERS
beta = .994; % discount factor 
sigma = 1.5; % coefficient of risk aversion
b = 0.5; % replacement ratio (unemployment benefits)
y_s = [1, b]; % endowment in employment states
PI = [.97 .03; .5 .5]; % transition matrix

% ASSET VECTOR
a_lo = -2; % lower bound of grid points
a_hi = 5; % upper bound of grid points
num_a = 100;
a = linspace(a_lo, a_hi, num_a); % asset (row) vector
q_min = 0.98;
q_max = 1;

% ITERATE OVER ASSET PRICES
aggsav = 1 ;
while abs(aggsav) >= 0.01
    % INITIAL GUESS FOR q
    q_guess = (q_min + q_max) / 2;
    
    % CURRENT RETURN (UTILITY) FUNCTION
    cons = bsxfun(@plus,bsxfun(@minus,a',q_guess*a),permute(y_s, [1 3 2]));
    ret = (cons .^ (1-sigma)) ./ (1 - sigma);
    ret (cons < 0) = -Inf;
    
    % INITIAL VALUE FUNCTION GUESS
    v_guess = zeros(2, num_a);
    
    % VALUE FUNCTION ITERATION
    e1 = 1;
    tol = 1e-06;
    while e1 > tol
        % CONSTRUCT RETURN + EXPECTED CONTINUATION VALUE
        v = ret + beta * repmat(permute((PI*v_guess),[3 2 1]),[num_a 1 1]);
        
        % CHOOSE HIGHEST VALUE (ASSOCIATED WITH a' CHOICE)
        [vfn, indx] = max(v, [], 2);
        
        e1 = max(max(abs(permute(vfn, [3 1 2])-v_guess)));
        
        v_guess = permute(vfn, [3 1 2]);
    end
    
    % KEEP DECSISION RULE
    pol_indx = permute(indx,[3,1,2]);
    pol_fn = a(pol_indx);
    
    % SET UP INITITAL DISTRIBUTION
    Mu = ones(2, num_a)/(2*num_a);
    
    % ITERATE OVER DISTRIBUTIONS
    e2 = 1;
    while e2 >= tol
        [emp_ind, a_ind, mass] = find(Mu); % find non-zero indices
        MuNew = zeros(size(Mu));
        for ii = 1:length(emp_ind)
            apr_ind = pol_indx(emp_ind(ii), a_ind(ii)); % which a prime does the policy fn prescribe?
            MuNew(:, apr_ind) = MuNew(:, apr_ind) + ... % which mass of households goes to which exogenous state?
                (PI(emp_ind(ii), :) * mass(ii))';
        end
        e2 = max(max(abs(MuNew-Mu)));
        Mu = MuNew;
    end
    
    aggsav = sum(sum(Mu.*pol_fn));
    if aggsav > 0
        q_min = q_guess;
    else
        q_max = q_guess;
    end 
end

d = reshape(Mu',[2*num_a 1]);
wealth = reshape(bsxfun(@plus, repmat(a, [2,1]), y_s')',[2*num_a 1]);
earnings = reshape(repmat(y_s',[1 num_a])',[2*num_a 1]);

d_wealth = cumsum(sortrows([d,d.*wealth,wealth],3));
L_wealth = bsxfun(@rdivide,d_wealth,d_wealth(end,:))*100;
Gini_wealth = 1-(sum((d_wealth(1:end-1,2)+d_wealth(2:end,2)).*diff(d_wealth(:,1))))-(d_wealth(1,1)*d_wealth(1,2));
d_earnings = cumsum(sortrows([d,d.*earnings,earnings],3));
L_earnings = bsxfun(@rdivide,d_earnings,d_earnings(end,:))*100;
Gini_earnings = 1-(sum((d_earnings(1:end-1,2)+d_earnings(2:end,2)).*diff(d_earnings(:,1))))-(d_earnings(1,1)*d_earnings(1,2));

figure
plot(L_wealth(:,1),L_wealth(:,2),L_wealth(:,1),L_wealth(:,1))
legend('Lorenz Curve','45 degree line')
title('Lorenz Curve for Wealth')
xlabel('Cumulative Share of Population')
ylabel('Cumulative Share of Wealth')

figure
plot(L_earnings(:,1),L_earnings(:,2),L_earnings(:,1),L_earnings(:,1))
legend('Lorenz Curve','45 degree line')
title('Lorenz Curve for Earnings')
xlabel('Cumulative Share of Population')
ylabel('Cumulative Share of Earnings')
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%7011 code by YS%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% PART 1 Initialisations

%Filename
filename = 'port1.txt';

%Openfile
fid = fopen(filename, 'r');

%Check number of assets
N = fscanf(fid, '%d', 1);

% Initialize vectors and matrices
mean_returns = zeros(N, 1);
std_devs = zeros(N, 1);
correlations = eye(N);  % Start with identity matrix for correlations

% Read the mean returns and standard deviations
for i = 1:N
    data = fscanf(fid, '%f %f', 2);
    mean_returns(i) = data(1);
    std_devs(i) = data(2);
end

% Read the correlations
while ~feof(fid)
    data = fscanf(fid, '%d %d %f', 3);
    if isempty(data)
        break;
    end
    i = data(1);
    j = data(2);
    correlation = data(3);
    correlations(i, j) = correlation;
    correlations(j, i) = correlation;  % Symmetric matrix
end

% Close the file
fclose(fid);

% Construct the covariance matrix
covariance_matrix = zeros(N, N);
% Calculate the lower triangular part of the covariance matrix
for i = 1:N
    for j = 1:i
        covariance_matrix(i, j) = correlations(i, j) * std_devs(i) * std_devs(j);
    end
end

% Reflect the lower triangular part to the upper triangular part
covariance_matrix = covariance_matrix + triu(covariance_matrix.', 1);
% I did like that to avoid floating point errors.





%%%Now for sample answer provided
%Filename
filename = 'portef1.txt';

%Openfile
data = dlmread('portef1.txt');

ans_returns = data(:, 1);
ans_variances = data(:, 2);
ans_risks = sqrt(ans_variances);

%% Check PD

d = eig(covariance_matrix);
isposdef = all(d > 0)
%% Part 2 Standard Portfolio Optimisation 


%Number of Assets
N = length(mean_returns);

% Define the range of target returns
R_min = min(mean_returns);
R_max = max(mean_returns);
R_range = linspace(R_min, R_max, 5000);  % 5000 points for the efficient frontier

% Storage for the results
portfolio_risks = zeros(size(R_range));
portfolio_returns = zeros(size(R_range));

% Quadratic Programming
options = optimoptions('quadprog', 'Display', 'off');  

% Vector of ones
ones_vector = ones(N, 1);

for i = 1:length(R_range)
    R = R_range(i);
    
    % Define the equality constraints
    Aeq = [ ones_vector' ; mean_returns'];
    beq = [1; R];
    
    % Solve the quadratic programming problem
    [w, ~, exitflag] = quadprog(covariance_matrix, [], [], [], Aeq, beq, zeros(N, 1), [], [], options);
    
    if exitflag == 1
        % Calculate risk (standard deviation)
        portfolio_risks(i) = sqrt(w' * covariance_matrix * w);
        portfolio_returns(i) = mean_returns' * w;
        
       
    else
        portfolio_risks(i) = NaN;
        portfolio_returns(i) = NaN;
    end
end

% Find minimum variance portfolio
[min_risk, min_risk_idx] = min(portfolio_risks);
min_risk_return = portfolio_returns(min_risk_idx);

% Filter portfolios above and below the minimum variance portfolio
above_min_idx = portfolio_returns >= min_risk_return;
below_min_idx = portfolio_returns < min_risk_return;

efficient_risks_above = portfolio_risks(above_min_idx);
efficient_returns_above = portfolio_returns(above_min_idx);

efficient_risks_below = portfolio_risks(below_min_idx);
efficient_returns_below = portfolio_returns(below_min_idx);

% Plot the efficient frontier
figure;
hold on;
plot(efficient_risks_above, efficient_returns_above, 'LineWidth', 2, 'Color', 'k');  % Efficient frontier above min variance portfolio
plot(efficient_risks_below, efficient_returns_below, 'LineWidth', 2, 'Color', 'm', 'LineStyle', '--');  % Efficient frontier below min variance portfolio
plot(ans_risks, ans_returns, 'LineWidth', 2, 'Color', 'r', 'LineStyle', '--');  % Sample Answer
% Plot minimum variance portfolio
plot(min_risk, min_risk_return, 'ro', 'MarkerSize', 10);  
hold off;
xlabel('Risk (Standard Deviation)');
ylabel('Expected Return');
title('Efficient Frontier with Minimum Variance Portfolio');
legend('Efficient Frontier (Above Min Variance)', 'Efficient Frontier (Below Min Variance)', 'Sample Answer', 'Minimum Variance Portfolio', 'Location', 'best');
grid on;




%% Part 2 using Monte Carlo using Standard Weights
rng(42)
N = length(mean_returns);

% Define the number of portfolios/sample size to generate. Higher is better
num_portfolios = 20000000; 

% Storage for the results
portfolio_risks = zeros(1, num_portfolios);
portfolio_returns = zeros(1, num_portfolios);
weights = zeros(N, num_portfolios);

% Generate random portfolios with uniformlly distributed weights
parfor i = 1:num_portfolios

    w = rand(N, 1);
    w = w / sum(w); % Normalize to ensure weights sum to 1

    % Calculate portfolio return and risk
    portfolio_returns(i) = mean_returns' * w;
    portfolio_risks(i) = sqrt(w' * covariance_matrix * w);
    
    % Store portfolio weights
    weights(:, i) = w;
end

% Find the efficient frontier
R_min = min(portfolio_returns);
R_max = max(portfolio_returns);
R_range = linspace(R_min, R_max, 5000);  % 5000 points for the efficient frontier

efficient_portfolio_risks = [];
efficient_portfolio_returns = [];

for target_return = R_range
    % Find portfolios that achieve at least the target return
    eligible_indices = find(portfolio_returns >= target_return);
    if ~isempty(eligible_indices)
        % Among those, find the one with the minimum risk
        [min_risk, min_risk_idx] = min(portfolio_risks(eligible_indices));
        efficient_portfolio_risks = [efficient_portfolio_risks, min_risk];
        efficient_portfolio_returns = [efficient_portfolio_returns, portfolio_returns(eligible_indices(min_risk_idx))];
    end
end

% Ensure unique efficient frontier points
[efficient_portfolio_risks, unique_indices] = unique(efficient_portfolio_risks);
efficient_portfolio_returns = efficient_portfolio_returns(unique_indices);

% Find minimum variance portfolio
[min_risk, min_risk_idx] = min(portfolio_risks);
min_risk_return = portfolio_returns(min_risk_idx);
min_risk_weights = weights(:, min_risk_idx);

% Plot the efficient frontier and minimum variance portfolio
figure;
hold on;
scatter(portfolio_risks, portfolio_returns, 1, 'b'); % Scatter plot of all portfolios
plot(efficient_portfolio_risks, efficient_portfolio_returns, 'LineWidth', 2, 'Color', 'k'); % Efficient frontier
plot(ans_risks, ans_returns, 'LineWidth', 2, 'Color', 'r');
plot(min_risk, min_risk_return, 'ro', 'MarkerSize', 10);  % Plot minimum variance portfolio
hold off;
xlabel('Risk (Standard Deviation)');
ylabel('Expected Return');
title('Theoretical Efficient Frontier with Empricial (Standard Simulation)');
legend('Random Portfolios', 'Empirical Efficient Frontier', 'Theoretical Efficient Frontier','Empirical Minimum Variance Portfolio', 'Location', 'best');
grid on;
%% With extreme values
rng(42)
N = length(mean_returns);

% Define the number of portfolios/sample size to generate. Higher is better
num_portfolios = 20000000; 

% Storage for the results
portfolio_risks = zeros(1, num_portfolios);
portfolio_returns = zeros(1, num_portfolios);
weights = zeros(N, num_portfolios);

% Generate random portfolios with extreme weights
parfor i = 1:num_portfolios
    if (binornd(1,0.7,1,1) == 1)
        % Generate random portfolio weights without extreme bias
        w = rand(N, 1);
        w = w / sum(w); % Normalize to ensure weights sum to 1
    else
        % Generate random portfolio weights with extreme bias using Dirichlet distribution
        alpha = ones(1, N) * 0.01; % Small alpha value for more extreme weights
        w = gamrnd(alpha, 1)';
        w = w / sum(w); % Normalize to ensure weights sum to 1
    end
    % Calculate portfolio return and risk
    portfolio_returns(i) = mean_returns' * w;
    portfolio_risks(i) = sqrt(w' * covariance_matrix * w);
    
    % Store portfolio weights
    weights(:, i) = w;
end

% Find the efficient frontier
R_min = min(portfolio_returns);
R_max = max(portfolio_returns);
R_range = linspace(R_min, R_max, 5000);  % 5000 points for the efficient frontier

efficient_portfolio_risks = [];
efficient_portfolio_returns = [];

for target_return = R_range
    % Find portfolios that achieve at least the target return
    eligible_indices = find(portfolio_returns >= target_return);
    if ~isempty(eligible_indices)
        % Among those, find the one with the minimum risk
        [min_risk, min_risk_idx] = min(portfolio_risks(eligible_indices));
        efficient_portfolio_risks = [efficient_portfolio_risks, min_risk];
        efficient_portfolio_returns = [efficient_portfolio_returns, portfolio_returns(eligible_indices(min_risk_idx))];
    end
end

% Ensure unique efficient frontier points
[efficient_portfolio_risks, unique_indices] = unique(efficient_portfolio_risks);
efficient_portfolio_returns = efficient_portfolio_returns(unique_indices);

% Find minimum variance portfolio
[min_risk, min_risk_idx] = min(portfolio_risks);
min_risk_return = portfolio_returns(min_risk_idx);
min_risk_weights = weights(:, min_risk_idx);

% Plot the efficient frontier and minimum variance portfolio
figure;
hold on;
scatter(portfolio_risks, portfolio_returns, 1, 'b'); % Scatter plot of all portfolios
plot(efficient_portfolio_risks, efficient_portfolio_returns, 'LineWidth', 2, 'Color', 'k'); % Efficient frontier
plot(ans_risks, ans_returns, 'LineWidth', 2, 'Color', 'r');
plot(min_risk, min_risk_return, 'ro', 'MarkerSize', 10);  % Plot minimum variance portfolio
hold off;
xlabel('Risk (Standard Deviation)');
ylabel('Expected Return');
title('Efficient Frontier with Empirical Frontier (With Extreme Value Simulation)');
legend('Random Portfolios', 'Empirical Efficient Frontier', 'Theoretical Efficient Frontier','Empirical Minimum Variance Portfolio', 'Location', 'best');
grid on;

%% Limited Asset Portfolio. At most K assets, use monte carlo

rng(42)
N = length(mean_returns);

% Define the number of portfolios to generate
num_portfolios = 20000000;

% Define the cardinality constraint (at most K assets in the portfolio)
cardinality = 2; 

% Storage for the results
portfolio_risks = zeros(1, num_portfolios);
portfolio_returns = zeros(1, num_portfolios);
weights = zeros(N, num_portfolios);

% Generate random portfolios with at most K assets in each portfolio
parfor i = 1:num_portfolios
    % Generate random portfolio weights with bias towards extremes
    w = zeros(N, 1);
    
    % Randomly select a number of assets between 1 and 'cardinality' to include in the portfolio
    num_selected_assets = randi([1, cardinality]);
    selected_assets = randperm(N, num_selected_assets);

     if (binornd(1,0.7,1,1) == 1)
        % Generate random portfolio weights without extreme bias
        random_weights = rand(num_selected_assets, 1);
        random_weights = random_weights / sum(random_weights); % Normalize to ensure weights sum to 1
    else
        % Generate random portfolio weights with extreme bias using Dirichlet distribution
        alpha = ones(1, num_selected_assets) * 0.01; % Small alpha value for more extreme weights
        random_weights = gamrnd(alpha, 1)';
        random_weights = random_weights / sum(random_weights); % Normalize to ensure weights sum to 1
     end
    
    
    % Place the weights in the weight vector
    w(selected_assets) = random_weights;
    
    % Calculate portfolio return and risk
    portfolio_returns(i) = mean_returns' * w;
    portfolio_risks(i) = sqrt(w' * covariance_matrix * w);
    
    % Store portfolio weights
    weights(:, i) = w;
end

% Find the efficient frontier
R_min = min(portfolio_returns);
R_max = max(portfolio_returns);
R_range = linspace(R_min, R_max, 5000);  % 5000 points for the efficient frontier

efficient_portfolio_risks = [];
efficient_portfolio_returns = [];

for target_return = R_range
    % Find portfolios that achieve at least the target return
    eligible_indices = find(portfolio_returns >= target_return);
    if ~isempty(eligible_indices)
        % Among those, find the one with the minimum risk
        [min_risk, min_risk_idx] = min(portfolio_risks(eligible_indices));
        efficient_portfolio_risks = [efficient_portfolio_risks, min_risk];
        efficient_portfolio_returns = [efficient_portfolio_returns, portfolio_returns(eligible_indices(min_risk_idx))];
    end
end

% Ensure unique efficient frontier points
[efficient_portfolio_risks, unique_indices] = unique(efficient_portfolio_risks);
efficient_portfolio_returns = efficient_portfolio_returns(unique_indices);

% Find minimum variance portfolio
[min_risk, min_risk_idx] = min(portfolio_risks);
min_risk_return = portfolio_returns(min_risk_idx);
min_risk_weights = weights(:, min_risk_idx);

% Plot the efficient frontier and minimum variance portfolio
figure;
hold on;
scatter(portfolio_risks, portfolio_returns, 2, 'b', 'filled'); 
plot(ans_risks, ans_returns, 'LineWidth', 2, 'Color', 'g');
scatter(efficient_portfolio_risks, efficient_portfolio_returns, 5, 'r', 'filled'); % Efficient frontier boundary points in red
plot(min_risk, min_risk_return, 'ro', 'MarkerSize', 10);  % Plot minimum variance portfolio
hold off;
xlabel('Risk (Standard Deviation)');
ylabel('Expected Return');
title('Efficient Frontier with Minimum Variance Portfolio (At Most K assets)');
legend('Random Portfolios','Standard Efficient Frontier' ,'Empirical Limited Efficient Frontier', 'Minimum Variance Portfolio', 'Location', 'best');
grid on;













%% MIQP

if ~isempty(gcp('nocreate'))
    delete(gcp);
end

% Start new parallel pool
parpool;

% Define range of lambda values
lambda_values = 10;
expected_returns = zeros(size(lambda_values));
risks = zeros(size(lambda_values));

parfor i = 1:length(lambda_values)
    lambda = lambda_values(i);
    
    % Define optimization problem
    N = length(mean_returns);
    xvars = optimvar('xvars', N, 1, 'LowerBound', 0, 'UpperBound', 1);
    vvars = optimvar('vvars', N, 1, 'Type', 'integer', 'LowerBound', 0, 'UpperBound', 1);
    zvar = optimvar('zvar', 1, 'LowerBound', 0);

    M = 3;  % Maximum number of assets in the portfolio
    m = 0;  % Minimum number of assets in the portfolio
    qpprob = optimproblem('ObjectiveSense', 'maximize');
    qpprob.Constraints.mconstr = sum(vvars) <= M;
    qpprob.Constraints.mconstr2 = sum(vvars) >= m;

    fmin = 0;
    fmax = 1;

    qpprob.Constraints.fmaxconstr = xvars <= fmax * vvars;
    qpprob.Constraints.fminconstr = fmin * vvars <= xvars;
    qpprob.Constraints.allin = sum(xvars) == 1;
    qpprob.Objective = mean_returns' * xvars - lambda * zvar;

    options = optimoptions(@intlinprog, 'Display', 'off'); % Suppress iterative display

    try
        % Initial solve
        [xLinInt, fval, exitflag, output] = solve(qpprob, 'options', options);

        if exitflag <= 0
            % If no solution is found, continue to the next lambda
            expected_returns(i) = NaN;
            risks(i) = NaN;
            continue;
        end

        thediff = 1e-4;
        iter = 1; % iteration counter
        assets = xLinInt.xvars;
        truequadratic = assets' * covariance_matrix * assets;
        zslack = xLinInt.zvar;
        
        while abs((zslack - truequadratic) / truequadratic) > thediff % relative error
            constr = 2 * assets' * covariance_matrix * xvars - zvar <= assets' * covariance_matrix * assets;
            newname = ['iteration', num2str(iter)];
            qpprob.Constraints.(newname) = constr;
            % Solve the problem with the new constraints
            [xLinInt, fval, exitflag, output] = solve(qpprob, 'options', options);
            if exitflag <= 0
                break;
            end
            assets = (assets + xLinInt.xvars) / 2; % Midway from the previous to the current
            truequadratic = xLinInt.xvars' * covariance_matrix * xLinInt.xvars;
            zslack = xLinInt.zvar;
            iter = iter + 1;
        end

        expected_returns(i) = mean_returns' * xLinInt.xvars;
        risks(i) = truequadratic;
    catch
        % If any error occurs, store NaN values
        expected_returns(i) = NaN;
        risks(i) = NaN;
    end
end

% Remove NaN values
expected_returns = expected_returns(~isnan(expected_returns));
risks = risks(~isnan(risks));

% Delete the parallel pool
delete(gcp);

% Plot the efficient frontier
figure;
plot(risks, expected_returns, 'b-o');
xlabel('Risk (Variance)');
ylabel('Expected Return');
title('Efficient Frontier');
grid on;





%{
RERCE-Fed Simulation

This script implements the RERCE-Fed (Resource-Efficient FL Robust to Communication Errors)

Key outputs:
- figure: Empirical result for a defined number of selected clients per iteration 
- Console output with detailed statistics

Based on methodology from:
- Elsevier Signal Processing paper: https://www.sciencedirect.com/science/article/pii/S0165168425005031

Author: Ehsan Lari
%}
%%

clear;
clc;

% rng(42);

%% Config
L = 128;     % Model length, default = 128
Nn = 100;    % Number of clients, default = 100
Monte = 1e2; % Number of Monte-Carlo simulations, default = 1e2
Iter = 5e2; % Number of iterations per simulation, default = 5e2
rho = 1; % Augmented Lagrange parameter, default = 1
sigma_u = sqrt(1e-2); sigma_d = sigma_u; % sqrt(6.25e-4); % Uplink and downlink noise variance parameter, default = 1e-2
Ncl = 4; % Number of clients per iteration, default = 4

%% Initialization

MSD_FedAvg = zeros(1, Iter);
msd_fedavg = zeros(1, Iter);
TimeDomain = zeros(L, Iter);
TimeDomainTotal = zeros(L, Iter);

Error_vec = cell(1,Monte);

X_BLUE1 = zeros(L, L); X_BLUE2 = zeros(L, 1); X = randn(L,1); X = X - mean(X); X = X/sqrt(sum(power(X,2)/L)); w_e = 0;

%% Data generation

Ns_train = load('Ns_train_uniform.mat', 'Ns_train');
Ns_train = Ns_train.Ns_train;

for k = 1:Nn
    mu(k) = -0.5 + rand(1); sigma2_h(k) = 0.5 + rand(1);
    H_train{k} = mu(k) + sqrt(sigma2_h(k)) * randn(Ns_train(k),L); y_train{k} = zeros(Ns_train(k),1);
    D_train_noise = randn(Ns_train(k),1); D_train_noise = D_train_noise - mean(D_train_noise); D_train_noise = D_train_noise/sqrt(sum(power(D_train_noise,2)/(Ns_train(k))));
    y_train{k} = H_train{k}*X + 0.01*D_train_noise;
    Sigma_mat{k} = 0.01*0.01*eye(Ns_train(k));
    X_BLUE1 = X_BLUE1 + (H_train{k})'*pinv(Sigma_mat{k})*(H_train{k});
    B{k} = pinv( (H_train{k})'*pinv(Sigma_mat{k})*(H_train{k}) + rho*eye(L) );
    X_BLUE2 = X_BLUE2 + (H_train{k})'*pinv(Sigma_mat{k})*(y_train{k});
    x_hat{k} = B{k}*(H_train{k})'*pinv(Sigma_mat{k})*(y_train{k});
end
X_BLUE1 = pinv(X_BLUE1);
X_BLUE = X_BLUE1*X_BLUE2;

%% Algorithm

for monte = 1:Monte % MC iteration

    monte

    x_local_noisy = zeros(L, Nn);   %xi  initialization

    for t = 1:Iter

        % Uniform Random Selection of Clients
        Pcl(:, 1)=[ones(Ncl,1); zeros(Nn-Ncl,1)];
        A_fedavg(:, 1)= Pcl(randperm(Nn), 1); a_fac(:,t) = A_fedavg;

        %%% ADMM-FedAvg
        if t == 1
            % Aggregation
            x_global_old = zeros(L,1);

            for k = 1:Nn

                x_local{k} = x_hat{k};

                if A_fedavg(k, 1) ~= 0

                    noise = randn(L,1);
                    x_local_noisy(:, k) = x_local{k} + 1*sigma_u*noise;

                end

            end

            x_global = (1/sum(A_fedavg)) * sum(kron(ones(L, 1), A_fedavg').*(x_local_noisy), 2);
            s_global = 2*x_global - x_global_old;


            % MSD
            for cnt_Nn = 1:Nn
                w_e = w_e + norm( x_local{cnt_Nn} - X_BLUE )^2;
            end

            msd_fedavg(1, t) = w_e; w_e = 0;
            TimeDomain(:,t) = x_global;

        else

            for k = 1:Nn

                if A_fedavg(k, 1) ~= 0

                    noise2 = randn(L,1);
                    s_global_local = s_global + 1*sigma_d*noise2;

                    x_local{k} = ( eye(L) - rho*B{k} ) * x_local{k} + ( rho*B{k} ) * ( s_global_local );

                    noise3 = randn(L,1);
                    x_local_noisy(:, k) = x_local{k} + 1*sigma_u*noise3;

                    s_global_local = zeros(L,1);

                end

            end

            x_global = (1/sum(A_fedavg)) * sum(kron(ones(L, 1), A_fedavg').*(x_local_noisy), 2);

            s_global = 2*x_global - x_global_old;
            x_global_old = x_global;

            % MSD
            for cnt_Nn = 1:Nn
                w_e = w_e + norm( x_local{cnt_Nn} - X_BLUE )^2;
            end

            msd_fedavg(1, t) = w_e; w_e = 0;

            TimeDomain(:,t) = x_global;

        end

        TimeDomainTotal(:,t) = TimeDomainTotal(:,t) + TimeDomain(:,t);

    end

    MSD_FedAvg = MSD_FedAvg + msd_fedavg;

    Error_vec{monte} = reshape(TimeDomain(:,Iter/2+1:Iter) - X_BLUE,[],1);

end

MSD_FedAvg = MSD_FedAvg/Monte/Nn/(norm(X_BLUE)^2);

figure;
set(groot,'defaultLineLineWidth',2.0);
set(0,'defaulttextinterpreter','latex');
set(gca,'TickLabelInterpreter','latex');
plot(1:Iter,10*log10(MSD_FedAvg(1:Iter)));
%{
RERCE-Fed with Continual Local Updates Simulation

This script implements the RERCE-Fed (Resource-Efficient FL Robust to Communication Errors) with Continual Local Updates

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
noise_var = 6.25e-4; % Uplink and downlink noise variance parameter, default = 1e-2
Ncl = 4; % Number of clients per iteration, default = 4

%% Initialization

MSD_FedAvg = zeros(1, Iter);
msd_fedavg = zeros(1, Iter);
TimeDomain = zeros(L, Iter);
TimeDomainTotal = zeros(L, Iter);

Error_vec = cell(1,Monte);

X_BLUE1 = zeros(L, L); X_BLUE2 = zeros(L, 1); w_e = 0; X = randn(L,1); X = X - mean(X); X = X/sqrt(sum(power(X,2)/L)); w_e = 0;

%% Data generation

Ns_train = load('Ns_train_uniform.mat', 'Ns_train');
Ns_train = Ns_train.Ns_train;

for k = 1:Nn
    mu(k) = -0.5+rand(1); sigma2_h(k) = 0.5+rand(1);
    H{k} = mu(k) + sqrt(sigma2_h(k))*randn(Ns_train(k),L); D_train{k} = zeros(Ns_train(k),1);
    D_train_noise = randn(Ns_train(k),1); D_train_noise = D_train_noise - mean(D_train_noise); D_train_noise = D_train_noise/sqrt(sum(power(D_train_noise,2)/(Ns_train(k))));
    D_train{k} = H{k}*X + 0.01*D_train_noise;
    Sigma_mat{k} = 0.01*0.01*eye(Ns_train(k));
    X_BLUE1 = X_BLUE1 + (H{k})'*pinv(Sigma_mat{k})*(H{k});
    B{k} = pinv(2*(H{k})'*pinv(Sigma_mat{k})*(H{k}) + rho*eye(L));
    X_BLUE2 = X_BLUE2 + (H{k})'*pinv(Sigma_mat{k})*(D_train{k});
    x_hat{k} = (B{k})*2*(H{k})'*pinv(Sigma_mat{k})*(D_train{k});
end
X_BLUE1 = pinv(X_BLUE1);
X_BLUE = X_BLUE1*X_BLUE2;

%% Algorithm

for monte = 1:Monte % MC iteration

    disp(monte)
    
    wt_loc_fedavg = zeros(L, Nn);   % w initialization
    wp_fedavg_new_local = zeros(L, Nn);   % w initialization

    for t = 1:Iter

        % Uniform Random Selection of Clients
        Pcl(:, 1)=[ones(Ncl,1); zeros(Nn-Ncl,1)];
        A_fedavg(:, 1)= Pcl(randperm(Nn), 1);
        % A_fedavg(:, 1)= ones(Nn,1);

        %%% ADMM-FedAvg
        if t==1

            % Aggregation

            wp_fedavg = zeros(L,1);

            for k = 1:Nn

                w_loc_fedavg{k} = x_hat{k};
                s_loc_fedavg{k} = 2*x_hat{k};

                wt_loc_fedavg(:, k) = s_loc_fedavg{k} + 1*sqrt(noise_var)*randn(L,1);


            end

            wp_fedavg = (1/Nn) * sum(wt_loc_fedavg, 2);

            % MSD
            for cnt_Nn = 1:Nn
                w_e = w_e + norm(w_loc_fedavg{cnt_Nn} - X_BLUE)^2;
            end

            msd_fedavg(1, t) = w_e; w_e = 0;

        else

            for k = 1:Nn

                if A_fedavg(k, 1)~=0

                    wp_fedavg_new_local(:, k) = wp_fedavg + 1*sqrt(noise_var)*randn(L,1);

                    w_loc_fedavg_new{k} = ( eye(L) - rho*(B{k}) ) * w_loc_fedavg{k} + ( rho*(B{k}) ) * ( wp_fedavg_new_local(:, k) );
                    s_loc_fedavg{k} = 2 * w_loc_fedavg_new{k} - w_loc_fedavg{k};
                    w_loc_fedavg{k} = w_loc_fedavg_new{k};

                    wt_loc_fedavg(:, k) = s_loc_fedavg{k} + 1*sqrt(noise_var)*randn(L,1);

                elseif A_fedavg(k, 1)==0

                    w_loc_fedavg_new{k} = ( eye(L) - rho*(B{k}) ) * w_loc_fedavg{k} + ( rho*(B{k}) ) * ( wp_fedavg_new_local(:, k) );
                    s_loc_fedavg{k} = 2 * w_loc_fedavg_new{k} - w_loc_fedavg{k};
                    w_loc_fedavg{k} = w_loc_fedavg_new{k};

                end

            end

            wp_fedavg = (1/Nn) * sum(wt_loc_fedavg, 2);

            % MSD
            for cnt_Nn = 1:Nn
                w_e = w_e + norm(w_loc_fedavg{cnt_Nn} - X_BLUE)^2;
            end

            msd_fedavg(1, t) = w_e; w_e = 0;

        end

    end

    MSD_FedAvg = MSD_FedAvg + msd_fedavg;

end

figure;
set(groot,'defaultLineLineWidth',2.0);
set(0,'defaulttextinterpreter','latex');
set(gca,'TickLabelInterpreter','latex');
plot(1:Iter,10*log10(MSD_FedAvg/Monte/Nn/(norm(X_BLUE)^2)));
% grid on;
xlabel('Iteration index (k)');
ylabel('NMSD (dB)');
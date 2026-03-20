function TMCMC_engine_inversion()
% TMCMC_engine_inversion  -- TMCMC Bayesian Posterior Inversion
%   for Aero-engine Thermodynamic Correction Parameters
%
%   Active inversion parameters (pre-selected sensitive parameters):
%     theta(1) = eta_k   -- 压气机绝热效率
%     theta(2) = eta_t   -- 涡轮绝热效率
%     theta(3) = eta_T   -- 燃烧放热系数
%     theta(4) = eta_m   -- 机械效率
%
%   Observations: specific thrust R_ud [N·s/kg] and
%                 specific fuel consumption C_ud [kg/(N·h)]
%
%   Algorithm: Transitional Markov Chain Monte Carlo (TMCMC)
%              Ching & Beck (2007), J. Eng. Mech.
%
%   Usage: TMCMC_engine_inversion()
%   (single file, no toolboxes required beyond base MATLAB)

clc; close all;

%% =====================================================================
%  1. PARAMETER BOUNDS AND TRUE VALUES
% ======================================================================
param_names = {'eta_k', 'eta_t', 'eta_T', 'eta_m'};
n_dim       = 4;

%  Prior: uniform on [lb, ub]
lb = [0.840, 0.860, 0.970, 0.980];
ub = [0.860, 0.920, 0.990, 0.995];

%  "True" values used to generate synthetic observations
theta_true = [0.850, 0.890, 0.980, 0.988];

%% =====================================================================
%  2. FIXED (NON-SENSITIVE) PARAMETERS AT NOMINAL VALUES
% ======================================================================
fp.eta_v      = 0.860;   % 风扇效率
fp.eta_tv     = 0.910;   % 风扇涡轮效率
fp.eta_c1     = 0.945;   % 一次喷管效率
fp.eta_c2     = 0.930;   % 二次喷管效率
fp.sigma_cc   = 0.990;   % 进气道激波总压恢复系数
fp.sigma_kan  = 0.985;   % 进气通道总压恢复系数
fp.sigma_kask = 0.985;   % 压气机级间总压恢复系数
fp.sigma_ks   = 0.950;   % 燃烧室总压恢复系数
fp.lambda     = 1.030;   % 风扇涡轮热恢复系数

%% =====================================================================
%  3. ENGINE OPERATING CONDITIONS
% ======================================================================
cond.T_H      = 288;    % 大气温度 [K]
cond.M_flight = 0.0;    % 飞行马赫数
cond.m        = 10.0;   % 涵道比
cond.pi_k     = 33.0;   % 压气机增压比
cond.T_g      = 1700.0; % 涡轮前燃气温度 [K]

%% =====================================================================
%  4. GENERATE SYNTHETIC OBSERVATIONS
% ======================================================================
[y_true, ~] = engine_fwd(theta_true, fp, cond);
assert(all(isfinite(y_true)), ...
    'Forward model at theta_true returns non-finite values. Check parameters.');

noise_frac = 0.001;                      % 0.1% relative observation noise
sigma_R    = noise_frac * y_true(1);
sigma_C    = noise_frac * y_true(2);

R_obs = y_true(1) + sigma_R * randn();
C_obs = y_true(2) + sigma_C * randn();

data.R_obs   = R_obs;    data.sigma_R = sigma_R;
data.C_obs   = C_obs;    data.sigma_C = sigma_C;
data.R_true  = y_true(1);
data.C_true  = y_true(2);

%% =====================================================================
%  5. PRINT SETUP
% ======================================================================
fprintf('==========================================\n');
fprintf('  TMCMC Engine Parameter Inversion\n');
fprintf('==========================================\n');
fprintf('Active parameters : %s\n\n', strjoin(param_names, ', '));
fprintf('%-8s   %8s   [%8s, %8s]\n', 'Param','True','lb','ub');
fprintf('%s\n', repmat('-', 1, 44));
for i = 1:n_dim
    fprintf('%-8s   %8.4f   [%8.4f, %8.4f]\n', ...
        param_names{i}, theta_true(i), lb(i), ub(i));
end
fprintf('\nSynthetic observations (0.1%% noise):\n');
fprintf('  R_ud = %.4f N·s/kg    (sigma = %.5f)\n', R_obs, sigma_R);
fprintf('  C_ud = %.6f kg/(N·h)  (sigma = %.7f)\n\n', C_obs, sigma_C);

%% =====================================================================
%  6. TMCMC SETTINGS
% ======================================================================
%  N      -- number of particles (1000 is light-weight yet reliable)
%  cov_tgt -- target COV of importance weights per stage (~1.0 standard)
%  n_mcmc  -- MH sub-chain length per particle (3-5 is typical)
%  s2      -- global scale for proposal covariance (0.04 = (0.2)^2)

tmcmc_opts.N        = 1000;
tmcmc_opts.cov_tgt  = 1.0;
tmcmc_opts.n_mcmc   = 4;
tmcmc_opts.s2       = 0.04;

%% =====================================================================
%  7. RUN TMCMC
% ======================================================================
fprintf('Running TMCMC  (N=%d particles) ...\n\n', tmcmc_opts.N);
[samples, log_post_vals, n_stages] = ...
    tmcmc_core(data, fp, cond, lb, ub, tmcmc_opts, n_dim);
fprintf('\nTMCMC finished.  Total stages: %d\n\n', n_stages);

%% =====================================================================
%  8. POSTERIOR STATISTICS
% ======================================================================
theta_mean = mean(samples, 1);
theta_std  = std(samples,  0, 1);

[~, map_idx] = max(log_post_vals);
theta_map    = samples(map_idx, :);

ci95 = zeros(n_dim, 2);
for i = 1:n_dim
    ci95(i, :) = quantile(samples(:, i), [0.025, 0.975]);
end

fprintf('%-8s  %8s  %8s  %8s  %8s  %8s  %8s\n', ...
    'Param','True','Mean','Std','MAP','CI_2.5%','CI_97.5%');
fprintf('%s\n', repmat('-', 1, 72));
for i = 1:n_dim
    fprintf('%-8s  %8.4f  %8.4f  %8.4f  %8.4f  %8.4f  %8.4f\n', ...
        param_names{i}, theta_true(i), theta_mean(i), theta_std(i), ...
        theta_map(i), ci95(i, 1), ci95(i, 2));
end
fprintf('\n');

%% =====================================================================
%  9. POSTERIOR PREDICTIVE CHECK
% ======================================================================
[y_mean, ~] = engine_fwd(theta_mean, fp, cond);
[y_map,  ~] = engine_fwd(theta_map,  fp, cond);

fprintf('Posterior predictive check:\n');
fprintf('  R_ud:  true=%.4f  obs=%.4f  pred(mean)=%.4f  pred(MAP)=%.4f\n', ...
    data.R_true, data.R_obs, y_mean(1), y_map(1));
fprintf('  C_ud:  true=%.6f  obs=%.6f  pred(mean)=%.6f  pred(MAP)=%.6f\n', ...
    data.C_true, data.C_obs, y_mean(2), y_map(2));
fprintf('\n');

%% =====================================================================
%  10. PLOT RESULTS
% ======================================================================
plot_posterior(samples, theta_true, theta_mean, theta_map, ...
    lb, ub, param_names, ci95);

end   % <<<  end of main function  >>>


%% ====================================================================
%%              LOCAL  FUNCTIONS
%% ====================================================================


% --------------------------------------------------------------------
%  TMCMC CORE
%  Implements the Transitional MCMC algorithm of Ching & Beck (2007).
%  Returns:
%    samples       -- N x n_dim matrix of posterior samples
%    log_post_vals -- N x 1 log-posterior values at samples
%    n_stages      -- number of tempering stages used
% --------------------------------------------------------------------
function [samples, log_post_vals, n_stages] = ...
        tmcmc_core(data, fp, cond, lb, ub, opts, n_dim)

N        = opts.N;
cov_tgt  = opts.cov_tgt;
n_mcmc   = opts.n_mcmc;
s2       = opts.s2;

% --- Stage 0: draw from uniform prior ---
samples = bsxfun(@plus, lb, ...
    bsxfun(@times, rand(N, n_dim), (ub - lb)));

% Evaluate log-likelihood for all particles
log_L = zeros(N, 1);
for i = 1:N
    log_L(i) = loglik(samples(i, :), data, fp, cond);
end

beta_curr = 0;
n_stages  = 0;

fprintf('  Stage   beta_prev ->  beta_new   COV_weights\n');
fprintf('  %s\n', repmat('-', 1, 48));

while beta_curr < 1.0

    % ------ (a) Find next beta via bisection ------
    %  Goal: COV of unnormalised weights w_i = L(theta_i)^{dBeta} equals cov_tgt
    beta_next = find_next_beta(log_L, beta_curr, cov_tgt);

    dBeta  = beta_next - beta_curr;
    log_w  = dBeta * log_L;
    log_w  = log_w - max(log_w);   % numerical stabilisation
    w      = exp(log_w);
    cov_w  = std(w) / (mean(w) + eps);

    fprintf('  %5d   %8.5f  ->  %8.5f    %.4f\n', ...
        n_stages + 1, beta_curr, beta_next, cov_w);

    % ------ (b) Normalised weights ------
    W = w / sum(w);

    % ------ (c) Weighted sample covariance for proposal ------
    mu_w = W' * samples;                          % 1 x n_dim
    diff = bsxfun(@minus, samples, mu_w);
    Cov  = zeros(n_dim, n_dim);
    for i = 1:N
        Cov = Cov + W(i) * (diff(i, :)' * diff(i, :));
    end
    Cov = s2 * Cov + 1e-8 * eye(n_dim);          % scale + regularise

    [L_chol, flag] = chol(Cov, 'lower');
    if flag ~= 0                                   % fallback: diagonal
        L_chol = diag(sqrt(s2) * (std(samples) + 1e-6));
    end

    % ------ (d) Resample using systematic resampling ------
    idx         = systematic_resample(W, N);
    samples_new = samples(idx, :);
    log_L_new   = log_L(idx);

    % ------ (e) MCMC perturbation with tempered target p^{beta_next} ------
    for i = 1:N
        th_curr   = samples_new(i, :);
        lL_curr   = log_L_new(i);
        lp_curr   = beta_next * lL_curr;    % log_prior = 0 inside bounds

        for k = 1:n_mcmc
            th_prop = th_curr + (L_chol * randn(n_dim, 1))';
            th_prop = reflect_bounds(th_prop, lb, ub);

            lL_prop = loglik(th_prop, data, fp, cond);
            if all(th_prop >= lb) && all(th_prop <= ub)
                lp_prop = beta_next * lL_prop;
            else
                lp_prop = -Inf;
            end

            if log(rand()) < (lp_prop - lp_curr)
                th_curr  = th_prop;
                lL_curr  = lL_prop;
                lp_curr  = lp_prop;
            end
        end

        samples_new(i, :) = th_curr;
        log_L_new(i)      = lL_curr;
    end

    samples   = samples_new;
    log_L     = log_L_new;
    beta_curr = beta_next;
    n_stages  = n_stages + 1;
end

% Compute final log-posterior for each particle
log_post_vals = zeros(N, 1);
for i = 1:N
    log_post_vals(i) = loglik(samples(i, :), data, fp, cond);
end
end


% --------------------------------------------------------------------
%  FIND NEXT BETA
%  Bisection to find dBeta such that COV of w_i = L^{dBeta} = cov_tgt
% --------------------------------------------------------------------
function beta_next = find_next_beta(log_L, beta_curr, cov_tgt)

    function cov_w = compute_cov(dBeta)
        lw = dBeta * log_L;
        lw = lw - max(lw);
        w  = exp(lw);
        cov_w = std(w) / (mean(w) + eps);
    end

% Check if going all the way to beta=1 is safe
cov_full = compute_cov(1.0 - beta_curr);
if cov_full <= cov_tgt
    beta_next = 1.0;
    return;
end

% Bisection on dBeta in [0, 1-beta_curr]
lo = 0;
hi = 1.0 - beta_curr;
for iter = 1:60
    mid = 0.5 * (lo + hi);
    if compute_cov(mid) > cov_tgt
        hi = mid;
    else
        lo = mid;
    end
    if (hi - lo) < 1e-7
        break;
    end
end
beta_next = min(beta_curr + lo, 1.0);   % conservative: use lo side
end


% --------------------------------------------------------------------
%  SYSTEMATIC RESAMPLING  (low-variance resampler)
% --------------------------------------------------------------------
function idx = systematic_resample(W, N)
cumW    = cumsum(W(:));
cumW(end) = 1.0;
u0      = rand() / N;
u       = u0 + (0 : N-1)' / N;
idx     = zeros(N, 1);
j       = 1;
for i = 1:N
    while j < length(cumW) && u(i) > cumW(j)
        j = j + 1;
    end
    idx(i) = j;
end
end


% --------------------------------------------------------------------
%  REFLECT INTO PARAMETER BOUNDS
% --------------------------------------------------------------------
function theta = reflect_bounds(theta, lb, ub)
for i = 1:length(theta)
    lo = lb(i);  hi = ub(i);  rng_i = hi - lo;
    v  = (theta(i) - lo) / rng_i;   % map to [0,1]
    n_iter = 0;
    while (v < 0 || v > 1) && n_iter < 10
        if v < 0,   v = -v;      end
        if v > 1,   v = 2 - v;   end
        n_iter = n_iter + 1;
    end
    theta(i) = lo + max(0, min(1, v)) * rng_i;
end
end


% --------------------------------------------------------------------
%  LOG-LIKELIHOOD
%  Gaussian likelihood for two observations (R_ud, C_ud)
% --------------------------------------------------------------------
function ll = loglik(theta, data, fp, cond)
ll = -Inf;
[y, ~] = engine_fwd(theta, fp, cond);
if ~all(isfinite(y)), return; end
res_R = (data.R_obs - y(1)) / data.sigma_R;
res_C = (data.C_obs - y(2)) / data.sigma_C;
ll    = -0.5 * (res_R^2 + res_C^2);
%  Constant terms -0.5*log(2*pi*sigma^2) omitted (cancel in MH ratio)
end


% --------------------------------------------------------------------
%  ENGINE FORWARD MODEL  (4 active parameters)
%
%  theta = [eta_k, eta_t, eta_T, eta_m]
%  fp    = struct of fixed (non-sensitive) parameters
%  cond  = struct of operating conditions
%
%  Returns y = [R_ud, C_ud]  or  [NaN, NaN] if physics fails.
% --------------------------------------------------------------------
function [y, aux] = engine_fwd(theta, fp, cond)

eta_k = theta(1);   % 压气机绝热效率
eta_t = theta(2);   % 涡轮绝热效率
eta_T = theta(3);   % 燃烧放热系数
eta_m = theta(4);   % 机械效率

eta_v      = fp.eta_v;
eta_tv     = fp.eta_tv;
eta_c1     = fp.eta_c1;
eta_c2     = fp.eta_c2;
sigma_cc   = fp.sigma_cc;
sigma_kan  = fp.sigma_kan;
sigma_kask = fp.sigma_kask;
sigma_ks   = fp.sigma_ks;
lambda     = fp.lambda;

T_H      = cond.T_H;
M_flight = cond.M_flight;
m        = cond.m;
pi_k     = cond.pi_k;
T_g      = cond.T_g;

y   = [NaN, NaN];
aux = struct();

try
    % ---- Constants ----
    k_air = 1.4;
    R_air = 287.3;
    kT    = piecewise_kT(T_g);
    RT    = piecewise_RT(T_g);
    d     = delta_cooling(T_g);

    % ---- Flight velocity ----
    V_flight = sqrt(k_air * R_air * T_H) * M_flight;

    % ---- Ram pressure ratio and inlet stagnation temperature ----
    inner = 1 + V_flight^2 / (2 * (k_air / (k_air - 1)) * R_air * T_H + eps);
    tau_v = inner^(k_air / (k_air - 1));
    T_B   = T_H * inner^k_air;
    if ~isfinite(T_B) || T_B <= 0, return; end

    % ---- Compressor exit temperature ----
    pi_k_rat = pi_k^((k_air - 1) / k_air);
    T_k = T_B * (1 + (pi_k_rat - 1) / eta_k);
    if ~isfinite(T_k) || T_k <= 0, return; end

    % ---- Relative fuel-air ratio ----
    g_T = 3e-5 * T_g - 2.69e-5 * T_k - 0.003;
    if ~isfinite(g_T) || g_T <= 0, return; end

    % ---- Thermal enthalpy balance -> lambda_heat ----
    compress_work = (k_air / (k_air - 1)) * R_air * T_B * (pi_k_rat - 1);
    gas_enthalpy  = (kT / (kT - 1)) * RT * T_g;
    if abs(gas_enthalpy) < 1e-6, return; end

    num_lh      = 1 - compress_work / (gas_enthalpy * eta_k);
    den_lh      = 1 - compress_work / (gas_enthalpy * eta_k * eta_t);
    if abs(den_lh) < 1e-10, return; end
    lambda_heat = num_lh / den_lh;
    if ~isfinite(lambda_heat), return; end

    % ---- Turbine expansion term -> specific free energy L_sv ----
    sigma_bx  = sigma_cc * sigma_kan;
    exp_T     = (kT - 1) / kT;
    ep_denom  = tau_v * sigma_bx * pi_k * sigma_kask * sigma_ks;
    if ep_denom <= 0, return; end

    expansion_term = (1 / ep_denom)^exp_T;
    if ~isfinite(expansion_term), return; end

    term1  = (kT / (kT - 1)) * RT * T_g * (1 - expansion_term);
    denom2 = (1 + g_T) * eta_k * eta_T * eta_t * eta_m * (1 - d);
    if abs(denom2) < 1e-10, return; end
    term2  = compress_work / denom2;

    L_sv = lambda_heat * (term1 - term2);
    if ~isfinite(L_sv) || L_sv <= 0, return; end

    % ---- Optimal power-split coefficient x_pc ----
    num_xpc = 1 + m * V_flight^2 / ...
              (2 * L_sv * eta_tv * eta_v * eta_c2 + eps);
    den_xpc = 1 + (m * eta_tv * eta_v * eta_c2) / (eta_c1 * lambda);
    if abs(den_xpc) < 1e-10, return; end
    x_pc = num_xpc / den_xpc;
    if ~isfinite(x_pc) || x_pc <= 0, return; end

    % ---- Nozzle exit velocities ----
    sq1 = 2 * eta_c1 * lambda * x_pc * L_sv;
    if sq1 < 0, return; end
    V_j1 = (1 + g_T) * sqrt(sq1) - V_flight;

    sq2 = 2 * (1 - x_pc) / m * L_sv * eta_tv * eta_v * eta_c2 + V_flight^2;
    if sq2 < 0, return; end
    V_j2 = sqrt(sq2) - V_flight;

    % ---- Specific thrust ----
    R_ud = (1 / (1 + m)) * V_j1 + (m / (1 + m)) * V_j2;
    if ~isfinite(R_ud) || R_ud <= 0, return; end

    % ---- Specific fuel consumption ----
    C_ud = 3600 * g_T * (1 - d) / (R_ud * (1 + m));
    if ~isfinite(C_ud) || C_ud <= 0, return; end

    % ---- Outputs ----
    y = [R_ud, C_ud];

    aux.T_B         = T_B;
    aux.T_k         = T_k;
    aux.g_T         = g_T;
    aux.lambda_heat = lambda_heat;
    aux.L_sv        = L_sv;
    aux.x_pc        = x_pc;

catch
    % Silent fail -- y remains [NaN, NaN]
end
end


% ---- Piecewise adiabatic index of combustion gas ----
function kT = piecewise_kT(T_g)
if     T_g > 1600,  kT = 1.25;
elseif T_g > 1400,  kT = 1.30;
else,               kT = 1.33;
end
end

% ---- Piecewise gas constant of combustion gas ----
function RT = piecewise_RT(T_g)
if     T_g > 1600,  RT = 288.6;
elseif T_g > 1400,  RT = 288.0;
else,               RT = 287.6;
end
end

% ---- Turbine cooling bleed fraction ----
function d = delta_cooling(T_g)
d = max(0.0, min(0.15, 0.02 + (T_g - 1200) / 100 * 0.02));
end


% --------------------------------------------------------------------
%  PLOTTING
%   Figure 1 -- Marginal posterior distributions (2x2 grid)
%   Figure 2 -- Pairwise joint posteriors (4x4 grid)
% --------------------------------------------------------------------
function plot_posterior(samples, theta_true, theta_mean, theta_map, ...
                        lb, ub, param_names, ci95)

n_dim = size(samples, 2);

%% ---- Figure 1: Marginal posteriors ----
fig1 = figure('Name', 'Marginal Posterior Distributions', ...
              'NumberTitle', 'off', ...
              'Position', [60, 60, 880, 680]);

colors_hist = [0.30, 0.58, 0.90];
for i = 1:n_dim
    subplot(2, 2, i);
    histogram(samples(:, i), 40, ...
        'Normalization', 'pdf', ...
        'FaceColor', colors_hist, ...
        'EdgeColor', 'none', ...
        'FaceAlpha', 0.78);
    hold on;
    xline(theta_true(i), 'g-',  'LineWidth', 2.2, 'DisplayName', 'True value');
    xline(theta_mean(i), 'r--', 'LineWidth', 1.8, 'DisplayName', 'Post. mean');
    xline(theta_map(i),  'm:',  'LineWidth', 1.8, 'DisplayName', 'MAP');
    xline(ci95(i, 1),    'k-',  'LineWidth', 0.8, 'Alpha', 0.6, 'DisplayName', '95% CI');
    xline(ci95(i, 2),    'k-',  'LineWidth', 0.8, 'Alpha', 0.6, 'HandleVisibility', 'off');

    xlim([lb(i), ub(i)]);
    xlabel(strrep(param_names{i}, '_', '\_'), 'FontSize', 11);
    ylabel('PDF', 'FontSize', 10);
    title(sprintf('%s\ntrue=%.4f,  mean=%.4f,  MAP=%.4f', ...
        strrep(param_names{i}, '_', '\_'), ...
        theta_true(i), theta_mean(i), theta_map(i)), ...
        'FontSize', 10);
    if i == 1
        legend('Location', 'best', 'FontSize', 9);
    end
    grid on;  box on;
end
sgtitle('Marginal Posterior Distributions  (TMCMC)', 'FontSize', 13);

try
    saveas(fig1, 'TMCMC_marginal_posteriors.png');
    fprintf('Figure saved: TMCMC_marginal_posteriors.png\n');
catch
    fprintf('(Could not save Figure 1 to file.)\n');
end

%% ---- Figure 2: Pairwise joint posteriors ----
fig2 = figure('Name', 'Pairwise Joint Posterior Distributions', ...
              'NumberTitle', 'off', ...
              'Position', [100, 50, 900, 880]);

for i = 1:n_dim
    for j = 1:n_dim
        ax = subplot(n_dim, n_dim, (i-1)*n_dim + j);

        if i == j
            % Diagonal: marginal histogram
            histogram(samples(:, i), 30, ...
                'Normalization', 'pdf', ...
                'FaceColor', [0.30, 0.58, 0.90], ...
                'EdgeColor', 'none', 'FaceAlpha', 0.80);
            hold on;
            xline(theta_true(i), 'g-', 'LineWidth', 1.8);
            xlim([lb(i), ub(i)]);
        else
            % Off-diagonal: 2-D scatter
            scatter(samples(:, j), samples(:, i), 3, ...
                [0.25, 0.48, 0.78], 'filled', 'MarkerFaceAlpha', 0.25);
            hold on;
            % True parameter location
            plot(theta_true(j), theta_true(i), 'g+', ...
                'MarkerSize', 11, 'LineWidth', 2.0);
            % 2-D kernel density contour (approximate using hist2)
            try
                n_bins2d = 30;
                x_edges = linspace(lb(j), ub(j), n_bins2d + 1);
                y_edges = linspace(lb(i), ub(i), n_bins2d + 1);
                [counts, ~, ~] = histcounts2(samples(:,j), samples(:,i), ...
                    x_edges, y_edges);
                xc = 0.5*(x_edges(1:end-1) + x_edges(2:end));
                yc = 0.5*(y_edges(1:end-1) + y_edges(2:end));
                contour(xc, yc, counts', 4, 'LineColor', [0.1, 0.3, 0.7], ...
                    'LineWidth', 0.8);
            catch
                % histcounts2 not available in older MATLAB -- skip contour
            end
            xlim([lb(j), ub(j)]);
            ylim([lb(i), ub(i)]);
        end

        % Axis labels on borders only
        if i == n_dim
            xlabel(strrep(param_names{j}, '_', '\_'), 'FontSize', 9);
        end
        if j == 1
            ylabel(strrep(param_names{i}, '_', '\_'), 'FontSize', 9);
        end
        set(ax, 'FontSize', 7, 'TickLength', [0.02, 0.02]);
    end
end
sgtitle('Pairwise Joint Posterior Distributions  (TMCMC)', 'FontSize', 13);

try
    saveas(fig2, 'TMCMC_pairwise_posteriors.png');
    fprintf('Figure saved: TMCMC_pairwise_posteriors.png\n');
catch
    fprintf('(Could not save Figure 2 to file.)\n');
end

end   % end plot_posterior

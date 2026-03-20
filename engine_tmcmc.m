param_names = {
    'eta_k',       ...  % 压气机绝热效率
    'eta_t',       ...  % 涡轮绝热效率
    'eta_m',       ...  % 机械效率
    'eta_v',       ...  % 风扇效率
    'eta_tv',      ...  % 风扇涡轮效率
    'eta_c1',      ...  % 一次喷管效率
    'eta_c2',      ...  % 二次喷管效率
    'sigma_cc',    ...  % 进气道激波总压恢复系数
    'sigma_kan',   ...  % 进气通道总压恢复系数
    'sigma_kask',  ...  % 压气机级间总压恢复系数
    'sigma_ks',    ...  % 燃烧室总压恢复系数
    'eta_T',       ...  % 燃烧放热系数
    'lambda'        ...  % 风扇涡轮热恢复系数
};

lb = [0.84, 0.86, 0.980, 0.85, 0.90, 0.94, 0.92, 0.98, 0.98, 0.98, 0.94, 0.97, 1.02];
ub = [0.86, 0.92, 0.995, 0.87, 0.92, 0.95, 0.94, 1.00, 0.99, 0.99, 0.96, 0.99, 1.04];

theta_true = [
    0.85,    ...  % eta_k
    0.89,    ...  % eta_t
    0.988,   ...  % eta_m
    0.860,   ...  % eta_v
    0.910,   ...  % eta_tv
    0.945,   ...  % eta_c1
    0.930,   ...  % eta_c2
    0.990,   ...  % sigma_cc
    0.985,   ...  % sigma_kan
    0.985,   ...  % sigma_kask
    0.950,   ...  % sigma_ks
    0.980,   ...  % eta_T
    1.030    ...  % lambda
];

cond.T_H       = 288;
cond.M_flight  = 0.0;
cond.m         = 10.0;
cond.pi_k      = 33.0;
cond.T_g       = 1700.0;

N          = 500;
COV_target = 1.0;
n_steps    = 3;
scale      = 0.2;

disp('参数定义完成')

% 测试：用 theta_true 调用前向模型
[y_test, aux_test] = engine_forward(theta_true, cond);
fprintf('engine_forward 测试结果：\n');
fprintf('  R_ud = %.4f [N·s/kg]\n', y_test(1));
fprintf('  C_ud = %.6f [kg/(N·h)]\n', y_test(2));
fprintf('  T_B  = %.2f K\n', aux_test.T_B);
fprintf('  T_k  = %.2f K\n', aux_test.T_k);
fprintf('  g_T  = %.6f\n',   aux_test.g_T);
fprintf('  lambda_heat = %.4f\n', aux_test.lambda_heat);
fprintf('  L_sv = %.2f J/kg\n', aux_test.L_sv);
fprintf('  x_pc = %.4f\n',   aux_test.x_pc);

% ── 步骤1：生成虚拟观测数据 ──────────────────────────────────────────
noise_level = 0.001;
rng(42);
y_true    = y_test;                           % [R_ud_true, C_ud_true]
sigma_obs = noise_level * abs(y_true);        % [sigma_R, sigma_C]
y_obs     = y_true + sigma_obs .* randn(1, 2);
fprintf('\n【步骤1】生成虚拟观测数据\n');
fprintf('  R_true = %.6f,  R_obs = %.6f,  sigma_R = %.8f\n', y_true(1), y_obs(1), sigma_obs(1));
fprintf('  C_true = %.8f,  C_obs = %.8f,  sigma_C = %.10f\n', y_true(2), y_obs(2), sigma_obs(2));

% ── 步骤2：运行 TMCMC ────────────────────────────────────────────────
fprintf('\n【步骤2】运行 TMCMC（N=%d, COV_target=%.1f, n_steps=%d, scale=%.2f）\n', ...
        N, COV_target, n_steps, scale);
t_start = tic;
[samples, beta_hist, ess_hist, acc_hist] = ...
    run_tmcmc(y_obs, sigma_obs, lb, ub, cond, N, COV_target, n_steps, scale);
t_elapsed = toc(t_start);
fprintf('TMCMC 完成，共 %d 层，耗时 %.1f 秒\n', length(acc_hist), t_elapsed);

% ── 步骤3：计算 MAP 估计 ──────────────────────────────────────────────
fprintf('\n【步骤3】计算 MAP 估计\n');
theta_map  = compute_map(samples, y_obs, sigma_obs, lb, ub, cond);
theta_mean = mean(samples, 1);

% ── 步骤4：打印后验统计（每参数一行）────────────────────────────────
n_params = length(lb);
ci95     = zeros(n_params, 2);
for i = 1:n_params
    ci95(i, :) = quantile(samples(:, i), [0.025, 0.975]);
end

fprintf('\n【步骤4】后验统计结果\n');
fprintf('%-14s  %8s  %8s  %8s  %8s  %8s\n', ...
        '参数', '真值', '后验均值', 'MAP', 'CI95下', 'CI95上');
fprintf('%s\n', repmat('-', 1, 68));
for i = 1:n_params
    fprintf('%-14s  %8.4f  %8.4f  %8.4f  %8.4f  %8.4f\n', ...
            param_names{i}, theta_true(i), theta_mean(i), theta_map(i), ...
            ci95(i,1), ci95(i,2));
end

% 后验预测对比
[y_mean_pred, ~] = engine_forward(theta_mean, cond);
[y_map_pred,  ~] = engine_forward(theta_map,  cond);
fprintf('\n后验预测对比：\n');
fprintf('  %-18s  %12s  %12s\n', '来源', 'R_ud [N·s/kg]', 'C_ud [kg/(N·h)]');
fprintf('  %-18s  %12.6f  %12.8f\n', '真值',         y_true(1),       y_true(2));
fprintf('  %-18s  %12.6f  %12.8f\n', '观测值',       y_obs(1),        y_obs(2));
fprintf('  %-18s  %12.6f  %12.8f\n', '后验均值预测', y_mean_pred(1),  y_mean_pred(2));
fprintf('  %-18s  %12.6f  %12.8f\n', 'MAP预测',      y_map_pred(1),   y_map_pred(2));

% ── 步骤5：绘图 ──────────────────────────────────────────────────────
fprintf('\n【步骤5】绘图\n');
plot_results_tmcmc(samples, theta_true, param_names, beta_hist, ess_hist, acc_hist);
plot_pairwise_posterior(samples, param_names);
fprintf('图形已生成并保存\n');

% ── 步骤6：保存结果 ───────────────────────────────────────────────────
fprintf('\n【步骤6】保存结果到 tmcmc_results.mat\n');
save('tmcmc_results.mat', 'samples', 'theta_map', 'theta_mean', 'ci95', ...
     'beta_hist', 'ess_hist', 'acc_hist', 'theta_true', 'y_obs', 'sigma_obs', ...
     'param_names', 'lb', 'ub', 'cond');
fprintf('保存完成：tmcmc_results.mat\n');
fprintf('\n全部流程完成。\n');

% -------------------------------------------------------------------------
% 局部函数
% -------------------------------------------------------------------------

function kT = piecewise_kT(T_g)
% 燃气绝热指数（分段）
if T_g > 800 && T_g <= 1400
    kT = 1.33;
elseif T_g > 1400 && T_g <= 1600
    kT = 1.30;
elseif T_g > 1600
    kT = 1.25;
else
    kT = 1.33;
end
end

function RT = piecewise_RT(T_g)
% 燃气气体常数（分段）
if T_g > 800 && T_g <= 1400
    RT = 287.6;
elseif T_g > 1400 && T_g <= 1600
    RT = 288.0;
elseif T_g > 1600
    RT = 288.6;
else
    RT = 287.6;
end
end

function d = delta_cooling(T_g)
% 涡轮冷却引气系数
d = 0.02 + (T_g - 1200) / 100 * 0.02;
d = max(0.0, min(d, 0.15));
end

function [y, aux] = engine_forward(theta, cond)
% 发动机前向模型
% 输入：theta(13) = [eta_k, eta_t, eta_m, eta_v, eta_tv,
%                    eta_c1, eta_c2, sigma_cc, sigma_kan, sigma_kask,
%                    sigma_ks, eta_T, lambda]
%       cond 结构体：T_H, M_flight, m, pi_k, T_g
% 输出：y = [R_ud, C_ud]，失效时返回 [NaN, NaN]

eta_k      = theta(1);
eta_t      = theta(2);
eta_m      = theta(3);
eta_v      = theta(4);
eta_tv     = theta(5);
eta_c1     = theta(6);
eta_c2     = theta(7);
sigma_cc   = theta(8);
sigma_kan  = theta(9);
sigma_kask = theta(10);
sigma_ks   = theta(11);
eta_T      = theta(12);
lambda     = theta(13);

T_H      = cond.T_H;
M_flight = cond.M_flight;
m        = cond.m;
pi_k     = cond.pi_k;
T_g      = cond.T_g;

y   = [NaN, NaN];
aux = struct();

try
    % (1) 基本常数
    k_air = 1.4;
    R_air = 287.3;
    a = sqrt(k_air * R_air * T_H);
    if ~isfinite(a) || a <= 0, return; end
    V_flight = a * M_flight;

    % (2) 分段函数值
    kT = piecewise_kT(T_g);
    RT = piecewise_RT(T_g);
    d  = delta_cooling(T_g);

    % (3) 进口总压比与压气机入口温度
    inner = 1 + V_flight^2 / (2 * (k_air/(k_air-1)) * R_air * T_H);
    if inner <= 0, return; end
    tau_v = inner^(k_air / (k_air - 1));
    T_B   = T_H * (inner^k_air);
    if ~isfinite(T_B) || T_B <= 0, return; end

    % (4) 压气机出口温度
    pi_k_ratio = pi_k^((k_air-1)/k_air);
    if ~isfinite(pi_k_ratio) || pi_k_ratio < 1, return; end
    T_k = T_B * (1 + (pi_k_ratio - 1) / eta_k);
    if ~isfinite(T_k) || T_k <= 0, return; end

    % (5) 相对耗油量
    g_T = 3e-5 * T_g - 2.69e-5 * T_k - 0.003;
    if ~isfinite(g_T) || g_T <= 0, return; end

    % (6) 热恢复系数
    compress_work = (k_air/(k_air-1)) * R_air * T_B * (pi_k_ratio - 1);
    gas_enthalpy  = (kT/(kT-1)) * RT * T_g;
    if abs(gas_enthalpy) < 1e-6, return; end

    num_lambda = 1 - compress_work / (gas_enthalpy * eta_k);
    den_lambda = 1 - compress_work / (gas_enthalpy * eta_k * eta_t);
    if abs(den_lambda) < 1e-10, return; end
    lambda_heat = num_lambda / den_lambda;
    if ~isfinite(lambda_heat), return; end

    % (7) 进口总压恢复系数
    sigma_bx = sigma_cc * sigma_kan;

    % (8) 单位自由能
    exp_T = (kT - 1) / kT;
    expansion_pr_denom = tau_v * sigma_bx * pi_k * sigma_kask * sigma_ks;
    if expansion_pr_denom <= 0, return; end
    expansion_term = (1.0 / expansion_pr_denom)^exp_T;
    if ~isfinite(expansion_term), return; end
    term1 = (kT / (kT - 1)) * RT * T_g * (1 - expansion_term);
    compress_work = (k_air / (k_air - 1)) * R_air * T_B * (pi_k^((k_air-1)/k_air) - 1);
    denom2 = (1 + g_T) * eta_k * eta_T * eta_t * eta_m * (1 - d);
    if abs(denom2) < 1e-10, return; end
    term2 = compress_work / denom2;
    L_sv  = lambda_heat * (term1 - term2);
    if ~isfinite(L_sv) || L_sv <= 0, return; end

    % (9) 最优自由能分配系数
    V2_term  = m * V_flight^2;
    num_xpc  = 1 + V2_term / (2 * L_sv * eta_tv * eta_v * eta_c2);
    den_xpc  = 1 + (m * eta_tv * eta_v * eta_c2) / (eta_c1 * lambda);
    if abs(den_xpc) < 1e-10, return; end
    x_pc = num_xpc / den_xpc;
    if ~isfinite(x_pc) || x_pc <= 0, return; end

    % (10) 比推力
    inner_sq1 = 2 * eta_c1 * lambda * x_pc * L_sv;
    if inner_sq1 < 0, return; end
    V_j1 = (1 + g_T) * sqrt(inner_sq1) - V_flight;
    inner_sq2 = 2 * (1 - x_pc) / m * L_sv * eta_tv * eta_v * eta_c2 + V_flight^2;
    if inner_sq2 < 0, return; end
    V_j2 = sqrt(inner_sq2) - V_flight;
    R_ud = (1/(1+m)) * V_j1 + (m/(1+m)) * V_j2;
    if ~isfinite(R_ud) || R_ud <= 0, return; end

    % (11) 比油耗
    denom_C = R_ud * (1 + m);
    if abs(denom_C) < 1e-10, return; end
    C_ud = 3600 * g_T * (1 - d) / denom_C;
    if ~isfinite(C_ud) || C_ud <= 0, return; end

    y = [R_ud, C_ud];

    aux.T_B         = T_B;
    aux.T_k         = T_k;
    aux.tau_v       = tau_v;
    aux.g_T         = g_T;
    aux.lambda_heat = lambda_heat;
    aux.sigma_bx    = sigma_bx;
    aux.L_sv        = L_sv;
    aux.x_pc        = x_pc;
    aux.kT          = kT;
    aux.RT          = RT;
    aux.delta       = d;

catch ME
    warning('engine_forward caught: %s', ME.message);
end
end

function lp = log_prior(theta, lb, ub)
% 均匀先验：参数在 [lb, ub] 内返回 0，否则返回 -Inf
if all(theta >= lb) && all(theta <= ub)
    lp = 0.0;
else
    lp = -Inf;
end
end

function ll = log_likelihood(theta, y_obs, sigma_obs, cond)
% 两个观测量的高斯对数似然
% y_obs(1)=R_obs, y_obs(2)=C_obs；sigma_obs 同维
ll = -Inf;
[y_pred, ~] = engine_forward(theta, cond);
if ~all(isfinite(y_pred))
    return;
end
if any(sigma_obs <= 0)
    return;
end
res = (y_obs - y_pred) ./ sigma_obs;
ll  = -0.5 * sum(res.^2 + log(2 * pi * sigma_obs.^2));
if ~isfinite(ll)
    ll = -Inf;
end
end

function lpost = log_posterior_fn(theta, y_obs, sigma_obs, lb, ub, cond)
% 对数后验 = 对数先验 + 对数似然
lp = log_prior(theta, lb, ub);
if isinf(lp)
    lpost = -Inf;
    return;
end
ll    = log_likelihood(theta, y_obs, sigma_obs, cond);
lpost = lp + ll;
end

function [samples, beta_hist, ess_hist, acc_hist] = run_tmcmc( ...
        y_obs, sigma_obs, lb, ub, cond, N, COV_target, n_steps, scale)
% TMCMC (Transitional Markov Chain Monte Carlo)
%
% 输入：
%   y_obs      - 观测值向量 [R_obs, C_obs]
%   sigma_obs  - 观测噪声  [sigma_R, sigma_C]
%   lb, ub     - 参数下/上界（1×13）
%   cond       - 工况结构体
%   N          - 粒子数
%   COV_target - 权重变异系数目标阈值（典型值 1.0）
%   n_steps    - 每粒子 MH 短链步数
%   scale      - 提议协方差缩放因子
%
% 输出：
%   samples    - 最终样本矩阵（N×13）
%   beta_hist  - 每级 beta 值历史
%   ess_hist   - 每级有效样本量历史
%   acc_hist   - 每级 MH 平均接受率历史

n_params = length(lb);

% ── 步骤 1：从均匀先验采 N 个初始样本 ─────────────────────────────────
samples  = bsxfun(@plus, lb, bsxfun(@times, rand(N, n_params), ub - lb));

% 计算每个粒子的对数似然（初始时 beta=0，只需先算好备用）
logL = zeros(N, 1);
for i = 1:N
    logL(i) = log_likelihood(samples(i,:), y_obs, sigma_obs, cond);
end

beta      = 0;
beta_hist = 0;
ess_hist  = N;        % 初始 ESS = N（均匀权重）
acc_hist  = NaN;      % 第0级无 MH 步

% ── 主循环：beta 从 0 推进到 1 ────────────────────────────────────────
stage = 0;
while beta < 1

    stage = stage + 1;

    % ── (a) 二分法确定 delta_beta ─────────────────────────────────────
    % 目标：COV(w) = std(w)/mean(w) ≤ COV_target
    % w_i = exp(delta_beta * logL_i)，对数稳定版本减去最大值
    db_lo = 0;
    db_hi = 1 - beta;

    % 先检查上界是否已满足 COV 约束
    cov_hi = compute_cov_weights(logL, db_hi);
    if cov_hi <= COV_target
        % 直接走到 beta=1
        delta_beta = db_hi;
    else
        % 二分搜索
        for bisect_iter = 1:50
            db_mid  = 0.5 * (db_lo + db_hi);
            cov_mid = compute_cov_weights(logL, db_mid);
            if cov_mid < COV_target
                db_lo = db_mid;
            else
                db_hi = db_mid;
            end
            if (db_hi - db_lo) < 1e-8
                break;
            end
        end
        delta_beta = db_lo;   % 保守侧：COV 不超标
        if delta_beta < 1e-10
            % 防止步长退化为零：强制最小步
            delta_beta = 1e-6;
        end
    end

    % 确保不超过 1
    delta_beta = min(delta_beta, 1 - beta);
    beta       = beta + delta_beta;

    % ── (b) 计算归一化权重与 ESS ──────────────────────────────────────
    log_w    = delta_beta * logL;
    log_w    = log_w - max(log_w);          % 数值稳定
    w        = exp(log_w);
    w_norm   = w / sum(w);
    ESS      = 1 / sum(w_norm.^2);

    % ── (c) 残差重采样 ────────────────────────────────────────────────
    idx      = residual_resample(w_norm, N);
    samples  = samples(idx, :);
    logL     = logL(idx);

    % ── (d) 协方差自适应 MH 短链更新 ─────────────────────────────────
    % 提议协方差：scale^2 × 当前样本经验协方差
    S        = scale^2 * cov(samples);
    % 保证正定：加小对角扰动
    S        = S + 1e-10 * eye(n_params);

    % Cholesky 分解用于高效采样
    [L_chol, flag] = chol(S, 'lower');
    if flag ~= 0
        % 降级为对角提议
        L_chol = diag(scale * (ub - lb) / 6);
    end

    n_accept_total = 0;
    for i = 1:N
        theta_curr = samples(i, :);
        lL_curr    = logL(i);
        lp_curr    = log_prior(theta_curr, lb, ub);
        lpost_curr = lp_curr + beta * lL_curr;

        n_accept_i = 0;
        for s = 1:n_steps
            % 生成提议样本
            z           = L_chol * randn(n_params, 1);
            theta_prop  = theta_curr + z';

            % 先验检查（越界直接拒绝）
            lp_prop = log_prior(theta_prop, lb, ub);
            if isinf(lp_prop)
                continue;
            end

            lL_prop    = log_likelihood(theta_prop, y_obs, sigma_obs, cond);
            lpost_prop = lp_prop + beta * lL_prop;

            % M-H 接受/拒绝
            if log(rand()) < (lpost_prop - lpost_curr)
                theta_curr = theta_prop;
                lL_curr    = lL_prop;
                lpost_curr = lpost_prop;
                n_accept_i = n_accept_i + 1;
            end
        end

        samples(i, :) = theta_curr;
        logL(i)       = lL_curr;
        n_accept_total = n_accept_total + n_accept_i;
    end

    acc_rate = n_accept_total / (N * n_steps);

    % ── 记录本级统计 ──────────────────────────────────────────────────
    beta_hist(end+1) = beta;          %#ok<AGROW>
    ess_hist(end+1)  = ESS;           %#ok<AGROW>
    acc_hist(end+1)  = acc_rate;      %#ok<AGROW>

    fprintf('  stage %2d | beta=%.6f | delta_beta=%.2e | ESS=%6.1f | acc=%.3f\n', ...
            stage, beta, delta_beta, ESS, acc_rate);
end

end  % run_tmcmc

% ── 辅助：计算给定 delta_beta 下权重的变异系数 ────────────────────────
function cov_val = compute_cov_weights(logL, delta_beta)
log_w   = delta_beta * logL;
log_w   = log_w - max(log_w);
w       = exp(log_w);
m       = mean(w);
if m < 1e-300
    cov_val = Inf;
    return;
end
cov_val = std(w) / m;
end

% ── 辅助：残差重采样 ──────────────────────────────────────────────────
function idx = residual_resample(w_norm, N)
% 残差重采样：先确定性分配整数份额，余量用多项式重采样
counts  = floor(N * w_norm(:));
n_det   = sum(counts);
n_res   = N - n_det;

% 余量权重
w_res   = N * w_norm(:) - counts;
w_res   = w_res / sum(w_res);

% 多项式重采样余量部分
edges   = [0; cumsum(w_res)];
u       = rand(n_res, 1);
idx_res = zeros(n_res, 1);
for k = 1:n_res
    idx_res(k) = find(u(k) >= edges(1:end-1) & u(k) < edges(2:end), 1, 'first');
end

% 汇总索引
idx_det = zeros(n_det, 1);
pos = 1;
for j = 1:N
    for c = 1:counts(j)
        idx_det(pos) = j;
        pos = pos + 1;
    end
end

idx = [idx_det; idx_res];
idx = idx(randperm(N));   % 打乱顺序
end

function theta_map = compute_map(samples, y_obs, sigma_obs, lb, ub, cond)
% 对所有样本计算完整对数后验，返回最大值对应的样本
N      = size(samples, 1);
lpost  = -Inf * ones(N, 1);
for i = 1:N
    lpost(i) = log_posterior_fn(samples(i,:), y_obs, sigma_obs, lb, ub, cond);
end
[~, best] = max(lpost);
theta_map = samples(best, :);
end

function plot_results_tmcmc(samples, theta_true, param_names, beta_hist, ess_hist, acc_hist)
% 图1：13个参数边缘后验直方图（4×4 subplot，最后格留空）
% 图2：beta演化、ESS、接受率

N        = size(samples, 1);
n_params = length(param_names);
mu       = mean(samples, 1);

% 计算 MAP（后验最大密度点用核密度估计众数近似）
theta_map = zeros(1, n_params);
for i = 1:n_params
    [f, xi] = ksdensity(samples(:, i));
    [~, mi] = max(f);
    theta_map(i) = xi(mi);
end

% ── 图1：边缘后验直方图 ────────────────────────────────────────────────
fig1 = figure('Name', 'TMCMC 边缘后验分布', 'Position', [50, 50, 1400, 960]);
n_cols = 4;
n_rows = 4;  % 4×4=16 格，13个参数 + 3格空白

for i = 1:n_params
    subplot(n_rows, n_cols, i);
    histogram(samples(:, i), 35, 'Normalization', 'pdf', ...
        'FaceColor', [0.35 0.60 0.88], 'EdgeColor', 'none', 'FaceAlpha', 0.75);
    hold on;
    xline(theta_true(i), 'g-',  'LineWidth', 2.0);
    xline(mu(i),         'r--', 'LineWidth', 1.8);
    xline(theta_map(i),  'm:',  'LineWidth', 1.8);
    xlabel(param_names{i}, 'FontSize', 8);
    ylabel('PDF',           'FontSize', 7);
    title(sprintf('%s\n真=%.4f  均=%.4f  MAP=%.4f', ...
          param_names{i}, theta_true(i), mu(i), theta_map(i)), 'FontSize', 7);
    if i == 1
        legend('后验', '真值', '均值', 'MAP', 'FontSize', 6, 'Location', 'best');
    end
end
sgtitle('TMCMC 边缘后验分布（各参数）', 'FontSize', 12);

% ── 图2：beta演化 / ESS / 接受率 ─────────────────────────────────────
fig2 = figure('Name', 'TMCMC 收敛诊断', 'Position', [100, 100, 1000, 700]);

% 去掉 beta_hist 第一个元素（初始 beta=0，无对应 ESS/acc）
stages = 1:length(acc_hist);

subplot(3, 1, 1);
plot(stages, beta_hist(2:end), 'b-o', 'MarkerSize', 4, 'LineWidth', 1.2);
yline(1.0, 'r--', 'LineWidth', 1.0);
xlabel('stage'); ylabel('\beta');
title('beta 演化');
grid on;

subplot(3, 1, 2);
plot(stages, ess_hist(2:end), 'k-s', 'MarkerSize', 4, 'LineWidth', 1.2);
yline(N * 0.5, 'r--', 'LineWidth', 1.0);
xlabel('stage'); ylabel('ESS');
title(sprintf('有效样本量（虚线 = N/2 = %d）', round(N * 0.5)));
grid on;

subplot(3, 1, 3);
plot(stages, acc_hist, 'r-^', 'MarkerSize', 4, 'LineWidth', 1.2);
yline(0.23, 'b--', 'LineWidth', 1.0);
ylim([0, 1]);
xlabel('stage'); ylabel('接受率');
title('MH 短链接受率（虚线 = 0.23 最优）');
grid on;

sgtitle('TMCMC 收敛诊断', 'FontSize', 12);

% 保存图形
try
    saveas(fig1, 'tmcmc_marginals.png');
    saveas(fig2, 'tmcmc_diagnostics.png');
catch
    fprintf('图形保存失败\n');
end
end

function plot_pairwise_posterior(samples, param_names)
% 前6个参数的 pair plot（6×6 subplot）
% 对角线：直方图；非对角线：随机取200点散点图

n_sub    = 6;
max_pts  = 200;
N        = size(samples, 1);
sub_samp = samples(:, 1:n_sub);

% 随机抽取用于散点图的索引
idx_sc = randperm(N, min(max_pts, N));

fig = figure('Name', 'TMCMC Pair Plot（前6参数）', 'Position', [80, 80, 1100, 1000]);

for row = 1:n_sub
    for col = 1:n_sub
        subplot(n_sub, n_sub, (row-1)*n_sub + col);

        if row == col
            % 对角线：边缘直方图
            histogram(sub_samp(:, col), 25, 'Normalization', 'pdf', ...
                'FaceColor', [0.35 0.60 0.88], 'EdgeColor', 'none', 'FaceAlpha', 0.8);
            xlabel(param_names{col}, 'FontSize', 6);
            ylabel('PDF', 'FontSize', 6);

        else
            % 非对角线：散点图（随机200点）
            scatter(sub_samp(idx_sc, col), sub_samp(idx_sc, row), ...
                4, [0.2 0.4 0.7], 'filled', 'MarkerFaceAlpha', 0.4);
            xlabel(param_names{col}, 'FontSize', 6);
            ylabel(param_names{row}, 'FontSize', 6);
        end

        ax = gca;
        ax.FontSize    = 6;
        ax.TickLength  = [0.02, 0.02];
        ax.XTickMode   = 'auto';
        ax.YTickMode   = 'auto';
    end
end

sgtitle('后验联合分布 Pair Plot（前6参数）', 'FontSize', 11);

try
    saveas(fig, 'tmcmc_pairplot.png');
catch
    fprintf('Pair plot 保存失败\n');
end
end

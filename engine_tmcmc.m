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

% 生成虚拟观测数据
noise_level = 0.001;
rng(42);
y_true   = y_test;                          % [R_ud_true, C_ud_true]
sigma_obs = noise_level * abs(y_true);      % [sigma_R, sigma_C]
y_obs     = y_true + sigma_obs .* randn(1, 2);
fprintf('\n生成虚拟观测数据：\n');
fprintf('  R_obs = %.4f [N·s/kg]，  sigma_R = %.6f\n', y_obs(1), sigma_obs(1));
fprintf('  C_obs = %.6f [kg/(N·h)]，sigma_C = %.8f\n', y_obs(2), sigma_obs(2));

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

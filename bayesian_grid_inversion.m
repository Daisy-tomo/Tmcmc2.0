%% ============================================================
%  贝叶斯后验反演 - 粗网格搜索方法
%  ============================================================
%  功能：根据观测量（比推力 R、比油耗 C）对 4 个发动机修正参数
%        eta_k、eta_t、eta_m、eta_T 进行贝叶斯后验反演
%  方法：粗网格穷举 + 高斯似然 + 均匀先验
%  输出：各参数边缘后验分布图、两两联合后验矩阵图
%  ============================================================
%  使用说明：
%    1. 修改第"观测量"节中的 R_obs、C_obs 为实际测量值
%    2. 修改各参数搜索范围和网格密度 N_grid
%    3. 如有真实前向模型，替换 forward_model 函数中标注的接口行
%  ============================================================

clear; clc; close all;

fprintf('============================================\n');
fprintf('   贝叶斯后验反演  -  粗网格搜索方法\n');
fprintf('   参数：eta_k / eta_t / eta_m / eta_T\n');
fprintf('============================================\n\n');

%% -------- 1. 发动机固定参数 --------
params = setup_engine_params();

%% -------- 2. 观测量与观测误差 --------
% 方法：先用"名义参数"（全为 1.0）跑一次前向模型得到参考输出，
%       再叠加少量偏差模拟真实观测（可直接替换为实测数据）
theta_true = [1.00, 1.00, 1.00, 1.00];
[R_ref, C_ref] = forward_model(theta_true, params);

if isnan(R_ref) || isnan(C_ref)
    error('名义参数下前向模型返回 NaN，请检查 forward_model 函数！');
end

% 模拟观测量（真值 + 1% 偏差，可替换为实测值）
R_obs   = R_ref * 1.01;   % 比推力观测值 [N·s/kg]
C_obs   = C_ref * 0.99;   % 比油耗观测值 [kg/(N·h)]

% 观测误差标准差（约 1%）
sigma_R = R_ref * 0.01;
sigma_C = C_ref * 0.01;

fprintf('【观测量】\n');
fprintf('  R_obs   = %10.4f  N·s/kg     (sigma_R = %.4f)\n', R_obs, sigma_R);
fprintf('  C_obs   = %10.6f  kg/(N·h)   (sigma_C = %.6f)\n\n', C_obs, sigma_C);

%% -------- 3. 待反演参数的搜索网格（均匀先验区间）--------
%  格式：[下界, 上界]，每个参数在此区间内均匀离散
eta_k_range = [0.88, 1.12];   % 压气机效率修正系数
eta_t_range = [0.88, 1.12];   % 涡轮效率修正系数
eta_m_range = [0.95, 1.05];   % 机械效率修正系数
eta_T_range = [0.95, 1.05];   % 燃烧室温度修正系数

N_grid = 12;   % 每个维度的网格点数（增大可提高精度，计算量 ∝ N^4）

eta_k_vec = linspace(eta_k_range(1), eta_k_range(2), N_grid);
eta_t_vec = linspace(eta_t_range(1), eta_t_range(2), N_grid);
eta_m_vec = linspace(eta_m_range(1), eta_m_range(2), N_grid);
eta_T_vec = linspace(eta_T_range(1), eta_T_range(2), N_grid);

total_pts = N_grid^4;
fprintf('【网格设置】每维 %d 点，总计 %d 个网格点\n\n', N_grid, total_pts);

%% -------- 4. 粗网格搜索：遍历所有参数组合 --------
% 维度顺序：(eta_k, eta_t, eta_m, eta_T)
logPost     = -Inf(N_grid, N_grid, N_grid, N_grid);
valid_cnt   = 0;
invalid_cnt = 0;
point_idx   = 0;
report_step = max(1, floor(total_pts / 40));   % 约报告 40 次进度

fprintf('开始网格搜索...\n');
t_start = tic;

for i1 = 1:N_grid
    for i2 = 1:N_grid
        for i3 = 1:N_grid
            for i4 = 1:N_grid
                point_idx = point_idx + 1;

                % 当前参数向量
                theta = [eta_k_vec(i1), eta_t_vec(i2), ...
                         eta_m_vec(i3), eta_T_vec(i4)];

                % 调用前向模型（含异常处理）
                try
                    [R_pred, C_pred] = forward_model(theta, params);
                catch ME
                    R_pred = NaN;
                    C_pred = NaN;
                end

                % 校验输出有效性
                if isnan(R_pred) || isnan(C_pred) || ...
                   ~isreal(R_pred) || ~isreal(C_pred) || ...
                   R_pred <= 0    || C_pred <= 0
                    invalid_cnt = invalid_cnt + 1;
                    % logPost 保持 -Inf（已预填）
                else
                    % 高斯对数似然（均匀先验 → 对数后验 = 对数似然）
                    logPost(i1,i2,i3,i4) = ...
                        -0.5 * ((R_pred - R_obs) / sigma_R)^2 ...
                        -0.5 * ((C_pred - C_obs) / sigma_C)^2;
                    valid_cnt = valid_cnt + 1;
                end

                % 进度显示
                if mod(point_idx, report_step) == 0 || point_idx == total_pts
                    elapsed = toc(t_start);
                    remain  = elapsed * (total_pts - point_idx) / max(point_idx,1);
                    fprintf('  进度 %5.1f%%  有效=%d  无效=%d  剩余≈%.1fs\n', ...
                            100*point_idx/total_pts, valid_cnt, invalid_cnt, remain);
                end
            end
        end
    end
end

fprintf('\n搜索完成，耗时 %.2f 秒，有效点 %d / %d (%.1f%%)\n\n', ...
        toc(t_start), valid_cnt, total_pts, 100*valid_cnt/total_pts);

if valid_cnt == 0
    error('所有网格点均无效！请检查 forward_model 函数或参数搜索范围。');
end

%% -------- 5. 数值稳定化 + 归一化 → 离散后验概率 --------
% 减去最大对数值，防止 exp() 数值溢出
valid_mask = isfinite(logPost);
max_lp     = max(logPost(valid_mask));

Post       = exp(logPost - max_lp);   % 平移后指数化
Post(~valid_mask) = 0;                % 无效点置零
Post_norm  = Post / sum(Post(:));     % 归一化为概率质量函数

fprintf('归一化后验：有效概率总质量 = %.8f\n\n', sum(Post_norm(:)));

%% -------- 6. 边缘后验分布 --------
% Post_norm 维度：dim1=eta_k, dim2=eta_t, dim3=eta_m, dim4=eta_T
% 各参数的一维边缘：对其余三个维度求和
marg    = cell(1, 4);
marg{1} = squeeze(sum(Post_norm, [2,3,4]));   % P(eta_k)
marg{2} = squeeze(sum(Post_norm, [1,3,4]));   % P(eta_t)
marg{3} = squeeze(sum(Post_norm, [1,2,4]));   % P(eta_m)
marg{4} = squeeze(sum(Post_norm, [1,2,3]));   % P(eta_T)

% 统一为列向量
for k = 1:4
    marg{k} = marg{k}(:);
end

%% -------- 7. 统计输出 --------
param_names = {'\eta_k', '\eta_t', '\eta_m', '\eta_T'};
param_vecs  = {eta_k_vec, eta_t_vec, eta_m_vec, eta_T_vec};
print_statistics(param_names, param_vecs, marg);

%% -------- 8. 绘图 --------
fprintf('正在绘图...\n');
plot_marginal_posteriors(param_names, param_vecs, marg);
plot_joint_matrix(param_names, param_vecs, marg, Post_norm);
fprintf('绘图完毕。\n\n');

fprintf('============================================\n');
fprintf('   程序运行完毕\n');
fprintf('============================================\n');


%% ============================================================
%%                      局 部 函 数
%% ============================================================

%% -------- 发动机固定参数 --------
function params = setup_engine_params()
    % 设置发动机基准参数（其余参数固定，不参与反演）
    params.pi_k   = 25;         % 压气机总压比
    params.T_t0   = 288.15;     % 进口总温 [K]
    params.p_t0   = 101325;     % 进口总压 [Pa]
    params.T_t4   = 1650;       % 涡轮进口设计总温 [K]
    params.H_u    = 43.1e6;     % 燃料低热值 [J/kg]
    params.cp     = 1004;       % 定压比热容 [J/(kg·K)]
    params.gamma  = 1.4;        % 比热比
    params.V_0    = 0;          % 飞行速度（地面静态）[m/s]
    params.eta_c0 = 0.88;       % 压气机基准绝热效率
    params.eta_t0 = 0.90;       % 涡轮基准绝热效率
    params.eta_m0 = 0.99;       % 传动机械基准效率

    fprintf('【固定参数】pi_k=%.0f  T_t0=%.2fK  T_t4=%.0fK\n\n', ...
            params.pi_k, params.T_t0, params.T_t4);
end

%% -------- 前向模型（接口预留，可替换为实际仿真程序）--------
function [R_pred, C_pred] = forward_model(theta, params)
    % ============================================================
    % 前向模型：由修正参数 theta 预测比推力 R 和比油耗 C
    %
    % 输入：
    %   theta(1) = eta_k  压气机效率修正系数
    %   theta(2) = eta_t  涡轮效率修正系数
    %   theta(3) = eta_m  传动机械效率修正系数
    %   theta(4) = eta_T  涡轮进口温度修正系数
    %   params           固定参数结构体（见 setup_engine_params）
    %
    % 输出：
    %   R_pred  预测比推力 [N·s/kg]
    %   C_pred  预测比油耗 [kg/(N·h)]
    %
    % ---- 外部函数接口（替换为实际模型时取消注释以下两行）----
    % R_pred = calc_Rud_from_theta(theta, params);
    % C_pred = calc_Cud_from_theta(theta, params);
    % return;
    % ============================================================

    % 提取修正参数
    eta_k = theta(1);
    eta_t = theta(2);
    eta_m = theta(3);
    eta_T = theta(4);

    % 提取固定参数
    gamma  = params.gamma;
    cp     = params.cp;
    T_t0   = params.T_t0;
    pi_k   = params.pi_k;
    T_t4_0 = params.T_t4;
    H_u    = params.H_u;
    V_0    = params.V_0;

    % 将修正系数限制在物理合理范围内（防止数值发散）
    eta_c  = min(max(params.eta_c0 * eta_k, 0.50), 0.99);   % 压气机绝热效率
    eta_tb = min(max(params.eta_t0 * eta_t, 0.50), 0.99);   % 涡轮绝热效率
    eta_mb = min(max(params.eta_m0 * eta_m, 0.80), 1.00);   % 机械传动效率
    T_t4   = T_t4_0 * eta_T;                                 % 修正后涡轮进口总温

    % ---- 压气机出口总温 ----
    tau_k_id = pi_k ^ ((gamma - 1) / gamma);                 % 理想温比
    T_t3 = T_t0 + T_t0 * (tau_k_id - 1) / eta_c;           % 实际压气机出口总温

    % ---- 油气比（由燃烧室能量守恒）----
    f = cp * (T_t4 - T_t3) / (H_u - cp * T_t4);
    if f <= 0 || f >= 0.10 || ~isreal(f)
        R_pred = NaN; C_pred = NaN; return;
    end

    % ---- 涡轮功平衡：涡轮功 = 压气机耗功 / 机械效率 ----
    W_c  = cp * (T_t3 - T_t0);
    W_t  = W_c / eta_mb;
    T_t5 = T_t4 - W_t / (cp * (1 + f));                     % 涡轮出口总温
    if T_t5 <= 200 || ~isreal(T_t5)
        R_pred = NaN; C_pred = NaN; return;
    end

    % ---- 涡轮膨胀压比 ----
    tau_t = T_t5 / T_t4;                                     % 涡轮温比 (<1)
    pi_t  = tau_t ^ (gamma / ((gamma - 1) * eta_tb));        % 涡轮压比 (<1)

    % ---- 尾喷管可用压比 ----
    pi_noz = pi_k / pi_t;
    if pi_noz < 1.0 || ~isreal(pi_noz)
        R_pred = NaN; C_pred = NaN; return;
    end

    % ---- 尾喷管出口静温与速度（等熵完全膨胀）----
    T_9 = T_t5 * (1 / pi_noz) ^ ((gamma - 1) / gamma);
    V_9 = sqrt(2 * cp * (T_t5 - T_9));
    if ~isreal(V_9) || V_9 <= 0
        R_pred = NaN; C_pred = NaN; return;
    end

    % ---- 输出计算结果 ----
    R_pred = (1 + f) * V_9 - V_0;          % 比推力 [N·s/kg]
    C_pred = f / R_pred * 3600;             % 比油耗 [kg/(N·h)]
end

%% -------- 统计结果打印 --------
function print_statistics(param_names, param_vecs, marg)
    fprintf('=============== 后验统计结果 ===============\n');
    str_labels = {'eta_k', 'eta_t', 'eta_m', 'eta_T'};

    for k = 1:4
        pvec = param_vecs{k}(:);
        p    = marg{k}(:);
        p    = p / sum(p);                         % 确保归一化

        mu_post  = dot(pvec, p);                   % 后验均值
        std_post = sqrt(dot((pvec - mu_post).^2, p));  % 后验标准差
        [~, imap] = max(p);
        map_val  = pvec(imap);                     % MAP 估计

        % 95% 等尾置信区间（累积概率）
        cdf_p  = cumsum(p);
        idx_lo = find(cdf_p >= 0.025, 1, 'first');
        idx_hi = find(cdf_p >= 0.975, 1, 'first');
        ci_lo  = pvec(max(idx_lo,1));
        ci_hi  = pvec(min(idx_hi, length(pvec)));

        fprintf('  %-8s | MAP = %.5f | 均值 = %.5f | 标准差 = %.5f | 95%%CI [%.5f, %.5f]\n', ...
                str_labels{k}, map_val, mu_post, std_post, ci_lo, ci_hi);
    end
    fprintf('============================================\n\n');
end

%% -------- 边缘后验分布图 --------
function plot_marginal_posteriors(param_names, param_vecs, marg)
    figure('Name', '参数边缘后验分布', 'NumberTitle', 'off', ...
           'Position', [80, 520, 1400, 320]);

    clr_bar   = [0.20, 0.55, 0.80];   % 柱图颜色（蓝）
    clr_post  = [0.08, 0.55, 0.08];   % 后验曲线颜色（绿）
    clr_prior = [1.00, 0.55, 0.10];   % 先验颜色（橙）

    for k = 1:4
        subplot(1, 4, k);
        pvec    = param_vecs{k}(:)';
        p       = marg{k}(:)' / sum(marg{k});
        dv      = pvec(2) - pvec(1);
        pdf_bar = p / dv;                       % 概率密度

        % 柱状图
        bar(pvec, pdf_bar, 1.0, 'FaceColor', clr_bar, ...
            'EdgeColor', [1 1 1]*0.9, 'FaceAlpha', 0.80);
        hold on;

        % 样条插值平滑后验曲线
        x_fine = linspace(pvec(1), pvec(end), 300);
        y_fine = max(interp1(pvec, pdf_bar, x_fine, 'spline'), 0);
        plot(x_fine, y_fine, '--', 'Color', clr_post, 'LineWidth', 2.2);

        % 均匀先验（常数线）
        prior_h = 1 / (pvec(end) - pvec(1));
        plot(pvec([1, end]), prior_h * [1, 1], '-', ...
             'Color', clr_prior, 'LineWidth', 2.2);

        xlabel(param_names{k}, 'Interpreter', 'tex', 'FontSize', 13);
        ylabel('概率密度', 'FontSize', 11);
        title(['$p(', regexprep(param_names{k},'\\',''), ' \mid \mathbf{y})$'], ...
              'Interpreter', 'latex', 'FontSize', 13);
        legend({'MCMC（网格）', '贝叶斯积分', '先验分布'}, ...
               'Location', 'best', 'FontSize', 8);
        set(gca, 'FontSize', 10, 'Box', 'on');
        grid on;
    end
    sgtitle('各参数边缘后验概率密度分布', 'FontSize', 14, 'FontWeight', 'bold');
end

%% -------- 联合后验矩阵图 --------
function plot_joint_matrix(param_names, param_vecs, marg, Post_norm)
    fig = figure('Name', '参数联合后验矩阵图', 'NumberTitle', 'off', ...
                 'Position', [60, 30, 1080, 980]);

    N = 4;
    clr_bar   = [0.20, 0.55, 0.78];
    clr_post  = [0.08, 0.55, 0.08];
    clr_prior = [1.00, 0.55, 0.10];
    teal_cmap = [linspace(0.0,0.0,64)', ...
                 linspace(0.45,0.75,64)', ...
                 linspace(0.45,0.75,64)'];   % 青绿色调散点配色

    for row = 1:N
        for col = 1:N
            ax = subplot(N, N, (row-1)*N + col);

            if row == col
                %% ---- 对角线：单参数边缘后验 ----
                pvec    = param_vecs{row}(:)';
                p       = marg{row}(:)' / sum(marg{row});
                dv      = pvec(2) - pvec(1);
                pdf_bar = p / dv;

                bar(pvec, pdf_bar, 1.0, 'FaceColor', clr_bar, ...
                    'EdgeColor', [0.9 0.9 0.9], 'FaceAlpha', 0.85);
                hold on;

                x_fine = linspace(pvec(1), pvec(end), 200);
                y_fine = max(interp1(pvec, pdf_bar, x_fine, 'spline'), 0);
                plot(x_fine, y_fine, '--', 'Color', clr_post, 'LineWidth', 1.8);

                prior_h = 1 / (pvec(end) - pvec(1));
                plot(pvec([1 end]), prior_h*[1 1], '-', ...
                     'Color', clr_prior, 'LineWidth', 1.8);

                xlabel(param_names{col}, 'Interpreter', 'tex', 'FontSize', 10);

            elseif row > col
                %% ---- 下三角：二维联合后验热图 ----
                %  x 轴 → param_vecs{col}，y 轴 → param_vecs{row}
                joint = get_joint2d(Post_norm, col, row);   % size: (N_col × N_row)
                px = param_vecs{col};
                py = param_vecs{row};

                % imagesc(x, y, C): C(iy, ix) → 需要转置
                imagesc(ax, px, py, joint');
                axis(ax, 'xy');
                colormap(ax, hot);

                xlabel(param_names{col}, 'Interpreter', 'tex', 'FontSize', 10);
                ylabel(param_names{row}, 'Interpreter', 'tex', 'FontSize', 10);

            else
                %% ---- 上三角：后验散点图 ----
                %  x 轴 → param_vecs{col}，y 轴 → param_vecs{row}
                joint = get_joint2d(Post_norm, row, col);   % size: (N_row × N_col)
                px = param_vecs{col};
                py = param_vecs{row};

                [X, Y] = meshgrid(px, py);                  % X/Y: (N_row × N_col)
                w = joint(:) / sum(joint(:));               % 权重向量

                scatter(ax, X(:), Y(:), 12, w, 'filled', ...
                        'MarkerFaceAlpha', 0.75);
                colormap(ax, teal_cmap);

                xlabel(param_names{col}, 'Interpreter', 'tex', 'FontSize', 10);
                ylabel(param_names{row}, 'Interpreter', 'tex', 'FontSize', 10);
            end

            set(ax, 'FontSize', 9, 'Box', 'on');
            grid(ax, 'on');
        end
    end

    % 图例说明条（底部注释）
    annotation(fig, 'textbox', [0.10, 0.003, 0.80, 0.030], ...
               'String', ...
               '先验分布 ——      贝叶斯积分 - - -      网格采样（热图/散点）■', ...
               'HorizontalAlignment', 'center', 'FontSize', 11, ...
               'EdgeColor', 'none', 'BackgroundColor', 'none');

    sgtitle('参数联合后验概率密度矩阵图', 'FontSize', 14, 'FontWeight', 'bold');
end

%% -------- 二维联合边缘分布（辅助函数）--------
function joint = get_joint2d(Post_norm, dim_a, dim_b)
    % 计算 Post_norm（4维）在 dim_a 和 dim_b 方向的二维联合边缘分布
    % 输出 joint(i,j) = P(param_{dim_a}=i, param_{dim_b}=j)
    %
    % 维度定义：dim1=eta_k, dim2=eta_t, dim3=eta_m, dim4=eta_T
    %
    % 实现：对其余两个维度由高到低求和（避免维度偏移），然后 squeeze

    other_dims = sort(setdiff(1:4, [dim_a, dim_b]), 'descend');

    result = Post_norm;
    for d = other_dims
        result = sum(result, d);
    end
    result = squeeze(result);

    % squeeze 后，剩余两个维度按升序排列
    % 若 dim_a < dim_b，result 已是 (dim_a × dim_b)；否则需转置
    if dim_a > dim_b
        result = result';
    end
    joint = result;
end

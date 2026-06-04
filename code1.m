% ============================================================
% 任务一：空域+频域结合图像增强算法
% ============================================================
% 算法流程：
%   1. DC偏移校正（减去均值偏移）
%   2. 频域：巴特沃斯陷波滤波器 → 去除周期噪声
%   3. 空域：高斯低通滤波器 → 抑制随机噪声
%   4. 评估：MSE / PSNR / SNR / SSIM
%
% 知识点框架内技术：
%   - 频域：陷波滤波器 (Notch Filter)
%   - 空域：高斯滤波器
% ============================================================
clear; close all; clc;

% ============================================================
% 0. 路径设置
% ============================================================
base_dir = 'd:\Github\School\digital_picture';
src_dir  = fullfile(base_dir, '原图与参考图');
out_dir  = fullfile(base_dir, '处理后图像');

% ============================================================
% 1. 加载图像
% ============================================================
fprintf('============================================================\n');
fprintf('任务一：空域+频域结合图像增强算法\n');
fprintf('============================================================\n\n');

original = im2double(imread(fullfile(src_dir, 'dogOriginal.bmp')));
distorted = im2double(imread(fullfile(src_dir, 'dogDistorted.bmp')));

% 转换为灰度（已是灰度图，但确保维度一致）
if size(original, 3) == 3
    original = rgb2gray(original);
end
if size(distorted, 3) == 3
    distorted = rgb2gray(distorted);
end

[H, W] = size(distorted);
fprintf('图像尺寸: %d x %d\n', W, H);
fprintf('原始图像均值=%.2f, 失真图像均值=%.2f\n\n', mean(original(:))*255, mean(distorted(:))*255);

% ============================================================
% 2. 预处理：评估原始失真 + DC偏移校正
% ============================================================
[mse_dist, psnr_dist, snr_dist, ssim_dist] = compute_metrics(distorted, original);
fprintf('【初始状态】\n');
fprintf('  MSE  = %.2f\n', mse_dist * 255^2);
fprintf('  PSNR = %.2f dB\n', psnr_dist);
fprintf('  SNR  = %.2f dB\n', snr_dist);
fprintf('  SSIM = %.4f\n\n', ssim_dist);

% DC偏移校正
dc_offset = mean(distorted(:)) - mean(original(:));
img_dc = distorted - dc_offset;
fprintf('【DC校正】减去直流偏移: %.4f (%.1f灰度级)\n\n', dc_offset, dc_offset*255);

% ============================================================
% 3. 频域：巴特沃斯陷波滤波器
% ============================================================
fprintf('【频域陷波滤波】\n');

% 周期噪声频率（基于精确分析——噪声特性发现阶段）
% 基频: f_u = 0.2, f_v ≈ 0.2005（归一化频率）
% 对应像素坐标: u0 = 0.2 * 355 = 71, v0 ≈ 0.2005 * 369 = 74
f_u = 71;
f_v = 74;

% 待滤除的频率对（只列正频率侧，函数自动处理共轭对称）
notch_peaks = [
    f_u,  0;      % f1: 水平基频
    2*f_u, 0;     % 2f1: 水平二次谐波
    0,    f_v;    % f2: 垂直基频
    0,    2*f_v;  % 2f2: 垂直二次谐波
    f_u,  f_v;    % f1+f2 交叉项
    f_u, -f_v;    % f1-f2 交叉项
    2*f_u, f_v;   % 2f1+f2
    2*f_u,-f_v;   % 2f1-f2
    f_u,  2*f_v;  % f1+2f2
    f_u, -2*f_v;  % f1-2f2
    2*f_u, 2*f_v; % 2f1+2f2
    2*f_u,-2*f_v; % 2f1-2f2
    ];

% 最佳参数（参数网格搜索确定）
D0 = 5;  % 陷波半径
n  = 1;  % 巴特沃斯阶数

img_notch = butterworth_notch_filter(img_dc, notch_peaks, D0, n);

[mse_n, psnr_n, snr_n, ssim_n] = compute_metrics(img_notch, original);
fprintf('  滤除 %d 个周期噪声频率对 (D0=%d, n=%d)\n', size(notch_peaks,1), D0, n);
fprintf('  -> MSE=%.2f, PSNR=%.2f dB, SNR=%.2f dB, SSIM=%.4f\n\n', ...
    mse_n*255^2, psnr_n, snr_n, ssim_n);

% 保存陷波后的中间结果
imwrite(img_notch, fullfile(out_dir, 'task1_step1_notch.bmp'));
fprintf('  已保存: task1_step1_notch.bmp\n');

% ============================================================
% 4. 空域：高斯低通滤波器
% ============================================================
fprintf('\n【空域高斯滤波】\n');

% 最佳 sigma（参数网格搜索确定）
sigma = 1.3;
img_final = imgaussfilt(img_notch, sigma);

fprintf('  高斯滤波 sigma=%.1f\n\n', sigma);

% 保存高斯滤波后的结果
imwrite(img_final, fullfile(out_dir, 'task1_step2_gaussian.bmp'));
fprintf('  已保存: task1_step2_gaussian.bmp\n');

% ============================================================
% 5. 最终评估
% ============================================================
fprintf('\n============================================================\n');
fprintf('最终评估结果\n');
fprintf('============================================================\n');

[mse_final, psnr_final, snr_final, ssim_final] = compute_metrics(img_final, original);

fprintf('处理前（失真图）:\n');
fprintf('  MSE  = %.2f\n', mse_dist * 255^2);
fprintf('  PSNR = %.2f dB\n', psnr_dist);
fprintf('  SNR  = %.2f dB\n', snr_dist);
fprintf('  SSIM = %.4f\n\n', ssim_dist);

fprintf('处理后（去噪结果）:\n');
fprintf('  MSE  = %.2f\n', mse_final * 255^2);
fprintf('  PSNR = %.2f dB\n', psnr_final);
fprintf('  SNR  = %.2f dB\n', snr_final);
fprintf('  SSIM = %.4f\n\n', ssim_final);

fprintf('改善幅度:\n');
fprintf('  PSNR: +%.2f dB\n', psnr_final - psnr_dist);
fprintf('  SNR:  +%.2f dB\n', snr_final - snr_dist);
fprintf('  SSIM: +%.4f\n', ssim_final - ssim_dist);
fprintf('  MSE 降低: %.1f%%\n', 100 * (1 - mse_final/mse_dist));

% ============================================================
% 6. 保存最终结果
% ============================================================
fprintf('\n============================================================\n');
fprintf('保存结果到: %s\n', out_dir);
fprintf('============================================================\n');

imwrite(img_final, fullfile(out_dir, 'task1_final_result.bmp'));
fprintf('  [OK] task1_final_result.bmp (最终去噪结果)\n');

imwrite(distorted, fullfile(out_dir, 'task1_distorted_orig.bmp'));
fprintf('  [OK] task1_distorted_orig.bmp (原始失真图)\n');

imwrite(original, fullfile(out_dir, 'task1_original_ref.bmp'));
fprintf('  [OK] task1_original_ref.bmp (原始清晰图)\n');

% 残差图像（放大5倍便于观察，加0.5偏移）
residual_before = (distorted - original) * 5 + 0.5;
residual_after  = (img_final - original) * 5 + 0.5;
imwrite(residual_before, fullfile(out_dir, 'task1_residual_before.bmp'));
imwrite(residual_after, fullfile(out_dir, 'task1_residual_after.bmp'));
fprintf('  [OK] task1_residual_before.bmp (去噪前残差x5)\n');
fprintf('  [OK] task1_residual_after.bmp  (去噪后残差x5)\n');

% ============================================================
% 7. 生成可视化对比图
% ============================================================
fprintf('\n生成可视化对比图...\n');

figure('Position', [100, 100, 1400, 900], 'Visible', 'off');

% 7a. 空域对比 (1行3列)
subplot(2, 3, 1);
imshow(original); title('Original (Reference)', 'FontWeight', 'bold');

subplot(2, 3, 2);
imshow(distorted);
title(sprintf('Distorted (PSNR=%.1fdB)', psnr_dist), 'FontWeight', 'bold');

subplot(2, 3, 3);
imshow(img_final);
title(sprintf('Denoised (PSNR=%.1fdB)', psnr_final), 'FontWeight', 'bold', 'Color', [0 0.6 0]);

% 7b. FFT频谱对比 (2行3列)
subplot(2, 3, 4);
fft_orig = fftshift(fft2(original));
imshow(log(1 + abs(fft_orig)), []);
title('FFT Spectrum: Original', 'FontWeight', 'bold');

subplot(2, 3, 5);
fft_dist = fftshift(fft2(distorted));
imshow(log(1 + abs(fft_dist)), []);
title('FFT Spectrum: Distorted', 'FontWeight', 'bold');
hold on;
% 标注周期噪声峰值
ch = H/2; cw = W/2;
for k = 1:size(notch_peaks, 1)
    plot(cw + notch_peaks(k,1), ch + notch_peaks(k,2), 'co', ...
        'MarkerSize', 5, 'LineWidth', 1);
end
hold off;

subplot(2, 3, 6);
fft_final = fftshift(fft2(img_final));
imshow(log(1 + abs(fft_final)), []);
title('FFT Spectrum: Denoised', 'FontWeight', 'bold', 'Color', [0 0.6 0]);

sgtitle(sprintf('Task 1: Butterworth Notch(D0=%d,n=%d) + Gaussian(\\sigma=%.1f)\nPSNR: %.2f dB | SNR: %.2f dB | SSIM: %.4f', ...
    D0, n, sigma, psnr_final, snr_final, ssim_final), 'FontSize', 12, 'FontWeight', 'bold');

saveas(gcf, fullfile(out_dir, 'task1_comparison.png'));
fprintf('  [OK] task1_comparison.png\n');

% 残差直方图对比
figure('Position', [150, 150, 1200, 500], 'Visible', 'off');
noise_before = (distorted(:) - original(:)) * 255;
noise_after  = (img_final(:) - original(:)) * 255;

subplot(1, 2, 1);
histogram(noise_before, 80, 'FaceColor', 'r', 'FaceAlpha', 0.6, 'EdgeColor', 'k', 'LineWidth', 0.1);
hold on; xline(0, 'b--', 'LineWidth', 1.5);
xlabel('Noise Intensity (gray level)'); ylabel('Pixel Count');
title(sprintf('Residual Before (std=%.1f)', std(noise_before)), 'FontWeight', 'bold');
xlim([-150, 200]);
legend(sprintf('Before (std=%.1f)', std(noise_before)));

subplot(1, 2, 2);
histogram(noise_after, 80, 'FaceColor', 'g', 'FaceAlpha', 0.6, 'EdgeColor', 'k', 'LineWidth', 0.1);
hold on; xline(0, 'b--', 'LineWidth', 1.5);
xlabel('Noise Intensity (gray level)'); ylabel('Pixel Count');
title(sprintf('Residual After (std=%.1f)', std(noise_after)), 'FontWeight', 'bold');
xlim([-150, 200]);
legend(sprintf('After (std=%.1f)', std(noise_after)));

sgtitle('Residual Noise Distribution Comparison', 'FontSize', 12, 'FontWeight', 'bold');
saveas(gcf, fullfile(out_dir, 'task1_residual_histogram.png'));
fprintf('  [OK] task1_residual_histogram.png\n');

fprintf('\nDone. 所有结果已保存至处理后图像文件夹。\n');

% ============================================================
% === 辅助函数 ===============================================
% ============================================================

function img_filtered = butterworth_notch_filter(img, notch_centers, D0, n)
% 巴特沃斯陷波滤波器
%   img:           输入图像 (H x W), double [0,1]
%   notch_centers: Nx2 矩阵，每行 [uc, vc] 为频率坐标（像素单位，相对DC）
%   D0:            陷波半径
%   n:             巴特沃斯阶数
%   img_filtered:  滤波后图像

[H, W] = size(img);
ch = floor(H / 2);
cw = floor(W / 2);

% FFT
F = fft2(img);
F_shifted = fftshift(F);

% 构建频率网格
u = (0:W-1) - cw;
v = (0:H-1) - ch;
[U, V] = meshgrid(u, v);

% 初始传递函数（全通）
H_filter = ones(H, W);

% 对每个陷波频率对（共轭对称），构建陷波
for k = 1:size(notch_centers, 1)
    uc = notch_centers(k, 1);
    vc = notch_centers(k, 2);

    % 正频率侧距离
    D_pos = sqrt((U - uc).^2 + (V - vc).^2);
    % 负频率侧距离（共轭对称）
    D_neg = sqrt((U + uc).^2 + (V + vc).^2);

    % 避免除零
    D_pos = max(D_pos, 1e-10);
    D_neg = max(D_neg, 1e-10);

    % 巴特沃斯陷波: H(u,v) = 1 / (1 + (D0^2 / (D_pos * D_neg))^n)
    H_notch = 1 ./ (1 + (D0^2 ./ (D_pos .* D_neg)).^n);
    H_filter = H_filter .* H_notch;
end

% 应用滤波器
F_filtered = F_shifted .* H_filter;
F_filtered = ifftshift(F_filtered);
img_filtered = real(ifft2(F_filtered));
end

function [mse, psnr, snr, ssim_val] = compute_metrics(img_proc, img_ref)
% 计算图像质量指标
%   输入: img_proc, img_ref 为 double [0,1] 范围
%   输出: MSE(归一化), PSNR(dB), SNR(dB), SSIM

diff = img_proc - img_ref;
mse = mean(diff(:).^2);

% PSNR
if mse > 0
    psnr = 10 * log10(1 / mse);  % 对于 [0,1] 范围
else
    psnr = inf;
end

% SNR (基于方差)
var_ref = var(img_ref(:));
var_noise = var(diff(:));
if var_noise > 0
    snr = 10 * log10(var_ref / var_noise);
else
    snr = inf;
end

% SSIM
K1 = 0.01; K2 = 0.03;
C1 = K1^2;
C2 = K2^2;

% 使用 11x11 高斯窗口
window = fspecial('gaussian', 11, 1.5);
window = window / sum(window(:));

mu1 = imfilter(img_ref, window, 'replicate');
mu2 = imfilter(img_proc, window, 'replicate');

mu1_sq = mu1.^2;
mu2_sq = mu2.^2;
mu1_mu2 = mu1 .* mu2;

sigma1_sq = imfilter(img_ref.^2, window, 'replicate') - mu1_sq;
sigma2_sq = imfilter(img_proc.^2, window, 'replicate') - mu2_sq;
sigma12 = imfilter(img_ref .* img_proc, window, 'replicate') - mu1_mu2;

% 数值稳定性
sigma1_sq = max(sigma1_sq, 0);
sigma2_sq = max(sigma2_sq, 0);

ssim_map = ((2*mu1_mu2 + C1) .* (2*sigma12 + C2)) ./ ...
    ((mu1_sq + mu2_sq + C1) .* (sigma1_sq + sigma2_sq + C2) + 1e-10);

ssim_val = mean(ssim_map(:));
end

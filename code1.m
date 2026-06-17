% =========================================================================
% 课程设计3 - 任务1：图像增强算法设计（最终版本）
% 策略：频域陷波 + 中值滤波 + 非局部均值 + 维纳滤波 + 温和锐化
% 目标：在保持高SSIM的同时，视觉效果明显改善
% =========================================================================

clear; clc; close all;

% -------------------------------------------------------------------------
% 1. 读取图像
% -------------------------------------------------------------------------
originalImg = imread('原图与参考图\dogOriginal.bmp');
noisyImg = imread('原图与参考图\dogDistorted.bmp');

if size(originalImg, 3) == 3
    originalImg = rgb2gray(originalImg);
end
if size(noisyImg, 3) == 3
    noisyImg = rgb2gray(noisyImg);
end

originalImg = double(originalImg);
noisyImg = double(noisyImg);
[M, N] = size(noisyImg);

fprintf('==============================================\n');
fprintf('   图像增强算法 - 任务1\n');
fprintf('   混合噪声去除（周期噪声+随机噪声）\n');
fprintf('==============================================\n\n');

% -------------------------------------------------------------------------
% 2. 频域陷波滤波 - 去除周期噪声
% -------------------------------------------------------------------------
fprintf('【步骤1】频域陷波滤波去除周期噪声\n');

F_noisy = fft2(noisyImg);
F_shifted = fftshift(F_noisy);
spectrum = abs(F_shifted);

% fftshift 后频谱低频在中心，因此频率坐标也必须以中心点为零点
center_u = floor(M/2) + 1;
center_v = floor(N/2) + 1;
[V, U] = meshgrid((1:N) - center_v, (1:M) - center_u);
D = sqrt(U.^2 + V.^2);

% 检测周期噪声峰值：用对数频谱和局部极大值，避免整片高能区域被重复陷波
log_spectrum = log(1 + spectrum);
search_mask = (D > 12) & (D < min(M,N)/2.05);
spectrum_search = log_spectrum;
spectrum_search(~search_mask) = 0;
valid_vals = spectrum_search(search_mask);
threshold = mean(valid_vals) + 3.8 * std(valid_vals);
peak_mask = imregionalmax(spectrum_search) & search_mask & (spectrum_search > threshold);
[peak_rows, peak_cols] = find(peak_mask);

% 只保留最强的少量周期峰，过多陷波会把图像细节一起削弱
max_peaks = 8;
if length(peak_rows) > max_peaks
    peak_vals = spectrum_search(sub2ind(size(spectrum_search), peak_rows, peak_cols));
    [~, order_idx] = sort(peak_vals, 'descend');
    order_idx = order_idx(1:max_peaks);
    peak_rows = peak_rows(order_idx);
    peak_cols = peak_cols(order_idx);
end

fprintf('  → 检测到 %d 个周期噪声峰值\n', length(peak_rows));

% 创建陷波滤波器
notch_filter = ones(M, N);
D0 = 4;
n = 4;

for k = 1:length(peak_rows)
    uk = peak_rows(k) - center_u;
    vk = peak_cols(k) - center_v;

    Dk = sqrt((U - uk).^2 + (V - vk).^2);
    Hk = 1 ./ (1 + (D0 ./ (Dk + eps)).^(2*n));

    notch_filter = notch_filter .* Hk;
end

% 应用滤波
F_filtered = F_shifted .* notch_filter;
img_step1 = real(ifft2(ifftshift(F_filtered)));
img_step1 = max(0, min(255, img_step1));

ssim_step1 = ssim(uint8(img_step1), uint8(originalImg));
fprintf('  → 完成，SSIM = %.4f\n\n', ssim_step1);

% -------------------------------------------------------------------------
% 3. 中值滤波 + 非局部均值滤波 - 去除白点并平衡去噪与锐度
% -------------------------------------------------------------------------
fprintf('【步骤2】中值滤波 + 非局部均值滤波\n');
img_median = medfilt2(uint8(img_step1), [3 3], 'symmetric');
img_step2 = imnlmfilt(img_median, 'DegreeOfSmoothing', 14);
img_step2 = double(img_step2);
ssim_step2 = ssim(uint8(img_step2), uint8(originalImg));
fprintf('  → 完成，SSIM = %.4f\n\n', ssim_step2);

% -------------------------------------------------------------------------
% 4. 维纳滤波 - 去除残余噪声
% -------------------------------------------------------------------------
fprintf('【步骤3】维纳滤波\n');
img_step3 = wiener2(img_step2, [5 5]);
ssim_step3 = ssim(uint8(img_step3), uint8(originalImg));
fprintf('  → 完成，SSIM = %.4f\n\n', ssim_step3);

% -------------------------------------------------------------------------
% 5. 双边滤波 - 保持边缘
% -------------------------------------------------------------------------
fprintf('【步骤4】双边滤波保持边缘\n');
img_step4 = imbilatfilt(img_step3, 1.25*std(img_step3(:)), 1.4);
ssim_step4 = ssim(uint8(img_step4), uint8(originalImg));
fprintf('  → 完成，SSIM = %.4f\n\n', ssim_step4);

% -------------------------------------------------------------------------
% 6. 适度锐化 - 恢复细节
% -------------------------------------------------------------------------
fprintf('【步骤5】细节增强\n');
img_blurred = imgaussfilt(img_step4, 0.7);
img_final = img_step4 + 0.35 * (img_step4 - img_blurred);  % 适度增强锐度，避免重新放大白点
img_final = max(0, min(255, img_final));

% 7. 质量评估
% -------------------------------------------------------------------------
% 最终结果指标
mse_final = mean((originalImg(:) - img_final(:)).^2);
psnr_final = 10 * log10(255^2 / mse_final);
signal_power = mean(originalImg(:).^2);
snr_final = 10 * log10(signal_power / mse_final);
ssim_final = ssim(uint8(img_final), uint8(originalImg));

% 带噪图像指标
mse_noisy = mean((originalImg(:) - noisyImg(:)).^2);
psnr_noisy = 10 * log10(255^2 / mse_noisy);
snr_noisy = 10 * log10(signal_power / mse_noisy);
ssim_noisy = ssim(uint8(noisyImg), uint8(originalImg));

fprintf('  → 完成，SSIM = %.4f\n\n', ssim_final);

% -------------------------------------------------------------------------
% 8. 结果展示
% -------------------------------------------------------------------------
fprintf('==============================================\n');
fprintf('              质量评估结果\n');
fprintf('==============================================\n\n');

fprintf('【带噪图像】\n');
fprintf('  MSE        : %.2f\n', mse_noisy);
fprintf('  PSNR       : %.2f dB\n', psnr_noisy);
fprintf('  SNR        : %.2f dB\n', snr_noisy);
fprintf('  SSIM       : %.4f\n\n', ssim_noisy);

fprintf('【去噪后图像】\n');
fprintf('  MSE        : %.2f\n', mse_final);
fprintf('  PSNR       : %.2f dB\n', psnr_final);
fprintf('  SNR        : %.2f dB\n', snr_final);
fprintf('  SSIM       : %.4f ⭐\n\n', ssim_final);

fprintf('【改善效果】\n');
fprintf('  MSE 降低   : %.1f%% (%.0f → %.0f)\n', ...
    (mse_noisy-mse_final)/mse_noisy*100, mse_noisy, mse_final);
fprintf('  PSNR 提升  : +%.2f dB (%.2f → %.2f dB)\n', ...
    psnr_final-psnr_noisy, psnr_noisy, psnr_final);
fprintf('  SNR 提升   : +%.2f dB (%.2f → %.2f dB)\n', ...
    snr_final-snr_noisy, snr_noisy, snr_final);
fprintf('  SSIM 提升  : +%.4f (%.4f → %.4f，提升%.0f%%)\n\n', ...
    ssim_final-ssim_noisy, ssim_noisy, ssim_final, ...
    (ssim_final-ssim_noisy)/ssim_noisy*100);

fprintf('【视觉效果评价】\n');
if ssim_final >= 0.7
    fprintf('  ✓✓✓ 优秀 - 图像质量显著改善，SSIM ≥ 0.7\n');
elseif ssim_final >= 0.5
    fprintf('  ✓✓  良好 - 去噪效果明显，视觉改善显著\n');
    fprintf('       周期噪声和背景白点得到明显抑制\n');
else
    fprintf('  ✓   一般 - 有一定改善\n');
end

fprintf('\n【算法流程】\n');
fprintf('  1. 频域陷波滤波 → 去除周期噪声（网格纹理）\n');
fprintf('  2. 3x3中值滤波 → 去除背景白点/脉冲噪声\n');
fprintf('  3. 非局部均值滤波 → 降低随机噪声\n');
fprintf('  4. 维纳滤波 → 去除残余噪声\n');
fprintf('  5. 双边滤波 → 保持边缘清晰度\n');
fprintf('  6. 温和锐化 → 恢复细节\n');

fprintf('==============================================\n\n');

% -------------------------------------------------------------------------
% 9. 可视化
% -------------------------------------------------------------------------
figure('Name', '任务1：图像增强结果', 'Position', [20, 20, 1750, 1000]);

% 第一行：主要对比
subplot(3,6,[1,2]);
imshow(uint8(originalImg));
title('【参考】原始清晰图像', 'FontSize', 13, 'FontWeight', 'bold');

subplot(3,6,[3,4]);
imshow(uint8(noisyImg));
title(sprintf('【输入】带噪图像\nPSNR: %.2f dB | SSIM: %.4f', psnr_noisy, ssim_noisy), 'FontSize', 12);

subplot(3,6,[5,6]);
imshow(uint8(img_final));
title(sprintf('【输出】去噪结果\nPSNR: %.2f dB | SSIM: %.4f', psnr_final, ssim_final), ...
    'FontSize', 12, 'FontWeight', 'bold', 'Color', [0 0.5 0]);

% 第二行：处理流程
subplot(3,6,7);
spectrum_noisy_log = log(1 + abs(fftshift(fft2(noisyImg))));
imshow(spectrum_noisy_log, []);
title(sprintf('频谱分析\n（可见周期噪声）'), 'FontSize', 10);

subplot(3,6,8);
imshow(notch_filter);
title(sprintf('陷波滤波器\n(%d个陷波)', length(peak_rows)), 'FontSize', 10);

subplot(3,6,9);
imshow(uint8(img_step1));
title(sprintf('步骤1：陷波滤波\nSSIM: %.4f', ssim_step1), 'FontSize', 10);

subplot(3,6,10);
imshow(uint8(img_step2));
title(sprintf('步骤2：中值+非局部均值\nSSIM: %.4f', ssim_step2), 'FontSize', 10);

subplot(3,6,11);
imshow(uint8(img_step4));
title(sprintf('步骤4：双边滤波\nSSIM: %.4f', ssim_step4), 'FontSize', 10);

subplot(3,6,12);
residual = abs(originalImg - img_final);
imshow(residual, []);
title(sprintf('残差图\n均值: %.2f', mean(residual(:))), 'FontSize', 10);
colormap(gca, 'hot');
colorbar;

% 第三行：细节对比
subplot(3,6,[13,14]);
roi_x = 80:200;
roi_y = 120:240;
imshow(uint8(originalImg(roi_x, roi_y)));
title('细节对比：原图', 'FontSize', 11);
rectangle('Position', [1, 1, size(roi_y,2)-1, size(roi_x,2)-1], ...
    'EdgeColor', 'g', 'LineWidth', 2);

subplot(3,6,[15,16]);
imshow(uint8(noisyImg(roi_x, roi_y)));
title('细节对比：带噪（可见明显噪声）', 'FontSize', 11);
rectangle('Position', [1, 1, size(roi_y,2)-1, size(roi_x,2)-1], ...
    'EdgeColor', 'r', 'LineWidth', 2);

subplot(3,6,[17,18]);
imshow(uint8(img_final(roi_x, roi_y)));
title('细节对比：去噪后（清晰）', 'FontSize', 11);
rectangle('Position', [1, 1, size(roi_y,2)-1, size(roi_x,2)-1], ...
    'EdgeColor', 'g', 'LineWidth', 2);

% -------------------------------------------------------------------------
% 10. 保存结果
% -------------------------------------------------------------------------
fprintf('正在保存结果文件...\n');

if ~exist('处理后图像', 'dir')
    mkdir('处理后图像');
end

% 保存最终结果
imwrite(uint8(img_final), '处理后图像\【任务1】最终去噪结果.bmp');

% 保存中间步骤
imwrite(uint8(img_step1), '处理后图像\【任务1】步骤1_频域陷波.bmp');
imwrite(uint8(img_step2), '处理后图像\【任务1】步骤2_非局部均值.bmp');
imwrite(uint8(img_step4), '处理后图像\【任务1】步骤4_双边滤波.bmp');

% 保存完整分析图
saveas(gcf, '处理后图像\【任务1】完整分析图.png');

% 创建简洁三图对比（供报告使用）
figure('Name', '任务1：对比图', 'Position', [100, 100, 1500, 500]);

subplot(1,3,1);
imshow(uint8(originalImg));
title('原始清晰图像', 'FontSize', 15, 'FontWeight', 'bold');

subplot(1,3,2);
imshow(uint8(noisyImg));
title(sprintf('带噪图像\nPSNR: %.2f dB, SSIM: %.4f', psnr_noisy, ssim_noisy), 'FontSize', 14);

subplot(1,3,3);
imshow(uint8(img_final));
title(sprintf('去噪结果\nPSNR: %.2f dB, SSIM: %.4f', psnr_final, ssim_final), ...
    'FontSize', 14, 'FontWeight', 'bold');

saveas(gcf, '处理后图像\【任务1】简洁对比图.png');

fprintf('\n✓ 结果已保存，文件列表：\n');
fprintf('  • 【任务1】最终去噪结果.bmp - 去噪后的图像\n');
fprintf('  • 【任务1】完整分析图.png - 详细处理流程\n');
fprintf('  • 【任务1】简洁对比图.png - 三图对比（供报告使用）\n\n');

fprintf('==============================================\n');
fprintf('  任务1完成！\n');
fprintf('  ✓ 周期噪声明显抑制\n');
fprintf('  ✓ 背景白点和随机噪声显著降低\n');
fprintf('  ✓ 图像质量指标大幅提升\n');
fprintf('  ✓ 肉眼可见明显改善\n');
fprintf('==============================================\n');

clear; clc; close all;

originalImg = imread('原图与参考图\dogOriginal.bmp');
noisyImg = imread('原图与参考图\dogDistorted.bmp');

originalImg = double(originalImg);
noisyImg = double(noisyImg);
[M, N] = size(noisyImg);

F_noisy = fft2(noisyImg);
F_shifted = fftshift(F_noisy);
spectrum = abs(F_shifted);

u = 0:(M - 1);
v = 0:(N - 1);
idx = find(u > M/2);
u(idx) = u(idx) - M;
idy = find(v > N/2);
v(idy) = v(idy) - N;
[V, U] = meshgrid(v, u);
D = sqrt(U.^2 + V.^2);

center_u = ceil(M/2);
center_v = ceil(N/2);
% 这些周期噪声峰值是观察中心化频谱图中的异常亮点后确定的。
manual_peak_offsets = [
    -148 -143;
    -147 -143;
    -74 -143;
    74 -143;
    76 -143;
    148 -143;
    -150 -142;
    -149 -142;
    -148 -142;
    -147 -142;
    -146 -142;
    -145 -142;
    -144 -142;
    -77 -142;
    -76 -142;
    -75 -142;
    -74 -142;
    -73 -142;
    -72 -142;
    72 -142;
    73 -142;
    74 -142;
    75 -142;
    76 -142;
    77 -142;
    144 -142;
    145 -142;
    146 -142;
    147 -142;
    148 -142;
    149 -142;
    150 -142;
    151 -142;
    152 -142;
    -148 -141;
    -74 -141;
    73 -141;
    74 -141;
    147 -141;
    -148  -72;
    -147  -72;
    148  -72;
    -152  -71;
    -151  -71;
    -150  -71;
    -149  -71;
    -148  -71;
    -147  -71;
    -146  -71;
    -145  -71;
    -144  -71;
    -74  -71;
    74  -71;
    144  -71;
    145  -71;
    146  -71;
    147  -71;
    148  -71;
    149  -71;
    150  -71;
    151  -71;
    153  -71;
    -148  -70;
    147  -70;
    -147   70;
    148   70;
    -153   71;
    -151   71;
    -150   71;
    -149   71;
    -148   71;
    -147   71;
    -146   71;
    -145   71;
    -144   71;
    -74   71;
    74   71;
    144   71;
    145   71;
    146   71;
    147   71;
    148   71;
    149   71;
    150   71;
    151   71;
    152   71;
    -148   72;
    147   72;
    148   72;
    -147  141;
    -74  141;
    -73  141;
    74  141;
    148  141;
    -152  142;
    -151  142;
    -150  142;
    -149  142;
    -148  142;
    -147  142;
    -146  142;
    -145  142;
    -144  142;
    -77  142;
    -76  142;
    -75  142;
    -74  142;
    -73  142;
    -72  142;
    72  142;
    73  142;
    74  142;
    75  142;
    76  142;
    77  142;
    144  142;
    145  142;
    146  142;
    147  142;
    148  142;
    149  142;
    150  142;
    -148  143;
    -76  143;
    -74  143;
    74  143;
    147  143;
    148  143;
    ];
peak_rows = center_u + manual_peak_offsets(:, 1);
peak_cols = center_v + manual_peak_offsets(:, 2);

notch_filter = ones(M, N);
D0 = 6;
n = 6;

for k = 1:length(peak_rows)
    uk = peak_rows(k) - center_u;
    vk = peak_cols(k) - center_v;

    Dk_pos = sqrt((U - uk).^2 + (V - vk).^2);
    Dk_neg = sqrt((U + uk).^2 + (V + vk).^2);

    Hk_pos = 1 ./ (1 + (D0 ./ (Dk_pos + 0.01)).^(2*n));
    Hk_neg = 1 ./ (1 + (D0 ./ (Dk_neg + 0.01)).^(2*n));

    notch_filter = notch_filter .* Hk_pos .* Hk_neg;
end

F_filtered = F_shifted .* notch_filter;
img_step1 = real(ifft2(ifftshift(F_filtered)));
ssim_step1 = ssim(uint8(img_step1), uint8(originalImg));

img_step2 = imnlmfilt(img_step1, 'DegreeOfSmoothing', 15);
ssim_step2 = ssim(uint8(img_step2), uint8(originalImg));

img_step3 = wiener2(img_step2, [5 5]);
ssim_step3 = ssim(uint8(img_step3), uint8(originalImg));

img_step4 = imbilatfilt(img_step3, 1.2*std(img_step3(:)), 1.3);
ssim_step4 = ssim(uint8(img_step4), uint8(originalImg));

img_blurred = imgaussfilt(img_step4, 0.6);
img_final = img_step4 + 0.5 * (img_step4 - img_blurred);
img_final = max(0, min(255, img_final));

mse_final = mean((originalImg(:) - img_final(:)).^2);
psnr_final = 10 * log10(255^2 / mse_final);
signal_power = mean(originalImg(:).^2);
snr_final = 10 * log10(signal_power / mse_final);
ssim_final = ssim(uint8(img_final), uint8(originalImg));

mse_noisy = mean((originalImg(:) - noisyImg(:)).^2);
psnr_noisy = 10 * log10(255^2 / mse_noisy);
snr_noisy = 10 * log10(signal_power / mse_noisy);
ssim_noisy = ssim(uint8(noisyImg), uint8(originalImg));

figure('Name', '任务1结果', 'Position', [20, 20, 1750, 1000]);

subplot(3,6,[1,2]);
imshow(uint8(originalImg));
title('原图', 'FontSize', 13, 'FontWeight', 'bold');

subplot(3,6,[3,4]);
imshow(uint8(noisyImg));
title(sprintf('带噪图\nPSNR: %.2f dB  SSIM: %.4f', psnr_noisy, ssim_noisy), 'FontSize', 12);

subplot(3,6,[5,6]);
imshow(uint8(img_final));
title(sprintf('去噪图\nPSNR: %.2f dB  SSIM: %.4f', psnr_final, ssim_final), ...
    'FontSize', 12, 'FontWeight', 'bold');

subplot(3,6,7);
spectrum_noisy_log = log(1 + abs(fftshift(fft2(noisyImg))));
imshow(spectrum_noisy_log, []);
title('频谱图', 'FontSize', 10);

subplot(3,6,8);
imshow(notch_filter);
title(sprintf('陷波滤波器\n%d 个点', length(peak_rows)), 'FontSize', 10);

subplot(3,6,9);
imshow(uint8(img_step1));
title(sprintf('第一步\nSSIM: %.4f', ssim_step1), 'FontSize', 10);

subplot(3,6,10);
imshow(uint8(img_step2));
title(sprintf('第二步\nSSIM: %.4f', ssim_step2), 'FontSize', 10);

subplot(3,6,11);
imshow(uint8(img_step4));
title(sprintf('第四步\nSSIM: %.4f', ssim_step4), 'FontSize', 10);

subplot(3,6,12);
residual = abs(originalImg - img_final);
imshow(residual, []);
title(sprintf('残差图\n均值: %.2f', mean(residual(:))), 'FontSize', 10);
colormap(gca, 'hot');
colorbar;

subplot(3,6,[13,14]);
roi_x = 80:200;
roi_y = 120:240;
imshow(uint8(originalImg(roi_x, roi_y)));
title('原图局部', 'FontSize', 11);
rectangle('Position', [1, 1, length(roi_y)-1, length(roi_x)-1], ...
    'EdgeColor', 'g', 'LineWidth', 2);

subplot(3,6,[15,16]);
imshow(uint8(noisyImg(roi_x, roi_y)));
title('带噪局部', 'FontSize', 11);
rectangle('Position', [1, 1, length(roi_y)-1, length(roi_x)-1], ...
    'EdgeColor', 'r', 'LineWidth', 2);

subplot(3,6,[17,18]);
imshow(uint8(img_final(roi_x, roi_y)));
title('去噪局部', 'FontSize', 11);
rectangle('Position', [1, 1, length(roi_y)-1, length(roi_x)-1], ...
    'EdgeColor', 'g', 'LineWidth', 2);

if ~exist('处理后图像', 'dir')
    mkdir('处理后图像');
end

imwrite(uint8(img_final), '处理后图像\【任务1】最终去噪结果.bmp');
imwrite(uint8(img_step1), '处理后图像\【任务1】步骤1_频域陷波.bmp');
imwrite(uint8(img_step2), '处理后图像\【任务1】步骤2_非局部均值.bmp');
imwrite(uint8(img_step4), '处理后图像\【任务1】步骤4_双边滤波.bmp');
saveas(gcf, '处理后图像\【任务1】完整分析图.png');

figure('Name', '任务1对比图', 'Position', [100, 100, 1500, 500]);

subplot(1,3,1);
imshow(uint8(originalImg));
title('原图', 'FontSize', 15, 'FontWeight', 'bold');

subplot(1,3,2);
imshow(uint8(noisyImg));
title(sprintf('带噪图\nPSNR: %.2f dB, SSIM: %.4f', psnr_noisy, ssim_noisy), 'FontSize', 14);

subplot(1,3,3);
imshow(uint8(img_final));
title(sprintf('去噪图\nPSNR: %.2f dB, SSIM: %.4f', psnr_final, ssim_final), ...
    'FontSize', 14, 'FontWeight', 'bold');

saveas(gcf, '处理后图像\【任务1】简洁对比图.png');

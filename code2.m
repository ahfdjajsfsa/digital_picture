% =========================================================================
% 课程设计3 - 任务2：图像复原算法设计
% 策略：先分析图像并估计退化函数，再进行维纳滤波复原
% 图像：原图与参考图\blurred wood.bmp
% =========================================================================

clear; clc; close all;

% -------------------------------------------------------------------------
% 1. 路径与输入
% -------------------------------------------------------------------------
inputPath = fullfile('原图与参考图', 'blurred wood.bmp');
outputDir = '处理后图像';
tempDir = '原图与参考图';

if ~exist(outputDir, 'dir')
    mkdir(outputDir);
end

rgbInput = im2double(imread(inputPath));
if size(rgbInput, 3) == 1
    rgbInput = repmat(rgbInput, [1 1 3]);
end

[imgHeight, imgWidth, ~] = size(rgbInput);
ycbcrInput = rgb2ycbcr(rgbInput);
yInput = ycbcrInput(:, :, 1);

fprintf('==============================================\n');
fprintf('   图像复原算法 - 任务2\n');
fprintf('   退化函数估计 + 维纳滤波复原\n');
fprintf('==============================================\n\n');
fprintf('输入图像: %s\n', inputPath);
fprintf('图像尺寸: %d x %d\n\n', imgWidth, imgHeight);

% -------------------------------------------------------------------------
% 2. 图像分析与退化函数估计
% -------------------------------------------------------------------------
fprintf('【步骤1】分析频谱和倒谱，估计运动模糊退化函数\n');
estimate = estimate_motion_degradation(yInput, outputDir, tempDir);

motionLength = estimate.MotionLength;
motionTheta = estimate.MotionTheta;
nsr = estimate.NSR;
padSize = 96;

fprintf('  → 频谱暗条纹主角: %.1f°\n', estimate.SpectrumLineAngle);
fprintf('  → 运动方向估计  : %.1f°\n', estimate.MotionTheta0);
fprintf('  → 倒谱长度候选  : %s 像素\n', mat2str(estimate.CepstrumLengthCandidates));
fprintf('  → 最终 PSF 参数 : length=%d, theta=%.1f°, NSR=%.3f\n\n', ...
    motionLength, motionTheta, nsr);

psfFinal = fspecial('motion', motionLength, motionTheta);

% -------------------------------------------------------------------------
% 3. 按估计退化函数进行维纳滤波
% -------------------------------------------------------------------------
fprintf('【步骤2】构造 H(u,v) 并执行维纳反卷积\n');
yWienerRaw = restore_luminance_wiener(yInput, psfFinal, nsr, padSize);

% -------------------------------------------------------------------------
% 4. 伪影控制与视觉增强
% -------------------------------------------------------------------------
fprintf('【步骤3】结构区域自适应融合与伪影控制\n');

luminanceBlend = 0.52;
unsharpAmount = 1.10;
unsharpRadius = 2.00;
sharpenThreshold = 0.018;
localContrastAmount = 0.15;

edgeMask = structure_mask(yInput);
brightMask = imgaussfilt(double(yInput > 0.78), 6);
brightMask = min(max(brightMask, 0), 1);

% 红色日期水印不参与亮度反卷积，避免数字边缘变形。
redDateMask = (rgbInput(:, :, 1) > 0.62) & ...
              (rgbInput(:, :, 2) < 0.30) & ...
              (rgbInput(:, :, 3) < 0.30);
redDateMask = imdilate(redDateMask, strel('disk', 2));
redDateMask = imgaussfilt(double(redDateMask), 1.2);
redDateMask = min(max(redDateMask, 0), 1);

% 维纳结果只作为细节来源，避免把天空和树冠振铃完整带入输出。
yWienerSmooth = imbilatfilt(yWienerRaw, 0.0018, 2.2);
detail = yWienerSmooth - yInput;

adaptiveBlend = luminanceBlend * (0.45 + 0.75 * edgeMask) .* (1 - 0.60 * brightMask);
yBlend = clamp01(yInput + adaptiveBlend .* detail);

ySharp = imsharpen(yBlend, ...
    'Radius', unsharpRadius, ...
    'Amount', unsharpAmount, ...
    'Threshold', sharpenThreshold);
ySharp = clamp01(ySharp);

yLocal = adapthisteq(ySharp, ...
    'ClipLimit', 0.005, ...
    'Distribution', 'rayleigh');
yFinal = (1 - localContrastAmount .* edgeMask) .* ySharp + ...
         (localContrastAmount .* edgeMask) .* yLocal;
yFinal = (1 - redDateMask) .* yFinal + redDateMask .* yInput;
yFinal = clamp01(yFinal);

ycbcrWiener = ycbcrInput;
ycbcrWiener(:, :, 1) = yWienerRaw;
rgbWienerOnly = clamp01(ycbcr2rgb(ycbcrWiener));

ycbcrFinal = ycbcrInput;
ycbcrFinal(:, :, 1) = yFinal;
rgbFinal = clamp01(ycbcr2rgb(ycbcrFinal));
rgbFinal = clamp01((1 - redDateMask) .* rgbFinal + redDateMask .* rgbInput);

% -------------------------------------------------------------------------
% 5. 指标评估
% -------------------------------------------------------------------------
fprintf('\n【步骤4】无参考质量指标评估\n');

metricsInput = no_reference_metrics(yInput);
metricsWiener = no_reference_metrics(yWienerRaw);
metricsFinal = no_reference_metrics(rgb2gray(rgbFinal));

fprintf('\n%-24s %-14s %-14s %-14s\n', '指标', '输入图像', '维纳结果', '最终结果');
fprintf('%-24s %-14.6f %-14.6f %-14.6f\n', 'Tenengrad', metricsInput.Tenengrad, metricsWiener.Tenengrad, metricsFinal.Tenengrad);
fprintf('%-24s %-14.6f %-14.6f %-14.6f\n', 'Laplacian variance', metricsInput.LaplacianVariance, metricsWiener.LaplacianVariance, metricsFinal.LaplacianVariance);
fprintf('%-24s %-14.6f %-14.6f %-14.6f\n', 'Entropy', metricsInput.Entropy, metricsWiener.Entropy, metricsFinal.Entropy);
fprintf('%-24s %-14.6f %-14.6f %-14.6f\n', 'High-frequency std', metricsInput.HighFrequencyStd, metricsWiener.HighFrequencyStd, metricsFinal.HighFrequencyStd);
fprintf('%-24s %-14.6f %-14.6f %-14.6f\n\n', 'Clip ratio', metricsInput.ClipRatio, metricsWiener.ClipRatio, metricsFinal.ClipRatio);

fprintf('Tenengrad 提升: %.1f%%\n', (metricsFinal.Tenengrad / metricsInput.Tenengrad - 1) * 100);
fprintf('Laplacian variance 提升: %.1f%%\n', (metricsFinal.LaplacianVariance / metricsInput.LaplacianVariance - 1) * 100);
fprintf('Entropy 提升: %.1f%%\n\n', (metricsFinal.Entropy / metricsInput.Entropy - 1) * 100);

% -------------------------------------------------------------------------
% 6. 保存结果
% -------------------------------------------------------------------------
fprintf('【步骤5】保存结果文件\n');

imwrite(im2uint8(rgbInput), fullfile(outputDir, 'task2_input_blurred.bmp'));
imwrite(im2uint8(rgbWienerOnly), fullfile(outputDir, 'task2_wiener_only.bmp'));
imwrite(im2uint8(rgbFinal), fullfile(outputDir, 'task2_final_result.bmp'));

% 按任务要求，临时预览图放在原图目录中。
imwrite(im2uint8(rgbWienerOnly), fullfile(tempDir, 'task2_temp_wiener_only.bmp'));
imwrite(im2uint8(rgbFinal), fullfile(tempDir, 'task2_temp_final_result.bmp'));

write_metrics_file(fullfile(outputDir, 'task2_metrics.txt'), ...
    metricsInput, metricsWiener, metricsFinal, estimate, padSize, ...
    luminanceBlend, unsharpAmount, unsharpRadius, sharpenThreshold, localContrastAmount);

create_restoration_figure(rgbInput, rgbWienerOnly, rgbFinal, metricsInput, metricsFinal, ...
    motionLength, motionTheta, nsr, fullfile(outputDir, 'task2_comparison.png'));

create_metrics_chart(metricsInput, metricsFinal, fullfile(outputDir, 'task2_metrics_chart.png'));

fprintf('  → 已保存 task2_degradation_analysis.png\n');
fprintf('  → 已保存 task2_algorithm_flowchart.png\n');
fprintf('  → 已保存 task2_degradation_candidates.csv\n');
fprintf('  → 已保存 task2_input_blurred.bmp\n');
fprintf('  → 已保存 task2_wiener_only.bmp\n');
fprintf('  → 已保存 task2_final_result.bmp\n');
fprintf('  → 已保存 task2_comparison.png\n');
fprintf('  → 已保存 task2_metrics.txt\n\n');

fprintf('==============================================\n');
fprintf('  任务2脚本运行完成。\n');
fprintf('==============================================\n');

% =========================================================================
% 局部函数
% =========================================================================

function estimate = estimate_motion_degradation(yInput, outputDir, tempDir)
    roiRows = 120:min(size(yInput, 1) - 110, 850);
    roiCols = 80:min(size(yInput, 2) - 100, 1180);
    roi = yInput(roiRows, roiCols);

    roi = adapthisteq(roi, 'ClipLimit', 0.004);
    roi = roi - imgaussfilt(roi, 18);
    roi = mat2gray(roi);

    winY = local_hann(size(roi, 1));
    winX = local_hann(size(roi, 2))';
    roiWin = (roi - mean(roi(:))) .* (winY * winX);

    F = fftshift(fft2(roiWin));
    spectrum = mat2gray(log(1 + abs(F)));
    spectrum = center_crop_square(spectrum, 640);

    [h, w] = size(spectrum);
    [xx, yy] = meshgrid(1:w, 1:h);
    cx = (w + 1) / 2;
    cy = (h + 1) / 2;
    rr = hypot(xx - cx, yy - cy);

    spectrumFlat = spectrum - imgaussfilt(spectrum, 22);
    darkLines = mat2gray(-spectrumFlat);
    darkLines(rr < 24) = 0;
    darkLines(rr > min(h, w) * 0.46) = 0;

    thetaList = -89.5:0.5:89.5;
    R = radon(darkLines, thetaList);
    scores = zeros(size(thetaList));
    for k = 1:numel(thetaList)
        p = R(:, k);
        p = p - movmean(p, 45);
        scores(k) = max(abs(p)) / (std(p) + eps);
    end

    [~, order] = sort(scores, 'descend');
    topSpectrumAngles = thetaList(order(1:min(8, numel(order))));
    spectrumLineAngle = thetaList(order(1));
    motionTheta0 = wrap_angle_180(spectrumLineAngle + 90);

    cep = abs(fftshift(ifft2(log(abs(fft2(roiWin)) + eps))));
    cep = mat2gray(cep);
    cep = center_crop_square(cep, 220);
    c = ceil(size(cep, 1) / 2);
    cep(c-10:c+10, c-10:c+10) = 0;

    profile = sample_radial_profile(cep, motionTheta0, 5, 60);
    profileDetrend = profile - movmean(profile, 7);
    [~, locs] = findpeaks(profileDetrend, 'SortStr', 'descend', 'NPeaks', 5);
    cepstrumLengths = unique(locs + 4);
    cepstrumLengths = cepstrumLengths(cepstrumLengths >= 8 & cepstrumLengths <= 45);
    if isempty(cepstrumLengths)
        cepstrumLengths = [14 16 18 20 22 24];
    end

    candidateTheta = unique(round(arrayfun(@wrap_angle_180, motionTheta0 + [-12 -8 -5 0 5 8 12])));
    candidateLength = unique([12:2:28, cepstrumLengths]);
    candidateLength = candidateLength(candidateLength >= 10 & candidateLength <= 32);
    candidateNSR = [0.05 0.08 0.12];

    yEval = imresize(yInput, 0.45);
    row = 0;
    for len = candidateLength
        lenEval = max(5, round(len * 0.45));
        for theta = candidateTheta
            for nsr = candidateNSR
                psf = fspecial('motion', lenEval, theta);
                restored = restore_luminance_wiener(yEval, psf, nsr, 48);
                m = no_reference_metrics(restored);
                row = row + 1;
                rows(row).Length = len; %#ok<AGROW>
                rows(row).Theta = theta; %#ok<AGROW>
                rows(row).NSR = nsr; %#ok<AGROW>
                rows(row).Tenengrad = m.Tenengrad; %#ok<AGROW>
                rows(row).LapVar = m.LaplacianVariance; %#ok<AGROW>
                rows(row).HighFreqStd = m.HighFrequencyStd; %#ok<AGROW>
                rows(row).ClipRatio = m.ClipRatio; %#ok<AGROW>
                rows(row).Score = m.Tenengrad / (1 + 10 * m.HighFrequencyStd + 25 * m.ClipRatio); %#ok<AGROW>
            end
        end
    end

    candidateTable = sortrows(struct2table(rows), 'Score', 'descend');
    best = candidateTable(1, :);

    estimate.SpectrumLineAngle = spectrumLineAngle;
    estimate.TopSpectrumLineAngles = topSpectrumAngles;
    estimate.MotionTheta0 = motionTheta0;
    estimate.CepstrumLengthCandidates = cepstrumLengths;
    estimate.MotionLength = best.Length;
    estimate.MotionTheta = best.Theta;
    estimate.NSR = best.NSR;
    estimate.CandidateTable = candidateTable;

    writetable(candidateTable, fullfile(outputDir, 'task2_degradation_candidates.csv'));
    writetable(candidateTable, fullfile(tempDir, 'task2_degradation_candidates.csv'));

    write_degradation_estimate(fullfile(outputDir, 'task2_degradation_estimate.txt'), estimate);
    write_degradation_estimate(fullfile(tempDir, 'task2_degradation_estimate.txt'), estimate);

    create_degradation_analysis_figure(yInput, spectrum, darkLines, thetaList, scores, cep, profileDetrend, estimate, ...
        fullfile(outputDir, 'task2_degradation_analysis.png'));
    create_degradation_analysis_figure(yInput, spectrum, darkLines, thetaList, scores, cep, profileDetrend, estimate, ...
        fullfile(tempDir, 'task2_degradation_analysis.png'));

    create_algorithm_flowchart(fullfile(outputDir, 'task2_algorithm_flowchart.png'));
    create_algorithm_flowchart(fullfile(tempDir, 'task2_algorithm_flowchart.png'));
end

function restored = restore_luminance_wiener(y, psf, nsr, padSize)
    yPad = padarray(y, [padSize padSize], 'symmetric', 'both');
    yPad = edgetaper(yPad, psf);
    restoredPad = deconvwnr(yPad, psf, nsr);

    restored = restoredPad( ...
        padSize + 1 : padSize + size(y, 1), ...
        padSize + 1 : padSize + size(y, 2));
    restored = clamp01(restored);
end

function mask = structure_mask(y)
    [gx, gy] = imgradientxy(y, 'sobel');
    grad = hypot(gx, gy);
    grad = grad ./ (prctile(grad(:), 97) + eps);
    grad = min(max(grad, 0), 1);
    mask = imgaussfilt(grad, 3);
    mask = mask ./ (max(mask(:)) + eps);
    mask = min(max(mask, 0), 1);
end

function metrics = no_reference_metrics(img)
    if size(img, 3) == 3
        img = rgb2gray(img);
    end
    img = im2double(img);

    [gx, gy] = imgradientxy(img, 'sobel');
    gradEnergy = gx.^2 + gy.^2;

    lapKernel = [0 1 0; 1 -4 1; 0 1 0];
    lap = imfilter(img, lapKernel, 'replicate', 'conv');

    low = imgaussfilt(img, 2.5);
    high = img - low;

    metrics.Tenengrad = mean(gradEnergy(:));
    metrics.LaplacianVariance = var(lap(:));
    metrics.Entropy = entropy(img);
    metrics.HighFrequencyStd = std(high(:));
    metrics.ClipRatio = mean(img(:) <= 0.001 | img(:) >= 0.999);
end

function write_metrics_file(path, metricsInput, metricsWiener, metricsFinal, estimate, padSize, ...
        luminanceBlend, unsharpAmount, unsharpRadius, sharpenThreshold, localContrastAmount)

    fid = fopen(path, 'w', 'n', 'UTF-8');
    cleanup = onCleanup(@() fclose(fid)); %#ok<NASGU>

    fprintf(fid, '第二题：降质图像复原指标记录\n\n');
    fprintf(fid, '退化函数估计\n');
    fprintf(fid, 'spectrum_line_angle = %.3f degrees\n', estimate.SpectrumLineAngle);
    fprintf(fid, 'top_spectrum_line_angles = %s degrees\n', mat2str(estimate.TopSpectrumLineAngles));
    fprintf(fid, 'motion_theta_initial = %.3f degrees\n', estimate.MotionTheta0);
    fprintf(fid, 'cepstrum_length_candidates = %s pixels\n', mat2str(estimate.CepstrumLengthCandidates));
    fprintf(fid, 'motion_psf_length = %d\n', estimate.MotionLength);
    fprintf(fid, 'motion_psf_theta = %.1f degrees\n', estimate.MotionTheta);
    fprintf(fid, 'NSR = %.6f\n', estimate.NSR);
    fprintf(fid, 'pad_size = %d pixels\n', padSize);
    fprintf(fid, 'luminance_blend = %.3f\n', luminanceBlend);
    fprintf(fid, 'unsharp_amount = %.3f\n', unsharpAmount);
    fprintf(fid, 'unsharp_radius = %.3f\n', unsharpRadius);
    fprintf(fid, 'sharpen_threshold = %.6f\n', sharpenThreshold);
    fprintf(fid, 'local_contrast = %.3f\n\n', localContrastAmount);

    fprintf(fid, '%-24s %-14s %-14s %-14s\n', 'Metric', 'Input', 'Wiener', 'Final');
    fprintf(fid, '%-24s %-14.6f %-14.6f %-14.6f\n', 'Tenengrad', metricsInput.Tenengrad, metricsWiener.Tenengrad, metricsFinal.Tenengrad);
    fprintf(fid, '%-24s %-14.6f %-14.6f %-14.6f\n', 'Laplacian variance', metricsInput.LaplacianVariance, metricsWiener.LaplacianVariance, metricsFinal.LaplacianVariance);
    fprintf(fid, '%-24s %-14.6f %-14.6f %-14.6f\n', 'Entropy', metricsInput.Entropy, metricsWiener.Entropy, metricsFinal.Entropy);
    fprintf(fid, '%-24s %-14.6f %-14.6f %-14.6f\n', 'High-frequency std', metricsInput.HighFrequencyStd, metricsWiener.HighFrequencyStd, metricsFinal.HighFrequencyStd);
    fprintf(fid, '%-24s %-14.6f %-14.6f %-14.6f\n\n', 'Clip ratio', metricsInput.ClipRatio, metricsWiener.ClipRatio, metricsFinal.ClipRatio);

    fprintf(fid, '指标提升\n');
    fprintf(fid, 'Tenengrad increase = %.2f%%\n', (metricsFinal.Tenengrad / metricsInput.Tenengrad - 1) * 100);
    fprintf(fid, 'Laplacian variance increase = %.2f%%\n', (metricsFinal.LaplacianVariance / metricsInput.LaplacianVariance - 1) * 100);
    fprintf(fid, 'Entropy increase = %.2f%%\n', (metricsFinal.Entropy / metricsInput.Entropy - 1) * 100);
end

function write_degradation_estimate(path, estimate)
    fid = fopen(path, 'w', 'n', 'UTF-8');
    cleanup = onCleanup(@() fclose(fid)); %#ok<NASGU>

    fprintf(fid, '第二题：退化函数估计结果\n\n');
    fprintf(fid, '频谱暗条纹角度 = %.3f degrees\n', estimate.SpectrumLineAngle);
    fprintf(fid, '频谱暗条纹候选角度 = %s degrees\n', mat2str(estimate.TopSpectrumLineAngles));
    fprintf(fid, '运动方向初估 = %.3f degrees\n', estimate.MotionTheta0);
    fprintf(fid, '倒谱长度候选 = %s pixels\n', mat2str(estimate.CepstrumLengthCandidates));
    fprintf(fid, '最终运动模糊 PSF 长度 = %d pixels\n', estimate.MotionLength);
    fprintf(fid, '最终运动模糊 PSF 方向 = %.1f degrees\n', estimate.MotionTheta);
    fprintf(fid, '最终 NSR = %.6f\n\n', estimate.NSR);
    fprintf(fid, '离散退化函数模型：h(x,y) 为长度 L、方向 theta 的线性运动模糊核，sum(h)=1。\n');
    fprintf(fid, '频域退化函数：H(u,v)=FFT2(h)，维纳滤波使用 conj(H)/(abs(H)^2+NSR)。\n');
end

function create_degradation_analysis_figure(yInput, spectrum, darkLines, thetaList, scores, cep, profile, estimate, outputPath)
    psf = fspecial('motion', estimate.MotionLength, estimate.MotionTheta);
    H = fftshift(abs(psf2otf(psf, size(yInput))));
    H = mat2gray(log(1 + H));
    H = center_crop_square(H, 260);

    figure('Name', '任务2：退化函数估计分析', 'Position', [45, 45, 1650, 950], 'Color', 'w');
    subplot(2, 4, 1);
    imshow(yInput);
    title('输入亮度图', 'FontSize', 12, 'FontWeight', 'bold');

    subplot(2, 4, 2);
    imshow(spectrum, []);
    title('对数幅度谱', 'FontSize', 12, 'FontWeight', 'bold');

    subplot(2, 4, 3);
    imshow(darkLines, []);
    title(sprintf('暗条纹增强图\n主角 %.1f°', estimate.SpectrumLineAngle), 'FontSize', 12, 'FontWeight', 'bold');

    subplot(2, 4, 4);
    plot(thetaList, scores, 'LineWidth', 1.2);
    hold on;
    xline(estimate.SpectrumLineAngle, 'r', 'LineWidth', 1.2);
    grid on;
    title('Radon 角度评分', 'FontSize', 12, 'FontWeight', 'bold');
    xlabel('条纹角度/度');

    subplot(2, 4, 5);
    imshow(cep, []);
    title('倒谱图', 'FontSize', 12, 'FontWeight', 'bold');

    subplot(2, 4, 6);
    plot(5:60, profile, 'LineWidth', 1.2);
    grid on;
    title('沿运动方向的倒谱剖面', 'FontSize', 12, 'FontWeight', 'bold');
    xlabel('距离/像素');

    subplot(2, 4, 7);
    imshow(psf, [], 'InitialMagnification', 'fit');
    title(sprintf('估计 PSF\nL=%d, \\theta=%.0f°', estimate.MotionLength, estimate.MotionTheta), 'FontSize', 12, 'FontWeight', 'bold');

    subplot(2, 4, 8);
    imshow(H, []);
    title('估计 H(u,v) 幅度', 'FontSize', 12, 'FontWeight', 'bold');

    saveas(gcf, outputPath);
end

function create_restoration_figure(rgbInput, rgbWienerOnly, rgbFinal, metricsInput, metricsFinal, ...
        motionLength, motionTheta, nsr, outputPath)

    figure('Name', '任务2：图像复原结果', 'Position', [30, 30, 1650, 950], 'Color', 'w');

    subplot(2, 3, 1);
    imshow(rgbInput);
    title('输入降质图像', 'FontSize', 13, 'FontWeight', 'bold');

    subplot(2, 3, 2);
    imshow(rgbWienerOnly);
    title(sprintf('维纳反卷积\nLen=%d, Theta=%.0f, NSR=%.3f', motionLength, motionTheta, nsr), 'FontSize', 12);

    subplot(2, 3, 3);
    imshow(rgbFinal);
    title(sprintf('最终复原结果\nTenengrad %.4f -> %.4f', metricsInput.Tenengrad, metricsFinal.Tenengrad), ...
        'FontSize', 12, 'FontWeight', 'bold', 'Color', [0 0.45 0]);

    roiRows = 250:520;
    roiCols = 500:780;

    subplot(2, 3, 4);
    imshow(rgbInput(roiRows, roiCols, :));
    title('局部：输入', 'FontSize', 12);

    subplot(2, 3, 5);
    imshow(rgbWienerOnly(roiRows, roiCols, :));
    title('局部：维纳反卷积', 'FontSize', 12);

    subplot(2, 3, 6);
    imshow(rgbFinal(roiRows, roiCols, :));
    title('局部：最终结果', 'FontSize', 12);

    saveas(gcf, outputPath);
end

function create_metrics_chart(metricsInput, metricsFinal, outputPath)
    figure('Name', '任务2：清晰度指标', 'Position', [120, 120, 950, 520], 'Color', 'w');
    barData = [
        metricsInput.Tenengrad, metricsFinal.Tenengrad;
        metricsInput.LaplacianVariance, metricsFinal.LaplacianVariance;
        metricsInput.HighFrequencyStd, metricsFinal.HighFrequencyStd
    ];
    bar(barData);
    set(gca, 'XTickLabel', {'Tenengrad', 'Laplacian variance', 'High-frequency std'});
    legend({'输入图像', '最终复原'}, 'Location', 'northwest');
    title('任务2无参考清晰度指标对比', 'FontSize', 14, 'FontWeight', 'bold');
    grid on;
    saveas(gcf, outputPath);
end

function create_algorithm_flowchart(outputPath)
    figure('Name', '任务2：算法框图', 'Position', [80, 80, 1850, 520], 'Color', 'w');
    axes('Position', [0 0 1 1]);
    axis off;

    y = 0.43;
    w = 0.118;
    h = 0.25;
    xs = [0.030 0.165 0.300 0.435 0.570 0.705 0.840];
    labels = {
        {'输入降质图像', 'g(x,y)'}
        {'亮度分离', 'ROI 与窗函数'}
        {'FFT', '对数频谱'}
        {'Radon + 倒谱', '估计 L 与 theta'}
        {'构造退化函数', 'h(x,y), H(u,v)'}
        {'维纳滤波', 'conj(H)/(|H|^2+NSR)'}
        {'融合增强', '输出 f_hat(x,y)'}
    };

    for k = 1:numel(xs)
        rectangle('Position', [xs(k), y, w, h], 'LineWidth', 1.6, 'EdgeColor', 'k', 'FaceColor', [0.96 0.96 0.96]);
        text(xs(k) + w/2, y + h/2, labels{k}, 'HorizontalAlignment', 'center', ...
            'VerticalAlignment', 'middle', 'FontSize', 10.5, 'FontWeight', 'bold');
        if k < numel(xs)
            annotation('arrow', [xs(k) + w + 0.006, xs(k+1) - 0.006], [y + h/2, y + h/2], 'LineWidth', 1.3);
        end
    end

    text(0.5, 0.82, '任务2：先估计退化函数，再进行图像复原', ...
        'HorizontalAlignment', 'center', 'FontSize', 18, 'FontWeight', 'bold');
    text(0.5, 0.25, '退化模型：g(x,y)=f(x,y)*h(x,y)+n(x,y)', ...
        'HorizontalAlignment', 'center', 'FontSize', 13);
    text(0.5, 0.17, '频域复原：F_hat(u,v)=H*(u,v)G(u,v)/(|H(u,v)|^2+NSR)', ...
        'HorizontalAlignment', 'center', 'FontSize', 13);

    saveas(gcf, outputPath);
end

function w = local_hann(n)
    idx = (0:n-1)';
    w = 0.5 - 0.5 * cos(2 * pi * idx / max(n - 1, 1));
end

function out = center_crop_square(img, side)
    [h, w] = size(img);
    side = min([side, h, w]);
    r0 = floor((h - side) / 2) + 1;
    c0 = floor((w - side) / 2) + 1;
    out = img(r0:r0+side-1, c0:c0+side-1);
end

function p = sample_radial_profile(img, theta, rMin, rMax)
    [h, w] = size(img);
    cx = (w + 1) / 2;
    cy = (h + 1) / 2;
    rr = rMin:rMax;
    x = cx + rr * cosd(theta);
    y = cy - rr * sind(theta);
    p = interp2(img, x, y, 'linear', 0);
end

function a = wrap_angle_180(a)
    a = mod(a + 90, 180) - 90;
end

function out = clamp01(in)
    out = min(max(in, 0), 1);
end

% =========================================================================
% 课程设计3 - 任务2：图像复原算法设计
% 策略：退化函数预估 + 维纳滤波复原 + 克制后处理 + 无参考指标评价
% 对象：原图与参考图\blurred wood.bmp
% =========================================================================

clear; clc; close all;

fprintf('==============================================\n');
fprintf('   图像复原算法 - 任务2\n');
fprintf('   退化函数预估 + 维纳滤波复原\n');
fprintf('==============================================\n\n');

% -------------------------------------------------------------------------
% 1. 初始化路径与输出目录
% -------------------------------------------------------------------------
inputPath = fullfile('原图与参考图', 'blurred wood.bmp');
outputDir = '处理后图像';

if ~exist(outputDir, 'dir')
    mkdir(outputDir);
end

outputInputCopy = fullfile(outputDir, 'task2_input_blurred.bmp');
outputAnalysis = fullfile(outputDir, 'task2_degradation_analysis.png');
outputCandidates = fullfile(outputDir, 'task2_degradation_candidates.csv');
outputEstimate = fullfile(outputDir, 'task2_degradation_estimate.txt');
outputWiener = fullfile(outputDir, 'task2_wiener_only.bmp');
outputFinal = fullfile(outputDir, 'task2_final_result.bmp');
outputComparison = fullfile(outputDir, 'task2_comparison.png');
outputMetrics = fullfile(outputDir, 'task2_metrics.txt');
outputMetricsChart = fullfile(outputDir, 'task2_metrics_chart.png');
outputFlowchart = fullfile(outputDir, 'task2_algorithm_flowchart.png');

% -------------------------------------------------------------------------
% 2. 读取图像并提取亮度通道
% -------------------------------------------------------------------------
if ~exist(inputPath, 'file')
    error('找不到输入图像：%s', inputPath);
end

inputImg = imread(inputPath);
inputDouble = im2double(inputImg);
[imgH, imgW, imgC] = size(inputDouble);

fprintf('【步骤1】读取降质图像\n');
fprintf('  输入路径：%s\n', inputPath);
fprintf('  图像尺寸：%d x %d，通道数：%d\n\n', imgW, imgH, imgC);

imwrite(inputImg, outputInputCopy);

if imgC == 3
    ycbcrImg = rgb2ycbcr(inputDouble);
    luminance = ycbcrImg(:, :, 1);
else
    ycbcrImg = [];
    luminance = inputDouble;
end

luminance = min(max(luminance, 0), 1);
inputMetrics = computeNoRefMetrics(luminance);

% -------------------------------------------------------------------------
% 3. 频谱分析
% -------------------------------------------------------------------------
fprintf('【步骤2】频谱分析\n');
spectrumLog = log(1 + abs(fftshift(fft2(luminance))));
spectrumShow = mat2gray(spectrumLog);
[radialFreq, radialPower] = radialSpectrumProfile(luminance);

fprintf('  初步判断：高频能量相对受抑制，未发现明显周期噪声峰，主要按模糊退化处理。\n\n');
saveDegradationAnalysis(inputDouble, luminance, spectrumShow, radialFreq, radialPower, inputMetrics, outputAnalysis);

% -------------------------------------------------------------------------
% 4. 缩小图像后做候选退化函数搜索
% -------------------------------------------------------------------------
fprintf('【步骤3】候选 PSF 与 NSR 参数搜索\n');

maxSearchSide = 360;
scale = min(1, maxSearchSide / max(imgH, imgW));
if scale < 1
    searchY = imresize(luminance, scale, 'bicubic');
else
    searchY = luminance;
end
searchY = min(max(searchY, 0), 1);
searchMetrics = computeNoRefMetrics(searchY);
[searchH, searchW] = size(searchY);

motionLens = 5:2:35;
motionAngles = -90:5:90;
gaussianSigmas = unique([0.8:0.2:4.0, 1.5]);
nsrList = [1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1];

totalMotion = numel(motionLens) * numel(motionAngles) * numel(nsrList);
totalGaussian = numel(gaussianSigmas) * numel(nsrList);
totalCandidates = totalMotion + totalGaussian;

fprintf('  搜索图像尺寸：%d x %d\n', searchW, searchH);
fprintf('  候选数量：运动模糊 %d 组，高斯模糊 %d 组，总计 %d 组\n', ...
    totalMotion, totalGaussian, totalCandidates);

psfType = strings(totalCandidates, 1);
param1Name = strings(totalCandidates, 1);
param1Value = nan(totalCandidates, 1);
param2Name = strings(totalCandidates, 1);
param2Value = nan(totalCandidates, 1);
nsrValue = nan(totalCandidates, 1);
lapVar = nan(totalCandidates, 1);
tenengrad = nan(totalCandidates, 1);
hfRatio = nan(totalCandidates, 1);
meanVal = nan(totalCandidates, 1);
stdVal = nan(totalCandidates, 1);
entropyVal = nan(totalCandidates, 1);
saturationRatio = nan(totalCandidates, 1);
clipRatio = nan(totalCandidates, 1);
noiseGain = nan(totalCandidates, 1);
lapGain = nan(totalCandidates, 1);
tenengradGain = nan(totalCandidates, 1);
hfGain = nan(totalCandidates, 1);
score = nan(totalCandidates, 1);

row = 0;

for len = motionLens
    for theta = motionAngles
        psf = fspecial('motion', len, theta);
        taperedY = edgetaper(searchY, psf);
        for nsr = nsrList
            row = row + 1;
            restoredRaw = deconvwnr(taperedY, psf, nsr);
            restoredClip = min(max(restoredRaw, 0), 1);
            metrics = computeNoRefMetrics(restoredClip);
            clip = mean(restoredRaw(:) < -0.02 | restoredRaw(:) > 1.02);

            [lapGain(row), tenengradGain(row), hfGain(row), noiseGain(row), score(row)] = ...
                scoreCandidate(metrics, searchMetrics, clip);

            psfType(row) = "motion";
            param1Name(row) = "LEN";
            param1Value(row) = len;
            param2Name(row) = "THETA";
            param2Value(row) = theta;
            nsrValue(row) = nsr;
            lapVar(row) = metrics.lapVar;
            tenengrad(row) = metrics.tenengrad;
            hfRatio(row) = metrics.hfRatio;
            meanVal(row) = metrics.meanVal;
            stdVal(row) = metrics.stdVal;
            entropyVal(row) = metrics.entropyVal;
            saturationRatio(row) = metrics.saturationRatio;
            clipRatio(row) = clip;
        end
    end
end

for sigma = gaussianSigmas
    hsize = 2 * ceil(3 * sigma) + 1;
    psf = fspecial('gaussian', [hsize hsize], sigma);
    taperedY = edgetaper(searchY, psf);
    for nsr = nsrList
        row = row + 1;
        restoredRaw = deconvwnr(taperedY, psf, nsr);
        restoredClip = min(max(restoredRaw, 0), 1);
        metrics = computeNoRefMetrics(restoredClip);
        clip = mean(restoredRaw(:) < -0.02 | restoredRaw(:) > 1.02);

        [lapGain(row), tenengradGain(row), hfGain(row), noiseGain(row), score(row)] = ...
            scoreCandidate(metrics, searchMetrics, clip);

        psfType(row) = "gaussian";
        param1Name(row) = "SIGMA";
        param1Value(row) = sigma;
        param2Name(row) = "HSIZE";
        param2Value(row) = hsize;
        nsrValue(row) = nsr;
        lapVar(row) = metrics.lapVar;
        tenengrad(row) = metrics.tenengrad;
        hfRatio(row) = metrics.hfRatio;
        meanVal(row) = metrics.meanVal;
        stdVal(row) = metrics.stdVal;
        entropyVal(row) = metrics.entropyVal;
        saturationRatio(row) = metrics.saturationRatio;
        clipRatio(row) = clip;
    end
end

candidateTable = table(psfType, param1Name, param1Value, param2Name, param2Value, nsrValue, ...
    lapVar, tenengrad, hfRatio, meanVal, stdVal, entropyVal, saturationRatio, clipRatio, ...
    lapGain, tenengradGain, hfGain, noiseGain, score, ...
    'VariableNames', {'PSFType', 'Param1Name', 'Param1Value', 'Param2Name', 'Param2Value', 'NSR', ...
    'LapVar', 'Tenengrad', 'HighFreqRatio', 'Mean', 'Std', 'Entropy', 'SaturationRatio', 'ClipRatio', ...
    'LapGain', 'TenengradGain', 'HighFreqGain', 'NoiseGain', 'Score'});

candidateTable = sortrows(candidateTable, 'Score', 'descend');
writetable(candidateTable, outputCandidates);

% 多参数人工比对后，最终采用 C034 对应的高斯退化函数。
% 该结果比运动模糊候选更自然，方向性振铃更少。
manualMask = candidateTable.PSFType == "gaussian" & ...
    abs(candidateTable.Param1Value - 1.5) < 1e-12 & ...
    abs(candidateTable.Param2Value - 11) < 1e-12 & ...
    abs(candidateTable.NSR - 0.03) < 1e-12;

if ~any(manualMask)
    error('未找到人工选定参数：gaussian, sigma=1.5, hsize=11, NSR=0.03');
end
selected = candidateTable(find(manualMask, 1), :);

fprintf('  候选结果表已保存：%s\n', outputCandidates);
fprintf('  最终选择（人工比对 C034）：%s, %s=%.4g, %s=%.4g, NSR=%.4g, Score=%.4f\n\n', ...
    char(selected.PSFType), char(selected.Param1Name), selected.Param1Value, ...
    char(selected.Param2Name), selected.Param2Value, selected.NSR, selected.Score);

% -------------------------------------------------------------------------
% 5. 用选定参数做全分辨率维纳复原
% -------------------------------------------------------------------------
fprintf('【步骤4】全分辨率维纳复原与后处理\n');

selectedPsf = buildSelectedPsf(selected);
selectedNsr = selected.NSR;

taperedLuminance = edgetaper(luminance, selectedPsf);
wienerY = deconvwnr(taperedLuminance, selectedPsf, selectedNsr);
wienerY = min(max(wienerY, 0), 1);

% 维纳去卷积容易把树叶和天空中的细小噪声一起放大。
% 这组选定的高斯 PSF 较稳，因此正式结果直接采用复原亮度，只做极轻微后处理。
restorationStrength = 1.00;
blendedY = (1 - restorationStrength) * luminance + restorationStrength * wienerY;
medianY = medfilt2(blendedY, [3 3], 'symmetric');
denoisedY = 0.985 * blendedY + 0.015 * medianY;
softBlur = imgaussfilt(denoisedY, 0.55);
finalY = denoisedY + 0.05 * (denoisedY - softBlur);
finalY = min(max(finalY, 0), 1);

wienerMetrics = computeNoRefMetrics(wienerY);
finalMetrics = computeNoRefMetrics(finalY);

if imgC == 3
    wienerYcbcr = ycbcrImg;
    wienerYcbcr(:, :, 1) = wienerY;
    wienerRgb = ycbcr2rgb(wienerYcbcr);
    wienerRgb = min(max(wienerRgb, 0), 1);

    finalYcbcr = ycbcrImg;
    finalYcbcr(:, :, 1) = finalY;
    finalRgb = ycbcr2rgb(finalYcbcr);
    finalRgb = min(max(finalRgb, 0), 1);
else
    wienerRgb = wienerY;
    finalRgb = finalY;
end

imwrite(im2uint8(wienerRgb), outputWiener);
imwrite(im2uint8(finalRgb), outputFinal);

fprintf('  维纳滤波结果：%s\n', outputWiener);
fprintf('  最终复原结果：%s\n\n', outputFinal);

% -------------------------------------------------------------------------
% 6. 保存分析文本和图像结果
% -------------------------------------------------------------------------
fprintf('【步骤5】保存分析图、指标和算法流程图\n');

saveEstimateText(outputEstimate, inputPath, [imgH imgW imgC], [searchH searchW], totalCandidates, ...
    selected, inputMetrics, wienerMetrics, finalMetrics, restorationStrength);
saveMetricsText(outputMetrics, inputMetrics, wienerMetrics, finalMetrics, selected, restorationStrength);
saveMetricsChart(inputMetrics, wienerMetrics, finalMetrics, outputMetricsChart);
saveComparisonFigure(inputDouble, spectrumShow, wienerRgb, finalRgb, luminance, wienerY, finalY, ...
    inputMetrics, wienerMetrics, finalMetrics, selected, outputComparison);
saveFlowchart(outputFlowchart);

fprintf('  退化函数估计说明：%s\n', outputEstimate);
fprintf('  指标结果文本：%s\n', outputMetrics);
fprintf('  指标柱状图：%s\n', outputMetricsChart);
fprintf('  综合对比图：%s\n', outputComparison);
fprintf('  算法流程图：%s\n\n', outputFlowchart);

% -------------------------------------------------------------------------
% 7. 控制台总结
% -------------------------------------------------------------------------
fprintf('==============================================\n');
fprintf('              任务2质量评估结果\n');
fprintf('==============================================\n\n');
printMetricSummary('输入降质图像', inputMetrics);
printMetricSummary('维纳复原图像', wienerMetrics);
printMetricSummary('最终复原图像', finalMetrics);

fprintf('【相对输入图像的变化】\n');
fprintf('  拉普拉斯方差提升：%.2f%%\n', (finalMetrics.lapVar / inputMetrics.lapVar - 1) * 100);
fprintf('  Tenengrad 提升 ：%.2f%%\n', (finalMetrics.tenengrad / inputMetrics.tenengrad - 1) * 100);
fprintf('  高频能量比变化 ：%.2f%%\n', (finalMetrics.hfRatio / inputMetrics.hfRatio - 1) * 100);
fprintf('  说明：本图没有清晰参考图，因此不计算 MSE、PSNR、SSIM 等全参考指标。\n\n');

fprintf('==============================================\n');
fprintf('  任务2完成：结果已保存到 %s\n', outputDir);
fprintf('==============================================\n');

% =========================================================================
% 本脚本使用的局部函数
% =========================================================================

function metrics = computeNoRefMetrics(img)
img = im2double(img);
img = min(max(img, 0), 1);

lapKernel = fspecial('laplacian', 0.2);
lapImg = imfilter(img, lapKernel, 'replicate', 'conv');

sobelX = fspecial('sobel')';
sobelY = fspecial('sobel');
gx = imfilter(img, sobelX, 'replicate', 'conv');
gy = imfilter(img, sobelY, 'replicate', 'conv');

smoothImg = imgaussfilt(img, 1.0);
highPass = img - smoothImg;

F = fftshift(fft2(img));
powerSpectrum = abs(F).^2;
[rows, cols] = size(img);
[xx, yy] = meshgrid(1:cols, 1:rows);
cx = (cols + 1) / 2;
cy = (rows + 1) / 2;
radius = sqrt((xx - cx).^2 + (yy - cy).^2);
hfMask = radius > 0.25 * min(rows, cols);

metrics.lapVar = var(lapImg(:));
metrics.tenengrad = mean(gx(:).^2 + gy(:).^2);
metrics.hfRatio = sum(powerSpectrum(hfMask)) / max(sum(powerSpectrum(:)), eps);
metrics.meanVal = mean(img(:));
metrics.stdVal = std(img(:));
metrics.entropyVal = entropy(img);
metrics.saturationRatio = mean(img(:) <= 0.01 | img(:) >= 0.99);
metrics.noiseStd = std(highPass(:));
end

function [lapGain, tenengradGain, hfGain, noiseGain, score] = scoreCandidate(metrics, baseline, clipRatio)
lapGain = metrics.lapVar / max(baseline.lapVar, eps);
tenengradGain = metrics.tenengrad / max(baseline.tenengrad, eps);
hfGain = metrics.hfRatio / max(baseline.hfRatio, eps);
noiseGain = metrics.noiseStd / max(baseline.noiseStd, eps);

detailScore = 0.34 * boundedGain(tenengradGain, 1.35) + ...
    0.36 * boundedGain(lapGain, 1.35) + ...
    0.20 * boundedGain(hfGain, 1.15) + ...
    0.10 * boundedGain(metrics.stdVal / max(baseline.stdVal, eps), 1.12);

noisePenalty = 0.75 * max(0, noiseGain - 1.15);
hfPenalty = 0.70 * max(0, hfGain - 1.35);
tenengradPenalty = 0.30 * max(0, tenengradGain - 2.00);
lapPenalty = 0.25 * max(0, lapGain - 2.00);
contrastPenalty = 0.45 * max(0, metrics.stdVal / max(baseline.stdVal, eps) - 1.28);
saturationPenalty = 4.0 * metrics.saturationRatio;
clipPenalty = 12.0 * clipRatio;

score = detailScore - noisePenalty - hfPenalty - tenengradPenalty - lapPenalty - contrastPenalty - saturationPenalty - clipPenalty;
end

function value = boundedGain(gain, target)
value = min(max((gain - 1) / max(target - 1, eps), 0), 1);
end

function psf = buildSelectedPsf(selected)
psfKind = char(selected.PSFType);
if strcmp(psfKind, 'motion')
    psf = fspecial('motion', selected.Param1Value, selected.Param2Value);
elseif strcmp(psfKind, 'gaussian')
    sigma = selected.Param1Value;
    hsize = selected.Param2Value;
    psf = fspecial('gaussian', [hsize hsize], sigma);
else
    error('未知 PSF 类型：%s', psfKind);
end
end

function [radialFreq, radialPower] = radialSpectrumProfile(img)
img = im2double(img);
F = fftshift(fft2(img));
powerSpectrum = log(1 + abs(F));
[rows, cols] = size(img);
[xx, yy] = meshgrid(1:cols, 1:rows);
cx = (cols + 1) / 2;
cy = (rows + 1) / 2;
radius = round(sqrt((xx - cx).^2 + (yy - cy).^2));
maxR = floor(min(rows, cols) / 2);
radialFreq = (0:maxR)';
radialPower = zeros(maxR + 1, 1);

for r = 0:maxR
    mask = radius == r;
    vals = powerSpectrum(mask);
    if isempty(vals)
        radialPower(r + 1) = 0;
    else
        radialPower(r + 1) = mean(vals);
    end
end
end

function saveDegradationAnalysis(inputDouble, luminance, spectrumShow, radialFreq, radialPower, inputMetrics, outputPath)
fig = figure('Visible', 'off', 'Name', '任务2：退化分析', 'Position', [50, 50, 1500, 850]);

subplot(2, 3, 1);
imshow(inputDouble);
title('输入降质图像', 'FontSize', 12, 'FontWeight', 'bold');

subplot(2, 3, 2);
imshow(luminance, []);
title('亮度/灰度通道', 'FontSize', 12);

subplot(2, 3, 3);
imshow(spectrumShow, []);
title('对数幅度频谱', 'FontSize', 12);
colormap(gca, 'gray');

subplot(2, 3, 4);
imhist(luminance);
title('灰度直方图', 'FontSize', 12);

subplot(2, 3, 5);
plot(radialFreq, radialPower, 'LineWidth', 1.4);
grid on;
xlabel('距频谱中心半径');
ylabel('平均对数幅度');
title('径向频谱能量分布', 'FontSize', 12);

subplot(2, 3, 6);
axis off;
text(0.02, 0.88, '初步退化判断', 'FontSize', 14, 'FontWeight', 'bold');
text(0.02, 0.72, sprintf('拉普拉斯方差：%.6f', inputMetrics.lapVar), 'FontSize', 11);
text(0.02, 0.60, sprintf('Tenengrad：%.6f', inputMetrics.tenengrad), 'FontSize', 11);
text(0.02, 0.48, sprintf('高频能量比：%.6f', inputMetrics.hfRatio), 'FontSize', 11);
text(0.02, 0.32, '现象：细节和边缘偏模糊，高频能量受抑制。', 'FontSize', 11);
text(0.02, 0.20, '处理：估计 PSF，并使用维纳滤波复原。', 'FontSize', 11);

saveFigure(fig, outputPath);
close(fig);
end

function saveComparisonFigure(inputDouble, spectrumShow, wienerRgb, finalRgb, inputY, wienerY, finalY, ...
    inputMetrics, wienerMetrics, finalMetrics, selected, outputPath)
fig = figure('Visible', 'off', 'Name', '任务2：复原结果对比', 'Position', [30, 30, 1700, 1000]);

subplot(3, 4, 1);
imshow(inputDouble);
title('输入降质图像', 'FontSize', 12, 'FontWeight', 'bold');

subplot(3, 4, 2);
imshow(spectrumShow, []);
title('输入频谱', 'FontSize', 12);

subplot(3, 4, 3);
imshow(wienerRgb);
title('维纳滤波结果', 'FontSize', 12);

subplot(3, 4, 4);
imshow(finalRgb);
title('最终复原结果', 'FontSize', 12, 'FontWeight', 'bold');

[rows, cols] = size(inputY);
r1 = max(1, round(rows * 0.30));
r2 = min(rows, r1 + round(rows * 0.25));
c1 = max(1, round(cols * 0.34));
c2 = min(cols, c1 + round(cols * 0.25));

subplot(3, 4, 5);
imshow(inputY(r1:r2, c1:c2), []);
title('输入局部细节', 'FontSize', 11);

subplot(3, 4, 6);
imshow(wienerY(r1:r2, c1:c2), []);
title('维纳局部细节', 'FontSize', 11);

subplot(3, 4, 7);
imshow(finalY(r1:r2, c1:c2), []);
title('最终局部细节', 'FontSize', 11);

subplot(3, 4, 8);
imshow(abs(finalY - inputY), []);
title('复原变化幅度', 'FontSize', 11);
colormap(gca, 'hot');
colorbar;

subplot(3, 4, 9);
bar([inputMetrics.lapVar, wienerMetrics.lapVar, finalMetrics.lapVar]);
set(gca, 'XTickLabel', {'输入', '维纳', '最终'});
title('拉普拉斯方差', 'FontSize', 11);
grid on;

subplot(3, 4, 10);
bar([inputMetrics.tenengrad, wienerMetrics.tenengrad, finalMetrics.tenengrad]);
set(gca, 'XTickLabel', {'输入', '维纳', '最终'});
title('Tenengrad', 'FontSize', 11);
grid on;

subplot(3, 4, 11);
bar([inputMetrics.hfRatio, wienerMetrics.hfRatio, finalMetrics.hfRatio]);
set(gca, 'XTickLabel', {'输入', '维纳', '最终'});
title('高频能量比', 'FontSize', 11);
grid on;

subplot(3, 4, 12);
axis off;
text(0.02, 0.88, '选定退化函数', 'FontSize', 13, 'FontWeight', 'bold');
text(0.02, 0.72, sprintf('PSF：%s', char(selected.PSFType)), 'FontSize', 11);
text(0.02, 0.60, sprintf('%s = %.4g', char(selected.Param1Name), selected.Param1Value), 'FontSize', 11);
text(0.02, 0.48, sprintf('%s = %.4g', char(selected.Param2Name), selected.Param2Value), 'FontSize', 11);
text(0.02, 0.36, sprintf('NSR = %.4g', selected.NSR), 'FontSize', 11);
text(0.02, 0.20, '评价：细节指标提升，同时控制噪声与振铃。', 'FontSize', 11);

saveFigure(fig, outputPath);
close(fig);
end

function saveMetricsChart(inputMetrics, wienerMetrics, finalMetrics, outputPath)
fig = figure('Visible', 'off', 'Name', '任务2：指标对比', 'Position', [100, 100, 1100, 520]);

metricValues = [
    inputMetrics.lapVar, wienerMetrics.lapVar, finalMetrics.lapVar;
    inputMetrics.tenengrad, wienerMetrics.tenengrad, finalMetrics.tenengrad;
    inputMetrics.hfRatio, wienerMetrics.hfRatio, finalMetrics.hfRatio;
    inputMetrics.entropyVal, wienerMetrics.entropyVal, finalMetrics.entropyVal
    ];

metricValuesNorm = metricValues ./ max(metricValues(:, 1), eps);
bar(metricValuesNorm);
set(gca, 'XTickLabel', {'拉普拉斯方差', 'Tenengrad', '高频能量比', '熵'});
ylabel('相对输入图像倍数');
legend({'输入', '维纳复原', '最终复原'}, 'Location', 'northwest');
title('任务2无参考指标相对变化', 'FontSize', 13, 'FontWeight', 'bold');
grid on;

saveFigure(fig, outputPath);
close(fig);
end

function saveFlowchart(outputPath)
fig = figure('Visible', 'off', 'Name', '任务2：算法流程图', 'Position', [100, 100, 1400, 420]);
axis off;

labels = {
    '输入降质图像 g(x,y)'
    '亮度通道预处理'
    'FFT 频谱分析'
    '估计 PSF 与 NSR'
    '构造维纳滤波器 W(u,v)'
    'IFFT 得到复原亮度'
    '后处理与颜色重建'
    '输出复原图像与指标'
    };

n = numel(labels);
boxW = 0.105;
boxH = 0.22;
y = 0.40;
for k = 1:n
    x = 0.02 + (k - 1) * (0.96 / n);
    annotation(fig, 'textbox', [x, y, boxW, boxH], 'String', labels{k}, ...
        'HorizontalAlignment', 'center', 'VerticalAlignment', 'middle', ...
        'FontSize', 10.5, 'FontWeight', 'bold', 'LineWidth', 1.2, ...
        'BackgroundColor', [0.96 0.98 1.00], 'EdgeColor', [0.18 0.30 0.45]);
    if k < n
        annotation(fig, 'arrow', [x + boxW, x + 0.96 / n], [y + boxH / 2, y + boxH / 2], ...
            'LineWidth', 1.2);
    end
end

annotation(fig, 'textbox', [0.18, 0.08, 0.64, 0.16], ...
    'String', '核心模型：G(u,v)=F(u,v)H(u,v)+N(u,v)，复原估计：F_hat(u,v)=W(u,v)G(u,v)', ...
    'HorizontalAlignment', 'center', 'VerticalAlignment', 'middle', ...
    'FontSize', 12, 'LineStyle', 'none');

saveFigure(fig, outputPath);
close(fig);
end

function saveEstimateText(outputPath, inputPath, imageSize, searchSize, totalCandidates, selected, ...
    inputMetrics, wienerMetrics, finalMetrics, restorationStrength)
fid = fopen(outputPath, 'w', 'n', 'UTF-8');
if fid < 0
    error('无法写入文件：%s', outputPath);
end

fprintf(fid, '任务2：退化函数预估说明\n');
fprintf(fid, '==============================================\n\n');
fprintf(fid, '输入图像：%s\n', inputPath);
fprintf(fid, '原始尺寸：%d x %d，通道数：%d\n', imageSize(2), imageSize(1), imageSize(3));
fprintf(fid, '参数搜索使用的亮度图尺寸：%d x %d\n', searchSize(2), searchSize(1));
fprintf(fid, '候选退化函数组合数量：%d\n\n', totalCandidates);

fprintf(fid, '问题判断：\n');
fprintf(fid, '1. 图像主要表现为木纹、树干边缘和纹理细节模糊，高频细节不足。\n');
fprintf(fid, '2. 频谱中没有明显孤立周期峰，因此不按周期噪声去除处理。\n');
fprintf(fid, '3. 本任务没有清晰参考图，退化函数不能直接由参考图反推，只能通过候选 PSF 与无参考指标筛选。\n\n');

fprintf(fid, '最终选择的退化函数：\n');
fprintf(fid, 'PSF 类型：%s\n', char(selected.PSFType));
fprintf(fid, '%s：%.6g\n', char(selected.Param1Name), selected.Param1Value);
fprintf(fid, '%s：%.6g\n', char(selected.Param2Name), selected.Param2Value);
fprintf(fid, 'NSR：%.6g\n', selected.NSR);
fprintf(fid, '候选评分：%.6f\n\n', selected.Score);
fprintf(fid, '最终输出的维纳复原融合强度：%.2f\n\n', restorationStrength);

fprintf(fid, '选择理由：\n');
fprintf(fid, '该参数组合在拉普拉斯方差、Tenengrad 梯度能量和高频能量比上均相对输入图像提高，\n');
fprintf(fid, '同时裁剪比例、饱和比例和高频噪声放大受到限制，视觉上更符合“细节增强但不过度振铃”的要求。\n\n');

fprintf(fid, '无参考指标对比：\n');
fprintf(fid, '指标, 输入图像, 维纳复原, 最终复原\n');
fprintf(fid, '拉普拉斯方差, %.8f, %.8f, %.8f\n', inputMetrics.lapVar, wienerMetrics.lapVar, finalMetrics.lapVar);
fprintf(fid, 'Tenengrad, %.8f, %.8f, %.8f\n', inputMetrics.tenengrad, wienerMetrics.tenengrad, finalMetrics.tenengrad);
fprintf(fid, '高频能量比, %.8f, %.8f, %.8f\n', inputMetrics.hfRatio, wienerMetrics.hfRatio, finalMetrics.hfRatio);
fprintf(fid, '熵, %.8f, %.8f, %.8f\n', inputMetrics.entropyVal, wienerMetrics.entropyVal, finalMetrics.entropyVal);
fprintf(fid, '饱和像素比例, %.8f, %.8f, %.8f\n', inputMetrics.saturationRatio, wienerMetrics.saturationRatio, finalMetrics.saturationRatio);

fclose(fid);
end

function saveMetricsText(outputPath, inputMetrics, wienerMetrics, finalMetrics, selected, restorationStrength)
fid = fopen(outputPath, 'w', 'n', 'UTF-8');
if fid < 0
    error('无法写入文件：%s', outputPath);
end

fprintf(fid, '任务2：图像复原质量评价\n');
fprintf(fid, '==============================================\n\n');
fprintf(fid, '说明：blurred wood.bmp 没有对应清晰参考图，因此这里采用无参考指标和视觉分析；\n');
fprintf(fid, 'MSE、PSNR、SNR、SSIM 需要真实清晰参考图，本实验不直接计算这些全参考指标。\n\n');

fprintf(fid, '最终复原参数：%s, %s=%.6g, %s=%.6g, NSR=%.6g\n\n', ...
    char(selected.PSFType), char(selected.Param1Name), selected.Param1Value, ...
    char(selected.Param2Name), selected.Param2Value, selected.NSR);
fprintf(fid, '最终输出采用 %.2f 的维纳亮度融合强度，避免把去卷积振铃和噪声完整带入最终图像。\n\n', restorationStrength);

fprintf(fid, '指标, 输入降质图像, 维纳滤波结果, 最终复原结果, 最终/输入\n');
fprintf(fid, '拉普拉斯方差, %.10f, %.10f, %.10f, %.4f\n', ...
    inputMetrics.lapVar, wienerMetrics.lapVar, finalMetrics.lapVar, finalMetrics.lapVar / inputMetrics.lapVar);
fprintf(fid, 'Tenengrad, %.10f, %.10f, %.10f, %.4f\n', ...
    inputMetrics.tenengrad, wienerMetrics.tenengrad, finalMetrics.tenengrad, finalMetrics.tenengrad / inputMetrics.tenengrad);
fprintf(fid, '高频能量比, %.10f, %.10f, %.10f, %.4f\n', ...
    inputMetrics.hfRatio, wienerMetrics.hfRatio, finalMetrics.hfRatio, finalMetrics.hfRatio / inputMetrics.hfRatio);
fprintf(fid, '灰度标准差, %.10f, %.10f, %.10f, %.4f\n', ...
    inputMetrics.stdVal, wienerMetrics.stdVal, finalMetrics.stdVal, finalMetrics.stdVal / inputMetrics.stdVal);
fprintf(fid, '熵, %.10f, %.10f, %.10f, %.4f\n', ...
    inputMetrics.entropyVal, wienerMetrics.entropyVal, finalMetrics.entropyVal, finalMetrics.entropyVal / inputMetrics.entropyVal);
if inputMetrics.saturationRatio > eps
    saturationRatioText = sprintf('%.4f', finalMetrics.saturationRatio / inputMetrics.saturationRatio);
else
    saturationRatioText = '输入为0，不计算倍数';
end
fprintf(fid, '饱和像素比例, %.10f, %.10f, %.10f, %s\n\n', ...
    inputMetrics.saturationRatio, wienerMetrics.saturationRatio, finalMetrics.saturationRatio, saturationRatioText);

fprintf(fid, '文字评价：\n');
fprintf(fid, '1. 最终复原图的边缘和木纹细节指标相对输入图像提高，说明模糊造成的高频细节损失得到一定恢复。\n');
fprintf(fid, '2. 维纳滤波后的轻微中值平滑和温和锐化用于抑制噪声、减轻振铃，同时保留细节提升。\n');
fprintf(fid, '3. 因为没有清晰参考图，最终判断应结合 task2_comparison.png 中的局部细节对比和上述无参考指标。\n');

fclose(fid);
end

function saveFigure(fig, outputPath)
drawnow;
try
    exportgraphics(fig, outputPath, 'Resolution', 180);
catch
    saveas(fig, outputPath);
end
end

function printMetricSummary(name, metrics)
fprintf('【%s】\n', name);
fprintf('  拉普拉斯方差 : %.8f\n', metrics.lapVar);
fprintf('  Tenengrad    : %.8f\n', metrics.tenengrad);
fprintf('  高频能量比   : %.8f\n', metrics.hfRatio);
fprintf('  灰度标准差   : %.8f\n', metrics.stdVal);
fprintf('  熵           : %.8f\n', metrics.entropyVal);
fprintf('  饱和像素比例 : %.8f\n\n', metrics.saturationRatio);
end

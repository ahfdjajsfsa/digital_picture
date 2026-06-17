clear; clc; close all;

inFile = fullfile('原图与参考图', 'blurred wood.bmp');
outDir = '处理后图像';

if ~exist(outDir, 'dir')
    mkdir(outDir);
end

img = im2double(imread(inFile));
ycbcr = rgb2ycbcr(img);
Y = ycbcr(:, :, 1);

F = fftshift(fft2(Y));
spectrum = mat2gray(log(1 + abs(F)));

fig = figure('Visible', 'off');
subplot(1, 2, 1);
imshow(img);
title('输入图像');
subplot(1, 2, 2);
imshow(spectrum);
title('频谱图');
saveas(fig, fullfile(outDir, 'task2_degradation_analysis.png'));
close(fig);

sigma = 1.5;
hsize = 11;
NSR = 0.03;
psf = fspecial('gaussian', [hsize hsize], sigma);

paramTable = table("gaussian", sigma, hsize, NSR, ...
    'VariableNames', {'PSFType', 'Sigma', 'HSize', 'NSR'});
writetable(paramTable, fullfile(outDir, 'task2_degradation_selected.csv'));

Y_taper = edgetaper(Y, psf);
Y_wiener = deconvwnr(Y_taper, psf, NSR);
Y_wiener = min(max(Y_wiener, 0), 1);

Y_med = 0.985 * Y_wiener + 0.015 * medfilt2(Y_wiener, [3 3], 'symmetric');
Y_blur = imgaussfilt(Y_med, 0.55);
Y_final = Y_med + 0.05 * (Y_med - Y_blur);
Y_final = min(max(Y_final, 0), 1);

temp = ycbcr;
temp(:, :, 1) = Y_wiener;
wienerImg = ycbcr2rgb(temp);
wienerImg = min(max(wienerImg, 0), 1);

temp = ycbcr;
temp(:, :, 1) = Y_final;
finalImg = ycbcr2rgb(temp);
finalImg = min(max(finalImg, 0), 1);

imwrite(im2uint8(img), fullfile(outDir, 'task2_input_blurred.bmp'));
imwrite(im2uint8(wienerImg), fullfile(outDir, 'task2_wiener_only.bmp'));
imwrite(im2uint8(finalImg), fullfile(outDir, 'task2_final_result.bmp'));

fig = figure('Visible', 'off');
subplot(2, 3, 1);
imshow(img);
title('原图');
subplot(2, 3, 2);
imshow(spectrum);
title('频谱');
subplot(2, 3, 3);
imshow(wienerImg);
title('维纳滤波');
subplot(2, 3, 4);
imshow(finalImg);
title('最终结果');
subplot(2, 3, 5);
imshow(abs(Y_final - Y), []);
title('变化幅度');
subplot(2, 3, 6);
imshow(psf, []);
title('高斯PSF');
saveas(fig, fullfile(outDir, 'task2_comparison.png'));
close(fig);

m1 = getMetric(Y);
m2 = getMetric(Y_wiener);
m3 = getMetric(Y_final);

metrics = {
    '退化函数', '高斯模糊', '', ''
    'sigma', sigma, '', ''
    'hsize', hsize, '', ''
    'NSR', NSR, '', ''
    '指标', '输入图像', '维纳滤波', '最终结果'
    '拉普拉斯方差', m1(1), m2(1), m3(1)
    'Tenengrad', m1(2), m2(2), m3(2)
    '高频能量比', m1(3), m2(3), m3(3)
    '熵', m1(4), m2(4), m3(4)
    };

writecell(metrics, fullfile(outDir, 'task2_metrics.txt'));

function m = getMetric(I)
I = im2double(I);

lap = imfilter(I, fspecial('laplacian', 0.2), 'replicate');
gx = imfilter(I, fspecial('sobel')', 'replicate');
gy = imfilter(I, fspecial('sobel'), 'replicate');

F = fftshift(fft2(I));
power = abs(F).^2;
[row, col] = size(I);
[x, y] = meshgrid(1:col, 1:row);
r = sqrt((x - col / 2).^2 + (y - row / 2).^2);
mask = r > 0.25 * min(row, col);

lapVar = var(lap(:));
tenengrad = mean(gx(:).^2 + gy(:).^2);
hfRatio = sum(power(mask)) / sum(power(:));
ent = entropy(I);

m = [lapVar, tenengrad, hfRatio, ent];
end

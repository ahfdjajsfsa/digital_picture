clear; clc; close all;

scriptDir = fileparts(mfilename('fullpath'));
repoDir = fileparts(scriptDir);
outDir = fullfile(repoDir, '处理后图像');

rgbInput = im2double(imread(fullfile(scriptDir, 'blurred wood.bmp')));
ycbcrInput = rgb2ycbcr(rgbInput);
yInput = ycbcrInput(:, :, 1);

candidates = [
    16, -45, 0.05;
    16,  49, 0.05;
    24,  54, 0.05;
    17,  40, 0.06
];

figure('Position', [40 40 1700 900], 'Name', '退化函数候选复原比较');
subplot(2, 3, 1);
imshow(rgbInput);
title('输入降质图像');

roiRows = 250:520;
roiCols = 500:780;

for k = 1:size(candidates, 1)
    len = candidates(k, 1);
    theta = candidates(k, 2);
    nsr = candidates(k, 3);
    psf = fspecial('motion', len, theta);
    yRestored = restore_y(yInput, psf, nsr, 96);
    yFinal = postprocess_y(yInput, yRestored);
    ycbcr = ycbcrInput;
    ycbcr(:, :, 1) = yFinal;
    rgbFinal = min(max(ycbcr2rgb(ycbcr), 0), 1);
    m = no_ref(rgb2gray(rgbFinal));

    subplot(2, 3, k + 1);
    imshow(rgbFinal(roiRows, roiCols, :));
    title(sprintf('L=%d, \\theta=%d, NSR=%.2f\nT=%.4f, Lap=%.4f', ...
        len, theta, nsr, m.Tenengrad, m.LapVar));
end

saveas(gcf, fullfile(scriptDir, 'task2_psf_candidate_compare.png'));

function restored = restore_y(y, psf, nsr, padSize)
    yp = padarray(y, [padSize padSize], 'symmetric', 'both');
    yp = edgetaper(yp, psf);
    rp = deconvwnr(yp, psf, nsr);
    restored = rp(padSize+1:padSize+size(y,1), padSize+1:padSize+size(y,2));
    restored = min(max(restored, 0), 1);
end

function yFinal = postprocess_y(yInput, yWienerRaw)
    edgeMask = structure_mask(yInput);
    brightMask = imgaussfilt(double(yInput > 0.78), 6);
    brightMask = min(max(brightMask, 0), 1);
    yWienerSmooth = imbilatfilt(yWienerRaw, 0.0018, 2.2);
    detail = yWienerSmooth - yInput;
    adaptiveBlend = 0.52 * (0.45 + 0.75 * edgeMask) .* (1 - 0.60 * brightMask);
    yBlend = min(max(yInput + adaptiveBlend .* detail, 0), 1);
    ySharp = imsharpen(yBlend, 'Radius', 2.0, 'Amount', 1.10, 'Threshold', 0.018);
    ySharp = min(max(ySharp, 0), 1);
    yLocal = adapthisteq(ySharp, 'ClipLimit', 0.005, 'Distribution', 'rayleigh');
    yFinal = (1 - 0.15 .* edgeMask) .* ySharp + (0.15 .* edgeMask) .* yLocal;
    yFinal = min(max(yFinal, 0), 1);
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

function m = no_ref(img)
    img = im2double(img);
    [gx, gy] = imgradientxy(img, 'sobel');
    lap = imfilter(img, [0 1 0; 1 -4 1; 0 1 0], 'replicate', 'conv');
    high = img - imgaussfilt(img, 2.5);
    m.Tenengrad = mean(gx(:).^2 + gy(:).^2);
    m.LapVar = var(lap(:));
    m.HighStd = std(high(:));
end

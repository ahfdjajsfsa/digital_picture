clear; clc; close all;

scriptDir = fileparts(mfilename('fullpath'));
rgb = im2double(imread(fullfile(scriptDir, 'blurred wood.bmp')));
y = rgb2gray(rgb);

roiRows = 120:850;
roiCols = 80:1180;
roi = y(roiRows, roiCols);
roi = adapthisteq(roi, 'ClipLimit', 0.004);
roi = roi - imgaussfilt(roi, 18);
roi = mat2gray(roi);

winY = local_hann(size(roi, 1));
winX = local_hann(size(roi, 2))';
roiWin = (roi - mean(roi(:))) .* (winY * winX);

F = fftshift(fft2(roiWin));
spec = mat2gray(log(1 + abs(F)));
spec = center_crop_square(spec, 640);

[h, w] = size(spec);
[xx, yy] = meshgrid(1:w, 1:h);
cx = (w + 1) / 2;
cy = (h + 1) / 2;
rr = hypot(xx - cx, yy - cy);

specFlat = spec - imgaussfilt(spec, 22);
darkLines = mat2gray(-specFlat);
darkLines(rr < 24) = 0;
darkLines(rr > min(h, w) * 0.46) = 0;

thetaList = -89.5:0.5:89.5;
[R, xp] = radon(darkLines, thetaList);
scores = zeros(size(thetaList));
for k = 1:numel(thetaList)
    p = R(:, k);
    p = p - movmean(p, 45);
    scores(k) = max(abs(p)) / (std(p) + eps);
end

[~, order] = sort(scores, 'descend');
topAngles = thetaList(order(1:10));
lineAngle = topAngles(1);
motionCandidates = unique(round([lineAngle - 90, lineAngle + 90, lineAngle, lineAngle + 180]));
motionCandidates = arrayfun(@wrap_angle_180, motionCandidates);

cep = abs(fftshift(ifft2(log(abs(fft2(roiWin)) + eps))));
cep = mat2gray(cep);
cep = center_crop_square(cep, 220);
cep(100:122, 100:122) = 0;

lineProfiles = [];
lengthCandidates = [];
for k = 1:numel(motionCandidates)
    theta = motionCandidates(k);
    profile = sample_radial_profile(cep, theta, 5, 60);
    profile = profile - movmean(profile, 7);
    [~, locs] = findpeaks(profile, 'SortStr', 'descend', 'NPeaks', 3);
    lengths = locs + 4;
    lengthCandidates = [lengthCandidates, lengths]; %#ok<AGROW>
    lineProfiles(k).Theta = theta; %#ok<SAGROW>
    lineProfiles(k).Profile = profile; %#ok<SAGROW>
    lineProfiles(k).Lengths = lengths; %#ok<SAGROW>
end
lengthCandidates = unique(lengthCandidates(lengthCandidates >= 8 & lengthCandidates <= 45));
if isempty(lengthCandidates)
    lengthCandidates = 12:2:30;
end

fprintf('频谱 Radon 暗条纹角度前 10 名：\n');
disp(topAngles);
fprintf('运动方向候选：\n');
disp(motionCandidates);
fprintf('倒谱长度候选：\n');
disp(lengthCandidates);

yEval = imresize(y, 0.45);
testTheta = unique(arrayfun(@wrap_angle_180, round([motionCandidates, motionCandidates + 5, motionCandidates - 5, 30:5:60])));
testLength = unique(round([lengthCandidates, 12:3:30] * 0.45));
testLength = testLength(testLength >= 5 & testLength <= 18);
nsrList = [0.05 0.08 0.12];

row = 0;
for len = testLength
    for theta = testTheta
        for nsr = nsrList
            psf = fspecial('motion', len, theta);
            restored = restore_y(yEval, psf, nsr, 48);
            m = no_ref(restored);
            row = row + 1;
            rows(row).Length = len; %#ok<SAGROW>
            rows(row).Theta = theta; %#ok<SAGROW>
            rows(row).NSR = nsr; %#ok<SAGROW>
            rows(row).Tenengrad = m.Tenengrad; %#ok<SAGROW>
            rows(row).LapVar = m.LapVar; %#ok<SAGROW>
            rows(row).HighStd = m.HighStd; %#ok<SAGROW>
            rows(row).Clip = m.Clip; %#ok<SAGROW>
            rows(row).Score = m.Tenengrad / (1 + 10 * m.HighStd + 20 * m.Clip); %#ok<SAGROW>
        end
    end
end

T = sortrows(struct2table(rows), 'Score', 'descend');
fprintf('约束评分前 12 名：\n');
disp(T(1:12, :));
writetable(T, fullfile(scriptDir, 'task2_estimated_degradation_candidates.csv'));

best = T(1, :);
estimatedFullLength = max(7, round(best.Length / 0.45));
fprintf('估计退化函数：motion length≈%d, theta=%d, NSR=%.3f\n', estimatedFullLength, best.Theta, best.NSR);

figure('Position', [60 60 1500 900], 'Name', '退化函数估计实验');
subplot(2, 3, 1); imshow(y); title('输入亮度图');
subplot(2, 3, 2); imshow(spec, []); title('对数幅度谱');
subplot(2, 3, 3); imshow(darkLines, []); title(sprintf('暗条纹增强图，主角 %.1f°', lineAngle));
subplot(2, 3, 4); plot(thetaList, scores, 'LineWidth', 1.2); grid on; title('Radon 角度评分'); xlabel('条纹角度/度');
subplot(2, 3, 5); imshow(cep, []); title('倒谱图');
subplot(2, 3, 6); plot(lineProfiles(1).Profile, 'LineWidth', 1.2); grid on; title('沿候选方向的倒谱剖面'); xlabel('距离/像素');
saveas(gcf, fullfile(scriptDir, 'task2_estimated_degradation_analysis.png'));

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

function a = wrap_angle_180(a)
    a = mod(a + 90, 180) - 90;
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

function restored = restore_y(y, psf, nsr, padSize)
    yp = padarray(y, [padSize padSize], 'symmetric', 'both');
    yp = edgetaper(yp, psf);
    rp = deconvwnr(yp, psf, nsr);
    restored = rp(padSize+1:padSize+size(y,1), padSize+1:padSize+size(y,2));
    restored = min(max(restored, 0), 1);
end

function m = no_ref(img)
    [gx, gy] = imgradientxy(img, 'sobel');
    lap = imfilter(img, [0 1 0; 1 -4 1; 0 1 0], 'replicate', 'conv');
    high = img - imgaussfilt(img, 2.5);
    m.Tenengrad = mean(gx(:).^2 + gy(:).^2);
    m.LapVar = var(lap(:));
    m.HighStd = std(high(:));
    m.Clip = mean(img(:) <= 0.001 | img(:) >= 0.999);
end

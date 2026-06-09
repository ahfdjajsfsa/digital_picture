% task2: image restoration with Wiener deconvolution
clear; clc; close all;

base = 'd:\Github\School\digital_picture';
src = fullfile(base, '原图与参考图');
out = fullfile(base, '处理后图像');
temp = fullfile(src, 'task2_wiener_search');
if ~exist(out, 'dir')
    mkdir(out);
end
if ~exist(temp, 'dir')
    mkdir(temp);
end

img = im2double(imread(fullfile(src, 'blurred wood.bmp')));
if size(img, 3) == 1
    img = repmat(img, [1 1 3]);
end

% The original clean image is not supplied, so the script uses blind metrics
% and visual ringing checks instead of MSE/SNR/SSIM against a reference.
fprintf('Input blind metrics:\n');
input_metrics = blind_metrics(img);
print_metrics(input_metrics);

% Coarse parameter search on a downsampled image. The final values below were
% selected from this range after rejecting over-sharpened/ringing candidates.
scale = 0.25;
small = imresize(img, scale);
small_gray = rgb2gray_local(small);
base_metrics = blind_metrics(repmat(small_gray, [1 1 3]));

lens = 13:2:25;
thetas = 30:5:50;
nsrs = [0.02 0.03 0.04 0.06 0.08];
rows = zeros(numel(lens) * numel(thetas) * numel(nsrs), 9);
k = 1;
for li = 1:numel(lens)
    for ti = 1:numel(thetas)
        for ni = 1:numel(nsrs)
            L = lens(li);
            theta = thetas(ti);
            nsr = nsrs(ni);
            psf = motion_psf(max(3, L * scale), theta, 0.42);
            raw = wiener_channel(small_gray, psf, nsr);
            restored = robust_rescale(raw);
            restored = (1 - 0.45) * restored + 0.45 * imgaussfilt(restored, 0.55);
            m = blind_metrics(repmat(restored, [1 1 3]));
            score = quality_score(m, base_metrics);
            rows(k, :) = [score, L, theta, nsr, ...
                          m.tenengrad / base_metrics.tenengrad, ...
                          m.lap_var / base_metrics.lap_var, ...
                          m.hf_std / base_metrics.hf_std, ...
                          m.entropy, m.clip_ratio];
            k = k + 1;
        end
    end
end

rows = sortrows(rows, -1);
writematrix(rows, fullfile(temp, 'task2_matlab_search_scores.csv'));
fprintf('\nTop 8 coarse search rows:\n');
fprintf('score      L   theta   NSR     TenRatio  LapRatio  HFRatio  Entropy  Clip\n');
for i = 1:min(8, size(rows, 1))
    fprintf('%8.4f  %2.0f  %6.0f  %.3f   %7.3f   %7.3f  %7.3f  %7.3f  %.4f\n', rows(i, :));
end

% Final parameters.
% The timestamp in the lower-right corner is a red camera overlay, not part
% of the blurred scene. Full RGB deconvolution damages it, so the final
% version restores only the luminance layer and preserves saturated red text.
final_len = 17;
final_theta = 40;
final_nsr = 0.06;
pad_size = 96;
post_smooth_weight = 0.00;
post_smooth_sigma = 0.00;
unsharp_amount = 1.35;
unsharp_sigma = 2.20;
luminance_blend = 0.55;
local_contrast_amount = 0.20;
sharpen_threshold = 0.016;

input_y = rgb2gray_local(img);
ycbcr = rgb2ycbcr(img);
denoised_y = edge_preserving_smooth(input_y);

% The direct Wiener result is only used as a conservative detail source. A
% full-strength deconvolution creates objectionable grains and ringing here.
psf = fspecial('motion', final_len, final_theta);
tapered_y = edgetaper(denoised_y, psf);
raw_y = deconvwnr(tapered_y, psf, final_nsr);
wiener_y = robust_rescale(raw_y);
if exist('imnlmfilt', 'file')
    wiener_y = imnlmfilt(wiener_y, 'DegreeOfSmoothing', 0.005);
end

structure_mask = adaptive_structure_mask(input_y);
sky_guard = input_y > 0.72 & imgaussfilt(structure_mask, 5.0) < 0.22;
blend_map = luminance_blend * (0.35 + 0.65 * structure_mask);
blend_map(sky_guard) = min(blend_map(sky_guard), 0.10);

mixed_y = (1 - blend_map) .* denoised_y + blend_map .* wiener_y;
final_y = imsharpen(mixed_y, 'Radius', unsharp_sigma, ...
                    'Amount', unsharp_amount, ...
                    'Threshold', sharpen_threshold);

contrast_y = adapthisteq(final_y, 'NumTiles', [8 8], 'ClipLimit', 0.004);
contrast_map = local_contrast_amount * structure_mask;
contrast_map(sky_guard) = 0;
final_y = clip01((1 - contrast_map) .* final_y + contrast_map .* contrast_y);

ycbcr(:, :, 1) = final_y;
final_img = clip01(ycbcr2rgb(ycbcr));
final_img = preserve_red_overlay(img, final_img);

wiener_ycbcr = ycbcr;
wiener_ycbcr(:, :, 1) = clip01(wiener_y);
wiener_only = preserve_red_overlay(img, clip01(ycbcr2rgb(wiener_ycbcr)));

fprintf('\nFinal parameters:\n');
fprintf('Motion PSF length = %.0f\n', final_len);
fprintf('Motion PSF theta  = %.0f degrees\n', final_theta);
fprintf('NSR               = %.4f\n', final_nsr);
fprintf('Pad size          = %d pixels\n', pad_size);
fprintf('Luminance blend   = %.2f\n', luminance_blend);
fprintf('Post smoothing    = edge-preserving prefilter\n');
fprintf('Unsharp amount    = %.2f, radius %.2f, threshold %.3f\n', unsharp_amount, unsharp_sigma, sharpen_threshold);
fprintf('Local contrast    = %.2f\n', local_contrast_amount);

fprintf('\nFinal blind metrics:\n');
final_metrics = blind_metrics(final_img);
print_metrics(final_metrics);

imwrite(img, fullfile(out, 'task2_input_blurred.bmp'));
imwrite(wiener_only, fullfile(out, 'task2_wiener_only.bmp'));
imwrite(final_img, fullfile(out, 'task2_final_result.bmp'));
imwrite([img, final_img], fullfile(out, 'task2_comparison.png'));

metrics_path = fullfile(out, 'task2_metrics.txt');
fid = fopen(metrics_path, 'w');
fprintf(fid, 'Task 2 image restoration metrics\n');
fprintf(fid, 'Input image: %s\n', fullfile(src, 'blurred wood.bmp'));
fprintf(fid, 'Final parameters: length=%g, theta=%g, nsr=%g, pad=%d, luminance_blend=%g, unsharp_amount=%g, unsharp_radius=%g, sharpen_threshold=%g, local_contrast_amount=%g\n', ...
        final_len, final_theta, final_nsr, pad_size, luminance_blend, unsharp_amount, unsharp_sigma, sharpen_threshold, local_contrast_amount);
fprintf(fid, 'Adaptive fusion: Wiener deconvolution is used as a conservative detail source; edge-preserving smoothing, controlled sharpening, and sky protection suppress grains and ringing.\n');
fprintf(fid, 'Note: saturated red camera overlays are preserved from the input to avoid timestamp ringing.\n\n');
fprintf(fid, 'Metric                 Input        Final       Ratio\n');
write_metric(fid, 'Tenengrad', input_metrics.tenengrad, final_metrics.tenengrad);
write_metric(fid, 'Laplacian variance', input_metrics.lap_var, final_metrics.lap_var);
write_metric(fid, 'Entropy', input_metrics.entropy, final_metrics.entropy);
write_metric(fid, 'High-frequency std', input_metrics.hf_std, final_metrics.hf_std);
fprintf(fid, 'Clip ratio             %10.6f  %10.6f\n', input_metrics.clip_ratio, final_metrics.clip_ratio);
fclose(fid);

fprintf('\nOutputs:\n');
fprintf('%s\n', fullfile(out, 'task2_final_result.bmp'));
fprintf('%s\n', fullfile(out, 'task2_comparison.png'));
fprintf('%s\n', metrics_path);
fprintf('\nDone.\n');

function y = wiener_gray(x, psf, nsr, pad_size)
padded = padarray(x, [pad_size pad_size], 'symmetric', 'both');
restored = wiener_channel(padded, psf, nsr);
y = restored(pad_size+1:end-pad_size, pad_size+1:end-pad_size);
end

function y = wiener_channel(x, psf, nsr)
[h, w] = size(x);
H = psf2otf_local(psf, [h w]);
G = fft2(x);
Y = conj(H) ./ (abs(H).^2 + nsr) .* G;
y = real(ifft2(Y));
end

function H = psf2otf_local(psf, out_size)
pad = zeros(out_size);
[ph, pw] = size(psf);
pad(1:ph, 1:pw) = psf;
pad = circshift(pad, [-floor(ph / 2), -floor(pw / 2)]);
H = fft2(pad);
end

function psf = motion_psf(len, theta_deg, sigma)
sz = ceil(len) * 2 + 1;
if mod(sz, 2) == 0
    sz = sz + 1;
end
c = (sz + 1) / 2;
[x, y] = meshgrid(1:sz, 1:sz);
x = x - c;
y = y - c;
theta = theta_deg * pi / 180;
along = x * cos(theta) + y * sin(theta);
perp = -x * sin(theta) + y * cos(theta);
psf = exp(-(perp .^ 2) / (2 * sigma ^ 2)) .* double(abs(along) <= len / 2);
if sum(psf(:)) <= 0
    psf(round(c), round(c)) = 1;
end
psf = psf / sum(psf(:));
end

function y = robust_rescale(x)
lo = quantile(x(:), 0.005);
hi = quantile(x(:), 0.995);
if hi <= lo
    y = clip01(x);
else
    y = (x - lo) / (hi - lo);
    y = clip01(y);
end
end

function y = clip01(x)
y = min(max(x, 0), 1);
end

function y = edge_preserving_smooth(x)
if exist('imguidedfilter', 'file')
    y = imguidedfilter(x, 'NeighborhoodSize', [7 7], 'DegreeOfSmoothing', 0.0008);
elseif exist('imbilatfilt', 'file')
    y = imbilatfilt(x, 0.018, 3);
else
    y = imgaussfilt(x, 0.45);
end
end

function mask = adaptive_structure_mask(gray)
sx = [1 0 -1; 2 0 -2; 1 0 -1] / 4;
sy = sx';
gx = imfilter(gray, sx, 'replicate');
gy = imfilter(gray, sy, 'replicate');
grad = sqrt(gx .^ 2 + gy .^ 2);
grad = imgaussfilt(grad, 1.2);
hi = quantile(grad(:), 0.98);
if hi <= 0
    mask = zeros(size(gray));
else
    mask = clip01(grad / hi);
end
mask = imgaussfilt(mask, 2.0);
mask = clip01(mask .^ 0.7);
end

function result = preserve_red_overlay(original, result)
r = original(:, :, 1);
g = original(:, :, 2);
b = original(:, :, 3);
mask = r > 0.45 & r > 1.35 * g & r > 1.35 * b;
mask = imdilate(mask, strel('disk', 2));
mask = imgaussfilt(double(mask), 0.8);
for c = 1:3
    result(:, :, c) = result(:, :, c) .* (1 - mask) + original(:, :, c) .* mask;
end
result = clip01(result);
end

function g = rgb2gray_local(x)
if size(x, 3) == 1
    g = x;
else
    g = 0.2989 * x(:, :, 1) + 0.5870 * x(:, :, 2) + 0.1140 * x(:, :, 3);
end
end

function m = blind_metrics(x)
gray = rgb2gray_local(clip01(x));
sx = [1 0 -1; 2 0 -2; 1 0 -1] / 4;
sy = sx';
gx = imfilter(gray, sx, 'replicate');
gy = imfilter(gray, sy, 'replicate');
lap = imfilter(gray, [0 1 0; 1 -4 1; 0 1 0], 'replicate');
hf = gray - imgaussfilt(gray, 1.0);
counts = histcounts(gray(:), 256, 'BinLimits', [0 1]);
p = counts / sum(counts);
p = p(p > 0);

m.tenengrad = mean(gx(:).^2 + gy(:).^2);
m.lap_var = var(lap(:));
m.entropy = -sum(p .* log2(p));
m.hf_std = std(hf(:));
m.clip_ratio = mean(x(:) < 0 | x(:) > 1);
end

function score = quality_score(m, base)
score = 0.35 * log1p(m.tenengrad / base.tenengrad) + ...
        0.20 * log1p(m.lap_var / base.lap_var) + ...
        0.20 * (m.entropy / base.entropy) - ...
        0.55 * max(0, m.hf_std / base.hf_std - 2.0) - ...
        6.00 * m.clip_ratio;
end

function print_metrics(m)
fprintf('Tenengrad          = %.6f\n', m.tenengrad);
fprintf('Laplacian variance = %.6f\n', m.lap_var);
fprintf('Entropy            = %.6f\n', m.entropy);
fprintf('High-freq std      = %.6f\n', m.hf_std);
fprintf('Clip ratio         = %.6f\n', m.clip_ratio);
end

function write_metric(fid, name, before, after)
fprintf(fid, '%-22s %10.6f  %10.6f  %10.4f\n', name, before, after, after / before);
end

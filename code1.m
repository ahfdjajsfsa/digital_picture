% task1: notch filter + gaussian filter
clear; clc; close all;

base = 'd:\Github\School\digital_picture';
src = fullfile(base, '原图与参考图');
out = fullfile(base, '处理后图像');
if ~exist(out, 'dir')
    mkdir(out);
end

ori = im2double(imread(fullfile(src, 'dogOriginal.bmp')));
img = im2double(imread(fullfile(src, 'dogDistorted.bmp')));

fprintf('Before:\n');
show_score(img, ori);

% 1. correct average brightness
img1 = img - (mean(img(:)) - mean(ori(:)));

% 2. remove periodic noise in frequency domain
fu = 71;
fv = 74;
points = [fu 0; 2*fu 0; 0 fv; 0 2*fv; fu fv; fu -fv; ...
          2*fu fv; 2*fu -fv; fu 2*fv; fu -2*fv; 2*fu 2*fv; 2*fu -2*fv];

img2 = notch_filter(img1, points, 5, 1);
imwrite(img2, fullfile(out, 'task1_step1_notch.bmp'));

% 3. smooth random noise
img3 = imgaussfilt(img2, 1.3);

fprintf('\nAfter:\n');
show_score(img3, ori);

imwrite(img3, fullfile(out, 'task1_final_result.bmp'));
imwrite(img3, fullfile(out, 'task1_step2_gaussian.bmp'));
imwrite(img, fullfile(out, 'task1_distorted_orig.bmp'));
imwrite(ori, fullfile(out, 'task1_original_ref.bmp'));
imwrite((img - ori) * 5 + 0.5, fullfile(out, 'task1_residual_before.bmp'));
imwrite((img3 - ori) * 5 + 0.5, fullfile(out, 'task1_residual_after.bmp'));

figure('Visible', 'off');
subplot(1,3,1); imshow(ori); title('original');
subplot(1,3,2); imshow(img); title('distorted');
subplot(1,3,3); imshow(img3); title('result');
saveas(gcf, fullfile(out, 'task1_comparison.png'));

fprintf('\nDone.\n');

function y = notch_filter(x, p, D0, n)
[h, w] = size(x);
cx = floor(w/2);
cy = floor(h/2);
[u, v] = meshgrid((0:w-1)-cx, (0:h-1)-cy);

H = ones(h, w);
for i = 1:size(p, 1)
    a = p(i, 1);
    b = p(i, 2);
    d1 = sqrt((u-a).^2 + (v-b).^2);
    d2 = sqrt((u+a).^2 + (v+b).^2);
    d1 = max(d1, 1e-10);
    d2 = max(d2, 1e-10);
    H = H .* (1 ./ (1 + (D0^2 ./ (d1 .* d2)).^n));
end

y = real(ifft2(ifftshift(fftshift(fft2(x)) .* H)));
end

function show_score(a, b)
e = a - b;
mse = mean(e(:).^2);
psnr_v = 10 * log10(1 / mse);
snr_v = 10 * log10(var(b(:)) / var(e(:)));
ssim_v = ssim(a, b);

fprintf('MSE  = %.2f\n', mse * 255^2);
fprintf('PSNR = %.2f dB\n', psnr_v);
fprintf('SNR  = %.2f dB\n', snr_v);
fprintf('SSIM = %.4f\n', ssim_v);
end

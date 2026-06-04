"""
任务一参数优化：网格搜索最佳陷波+高斯滤波参数
====================================================
"""
import numpy as np
from PIL import Image
import os
import itertools

BASE = r"d:\Github\School\digital_picture"
TEMP = os.path.join(BASE, "临时文件")
SRC = os.path.join(BASE, "原图与参考图")
OUT = os.path.join(BASE, "处理后图像")


def butterworth_notch_filter(img, notch_centers, D0=3, n=2):
    """巴特沃斯陷波滤波器"""
    H, W = img.shape
    center_h, center_w = H // 2, W // 2
    fft = np.fft.fft2(img)
    fft_shifted = np.fft.fftshift(fft)
    v, u = np.indices((H, W))
    u_rel = u - center_w
    v_rel = v - center_h
    H_filter = np.ones((H, W), dtype=np.float64)
    for uc, vc in notch_centers:
        D_pos = np.maximum(np.sqrt((u_rel - uc)**2 + (v_rel - vc)**2), 1e-10)
        D_neg = np.maximum(np.sqrt((u_rel + uc)**2 + (v_rel + vc)**2), 1e-10)
        H_notch = 1.0 / (1.0 + (D0**2 / (D_pos * D_neg))**n)
        H_filter *= H_notch
    fft_filtered = fft_shifted * H_filter
    return np.real(np.fft.ifft2(np.fft.ifftshift(fft_filtered)))


def compute_ssim(img1, img2, L=255.0):
    """SSIM 结构相似度"""
    from scipy.ndimage import uniform_filter
    K1, K2 = 0.01, 0.03
    C1, C2 = (K1*L)**2, (K2*L)**2
    ws = 11
    mu1 = uniform_filter(img1, size=ws)
    mu2 = uniform_filter(img2, size=ws)
    mu1_sq, mu2_sq = mu1**2, mu2**2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = uniform_filter(img1**2, size=ws) - mu1_sq
    sigma2_sq = uniform_filter(img2**2, size=ws) - mu2_sq
    sigma12 = uniform_filter(img1*img2, size=ws) - mu1_mu2
    sigma1_sq = np.maximum(sigma1_sq, 0)
    sigma2_sq = np.maximum(sigma2_sq, 0)
    num = (2*mu1_mu2 + C1) * (2*sigma12 + C2)
    den = (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)
    return np.mean(num / (den + 1e-10))


def evaluate(processed, original):
    """综合评估"""
    diff = processed - original
    mse = np.mean(diff**2)
    psnr = 10 * np.log10(255**2 / mse) if mse > 0 else float('inf')
    var_orig = np.var(original)
    var_noise = np.var(diff)
    snr = 10 * np.log10(var_orig / var_noise) if var_noise > 0 else float('inf')
    ssim = compute_ssim(processed, original)
    return {'MSE': mse, 'PSNR': psnr, 'SNR': snr, 'SSIM': ssim}


def main():
    print("=" * 60)
    print("任务一：参数网格搜索优化")
    print("=" * 60)

    # 加载图像
    distorted = np.array(Image.open(os.path.join(SRC, "dogDistorted.bmp")).convert('L'), dtype=np.float64)
    original = np.array(Image.open(os.path.join(SRC, "dogOriginal.bmp")).convert('L'), dtype=np.float64)

    # DC偏移校正
    dc_offset = distorted.mean() - original.mean()
    img_dc_corrected = distorted - dc_offset
    print(f"DC偏移校正: {dc_offset:.2f}")

    # 原始评估
    baseline = evaluate(distorted, original)
    print(f"原始失真: MSE={baseline['MSE']:.2f}, PSNR={baseline['PSNR']:.2f} dB, "
          f"SNR={baseline['SNR']:.2f} dB, SSIM={baseline['SSIM']:.4f}\n")

    # 周期噪声频率（基于之前分析的精确发现）
    f_u, f_v = 71, 74  # 基频
    notch_peaks = [
        (f_u, 0), (2*f_u, 0),
        (0, f_v), (0, 2*f_v),
        (f_u, f_v), (f_u, -f_v),
        (2*f_u, f_v), (2*f_u, -f_v),
        (f_u, 2*f_v), (f_u, -2*f_v),
        (2*f_u, 2*f_v), (2*f_u, -2*f_v),
    ]

    # === 阶段1: 陷波滤波器参数网格搜索 ===
    print("-" * 60)
    print("阶段1: 陷波滤波器参数搜索")
    print("-" * 60)
    D0_values = [2, 3, 4, 5, 6]
    n_values = [1, 2, 3, 4]
    notch_results = []

    for D0, n in itertools.product(D0_values, n_values):
        filtered = butterworth_notch_filter(img_dc_corrected, notch_peaks, D0=D0, n=n)
        m = evaluate(filtered, original)
        notch_results.append({'D0': D0, 'n': n, 'metrics': m, 'image': filtered})
        print(f"  D0={D0}, n={n} -> MSE={m['MSE']:.2f}, PSNR={m['PSNR']:.2f} dB, SSIM={m['SSIM']:.4f}")

    # 找最佳陷波参数
    best_notch = min(notch_results, key=lambda x: x['metrics']['MSE'])
    print(f"\n  最佳陷波: D0={best_notch['D0']}, n={best_notch['n']} "
          f"(PSNR={best_notch['metrics']['PSNR']:.2f} dB)")

    # === 阶段2: 空域滤波器参数搜索 ===
    print("\n" + "-" * 60)
    print("阶段2: 空域滤波器参数搜索（基于最佳陷波结果）")
    print("-" * 60)

    img_after_notch = best_notch['image']

    # 2a: 高斯滤波参数搜索
    from scipy.ndimage import gaussian_filter
    sigma_values = [0.5, 0.8, 1.0, 1.2, 1.5, 1.8, 2.0, 2.5, 3.0]
    gauss_results = []
    for sigma in sigma_values:
        filtered = gaussian_filter(img_after_notch, sigma=sigma)
        m = evaluate(filtered, original)
        gauss_results.append({'sigma': sigma, 'metrics': m, 'image': filtered})
        print(f"  Gaussian sigma={sigma:<4} -> MSE={m['MSE']:8.2f}, PSNR={m['PSNR']:.2f} dB, "
              f"SNR={m['SNR']:.2f} dB, SSIM={m['SSIM']:.4f}")

    best_gauss = min(gauss_results, key=lambda x: x['metrics']['MSE'])
    print(f"\n  最佳高斯: sigma={best_gauss['sigma']} "
          f"(PSNR={best_gauss['metrics']['PSNR']:.2f} dB, SSIM={best_gauss['metrics']['SSIM']:.4f})")

    # 2b: 算术均值滤波参数搜索
    from scipy.ndimage import uniform_filter
    ks_values = [3, 5]
    mean_results = []
    for ks in ks_values:
        filtered = uniform_filter(img_after_notch, size=ks)
        m = evaluate(filtered, original)
        mean_results.append({'ks': ks, 'metrics': m, 'image': filtered})
        print(f"  Mean ks={ks}      -> MSE={m['MSE']:8.2f}, PSNR={m['PSNR']:.2f} dB, "
              f"SNR={m['SNR']:.2f} dB, SSIM={m['SSIM']:.4f}")

    # === 最终对比 ===
    print("\n" + "=" * 70)
    print("最终结果对比")
    print("=" * 70)
    print(f"{'方案':<30} {'MSE':>10} {'PSNR(dB)':>10} {'SNR(dB)':>9} {'SSIM':>8}")
    print("-" * 70)
    print(f"{'原始失真':<30} {baseline['MSE']:>10.2f} {baseline['PSNR']:>10.2f} {baseline['SNR']:>9.2f} {baseline['SSIM']:>8.4f}")
    print(f"{'陷波(D0=' + str(best_notch['D0']) + ',n=' + str(best_notch['n']) + ')':<30} "
          f"{best_notch['metrics']['MSE']:>10.2f} {best_notch['metrics']['PSNR']:>10.2f} "
          f"{best_notch['metrics']['SNR']:>9.2f} {best_notch['metrics']['SSIM']:>8.4f}")
    print(f"{'陷波+高斯(sigma=' + str(best_gauss['sigma']) + ')':<30} "
          f"{best_gauss['metrics']['MSE']:>10.2f} {best_gauss['metrics']['PSNR']:>10.2f} "
          f"{best_gauss['metrics']['SNR']:>9.2f} {best_gauss['metrics']['SSIM']:>8.4f}")
    for r in mean_results:
        print(f"{'陷波+均值(ks=' + str(r['ks']) + ')':<30} "
              f"{r['metrics']['MSE']:>10.2f} {r['metrics']['PSNR']:>10.2f} "
              f"{r['metrics']['SNR']:>9.2f} {r['metrics']['SSIM']:>8.4f}")

    # 确定全局最佳
    all_results = [{'name': 'notch', **best_notch}]
    all_results.append({'name': 'notch+gauss', 'params': f'sigma={best_gauss["sigma"]}',
                        'metrics': best_gauss['metrics'], 'image': best_gauss['image']})
    for r in mean_results:
        all_results.append({'name': f'notch+mean(ks={r["ks"]})', 'metrics': r['metrics'], 'image': r['image']})

    global_best = min(all_results, key=lambda x: x['metrics']['MSE'])
    print(f"\n*** 全局最佳方案: {global_best['name']} ***")
    print(f"    PSNR = {global_best['metrics']['PSNR']:.2f} dB")
    print(f"    SNR  = {global_best['metrics']['SNR']:.2f} dB")
    print(f"    SSIM = {global_best['metrics']['SSIM']:.4f}")
    print(f"    MSE  = {global_best['metrics']['MSE']:.2f}")

    # 保存最佳结果
    best_img = np.clip(global_best['image'], 0, 255).astype(np.uint8)
    Image.fromarray(best_img, mode='L').save(os.path.join(OUT, "task1_best_result.bmp"))
    print(f"\n最佳结果已保存至: {os.path.join(OUT, 'task1_best_result.bmp')}")

    # 保存对比图
    for label, arr in [("distorted", distorted), ("original", original)]:
        img = np.clip(arr, 0, 255).astype(np.uint8)
        Image.fromarray(img, mode='L').save(os.path.join(OUT, f"task1_{label}.bmp"))
    print("原始/失真对照已保存。")
    print("Done.")


if __name__ == '__main__':
    main()

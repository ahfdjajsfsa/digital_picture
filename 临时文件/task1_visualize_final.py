"""
任务一最终可视化：原图 / 失真图 / 去噪结果 / FFT频谱 / 残差对比
"""
import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter, uniform_filter
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os

BASE = r"d:\Github\School\digital_picture"
TEMP = os.path.join(BASE, "临时文件")
SRC = os.path.join(BASE, "原图与参考图")
OUT = os.path.join(BASE, "处理后图像")


def butterworth_notch_filter(img, notch_centers, D0=3, n=2):
    H, W = img.shape
    ch, cw = H // 2, W // 2
    fft = np.fft.fft2(img)
    fft_s = np.fft.fftshift(fft)
    v, u = np.indices((H, W))
    u_rel, v_rel = u - cw, v - ch
    H_f = np.ones((H, W), dtype=np.float64)
    for uc, vc in notch_centers:
        Dp = np.maximum(np.sqrt((u_rel-uc)**2 + (v_rel-vc)**2), 1e-10)
        Dn = np.maximum(np.sqrt((u_rel+uc)**2 + (v_rel+vc)**2), 1e-10)
        H_f *= 1.0 / (1.0 + (D0**2 / (Dp * Dn))**n)
    return np.real(np.fft.ifft2(np.fft.ifftshift(fft_s * H_f)))


def compute_ssim(img1, img2, L=255.0):
    K1, K2 = 0.01, 0.03
    C1, C2 = (K1*L)**2, (K2*L)**2
    ws = 11
    mu1 = uniform_filter(img1, size=ws)
    mu2 = uniform_filter(img2, size=ws)
    s1_sq = np.maximum(uniform_filter(img1**2, size=ws)-mu1**2, 0)
    s2_sq = np.maximum(uniform_filter(img2**2, size=ws)-mu2**2, 0)
    s12 = uniform_filter(img1*img2, size=ws)-mu1*mu2
    num = (2*mu1*mu2+C1)*(2*s12+C2)
    den = (mu1**2+mu2**2+C1)*(s1_sq+s2_sq+C2)
    return np.mean(num/(den+1e-10))


def fft_log_mag(img):
    fft = np.fft.fftshift(np.fft.fft2(img))
    return np.log1p(np.abs(fft))


def main():
    print("生成任务一最终可视化...")

    # 加载图像
    original = np.array(Image.open(os.path.join(SRC, "dogOriginal.bmp")).convert('L'), dtype=np.float64)
    distorted = np.array(Image.open(os.path.join(SRC, "dogDistorted.bmp")).convert('L'), dtype=np.float64)

    # 最佳参数处理
    f_u, f_v = 71, 74
    notch_peaks = [
        (f_u,0), (2*f_u,0), (0,f_v), (0,2*f_v),
        (f_u,f_v), (f_u,-f_v), (2*f_u,f_v), (2*f_u,-f_v),
        (f_u,2*f_v), (f_u,-2*f_v), (2*f_u,2*f_v), (2*f_u,-2*f_v),
    ]
    BEST_D0, BEST_N = 5, 1
    BEST_SIGMA = 1.3

    dc_offset = distorted.mean() - original.mean()
    img_dc = distorted - dc_offset
    img_notch = butterworth_notch_filter(img_dc, notch_peaks, D0=BEST_D0, n=BEST_N)
    img_final = gaussian_filter(img_notch, sigma=BEST_SIGMA)

    # 计算指标
    def eval_metrics(proc, orig):
        diff = proc - orig
        mse = np.mean(diff**2)
        psnr = 10*np.log10(255**2/mse) if mse>0 else float('inf')
        snr = 10*np.log10(np.var(orig)/np.var(diff)) if np.var(diff)>0 else float('inf')
        ssim = compute_ssim(proc, orig)
        return mse, psnr, snr, ssim

    mse_dist, psnr_dist, snr_dist, ssim_dist = eval_metrics(distorted, original)
    mse_final, psnr_final, snr_final, ssim_final = eval_metrics(img_final, original)

    # 残差
    noise_dist = distorted - original
    noise_final = img_final - original

    H, W = original.shape
    ch, cw = H//2, W//2

    # ==========================================
    # 大图: 3x3 布局
    # ==========================================
    fig, axes = plt.subplots(3, 3, figsize=(16, 14))
    fig.suptitle('Task 1: Spatial + Frequency Domain Image Enhancement\n'
                 f'Butterworth Notch(D0={BEST_D0}, n={BEST_N}) + Gaussian(sigma={BEST_SIGMA})',
                 fontsize=14, fontweight='bold')

    # Row 1: 空域图像
    axes[0,0].imshow(original, cmap='gray', vmin=0, vmax=255)
    axes[0,0].set_title('Original Image (Reference)', fontweight='bold')
    axes[0,0].axis('off')

    axes[0,1].imshow(distorted, cmap='gray', vmin=0, vmax=255)
    axes[0,1].set_title(f'Distorted (Noisy)\nPSNR={psnr_dist:.1f}dB, SSIM={ssim_dist:.3f}', fontweight='bold')
    axes[0,1].axis('off')

    axes[0,2].imshow(img_final, cmap='gray', vmin=0, vmax=255)
    axes[0,2].set_title(f'Denoised Result\nPSNR={psnr_final:.1f}dB, SSIM={ssim_final:.3f}', fontweight='bold', color='green')
    axes[0,2].axis('off')

    # Row 2: FFT频谱 (对数幅度)
    vmax_fft = np.percentile(fft_log_mag(distorted), 98)
    axes[1,0].imshow(fft_log_mag(original), cmap='hot', aspect='equal')
    axes[1,0].set_title('FFT Spectrum: Original', fontweight='bold')
    axes[1,0].axis('off')

    axes[1,1].imshow(fft_log_mag(distorted), cmap='hot', aspect='equal', vmax=vmax_fft)
    axes[1,1].set_title('FFT Spectrum: Distorted\n(periodic noise peaks visible)', fontweight='bold')
    # 标注峰值位置
    for uc, vc in notch_peaks:
        axes[1,1].plot(cw+uc, ch+vc, 'o', markersize=4, markerfacecolor='none',
                       markeredgecolor='cyan', markeredgewidth=1, alpha=0.7)
    axes[1,1].axis('off')

    axes[1,2].imshow(fft_log_mag(img_final), cmap='hot', aspect='equal', vmax=vmax_fft)
    axes[1,2].set_title('FFT Spectrum: Denoised\n(periodic peaks removed)', fontweight='bold', color='green')
    axes[1,2].axis('off')

    # Row 3: 残差分析
    vmax_resid = max(abs(noise_dist).max(), abs(noise_final).max()) * 0.7
    im0 = axes[2,0].imshow(noise_dist, cmap='RdBu_r', vmin=-vmax_resid, vmax=vmax_resid)
    axes[2,0].set_title(f'Noise (Dist-Orig)\nstd={noise_dist.std():.1f}', fontweight='bold')
    axes[2,0].axis('off')
    plt.colorbar(im0, ax=axes[2,0], fraction=0.046, pad=0.04)

    im1 = axes[2,1].imshow(noise_final, cmap='RdBu_r', vmin=-vmax_resid, vmax=vmax_resid)
    axes[2,1].set_title(f'Residual (Denoised-Orig)\nstd={noise_final.std():.1f}', fontweight='bold', color='green')
    axes[2,1].axis('off')
    plt.colorbar(im1, ax=axes[2,1], fraction=0.046, pad=0.04)

    # 残差直方图对比
    axes[2,2].hist(noise_dist.flatten(), bins=80, alpha=0.5, color='red', label=f'Before(std={noise_dist.std():.1f})')
    axes[2,2].hist(noise_final.flatten(), bins=80, alpha=0.5, color='green', label=f'After(std={noise_final.std():.1f})')
    axes[2,2].axvline(x=0, color='black', linestyle='--', linewidth=1)
    axes[2,2].set_xlabel('Noise Intensity')
    axes[2,2].set_ylabel('Pixel Count')
    axes[2,2].set_title('Residual Noise Distribution', fontweight='bold')
    axes[2,2].legend()

    plt.tight_layout()
    plt.savefig(os.path.join(TEMP, "task1_final_comparison.png"), dpi=150, bbox_inches='tight')
    plt.close()
    print("  [OK] task1_final_comparison.png")

    # ==========================================
    # 补充图: 去噪过程阶段展示
    # ==========================================
    fig2, axes2 = plt.subplots(2, 3, figsize=(16, 9))
    fig2.suptitle('Task 1: Denoising Pipeline Stages', fontsize=14, fontweight='bold')

    stages = [
        ('Original', original, None),
        ('Distorted (Noisy)', distorted, psnr_dist),
        ('After DC Correction', img_dc, None),
        ('After Notch Filter', img_notch, None),
        ('Final (Notch+Gaussian)', img_final, psnr_final),
    ]

    # 在 (1,0) 放中间行剖面
    row_idx = H // 2
    ax_profile = axes2[1, 1]
    ax_profile.plot(original[row_idx,:], 'b-', alpha=0.5, linewidth=0.8, label='Original')
    ax_profile.plot(distorted[row_idx,:], 'r-', alpha=0.4, linewidth=0.6, label='Distorted')
    ax_profile.plot(img_final[row_idx,:], 'g-', alpha=0.7, linewidth=0.8, label='Denoised')
    ax_profile.set_xlabel('Column (pixel)')
    ax_profile.set_ylabel('Intensity')
    ax_profile.set_title(f'Line Profile (Row {row_idx})', fontweight='bold')
    ax_profile.legend(fontsize=7)
    ax_profile.set_ylim(0, 280)

    # 指标摘要
    ax_summary = axes2[1, 2]
    ax_summary.axis('off')
    metrics_text = (
        f"=== QUALITY METRICS ===\n\n"
        f"BEFORE (Distorted):\n"
        f"  MSE  = {mse_dist:.1f}\n"
        f"  PSNR = {psnr_dist:.2f} dB\n"
        f"  SNR  = {snr_dist:.2f} dB\n"
        f"  SSIM = {ssim_dist:.4f}\n\n"
        f"AFTER (Denoised):\n"
        f"  MSE  = {mse_final:.1f}\n"
        f"  PSNR = {psnr_final:.2f} dB\n"
        f"  SNR  = {snr_final:.2f} dB\n"
        f"  SSIM = {ssim_final:.4f}\n\n"
        f"IMPROVEMENT:\n"
        f"  PSNR +{psnr_final-psnr_dist:.2f} dB\n"
        f"  MSE reduced {100*(1-mse_final/mse_dist):.1f}%\n"
        f"  SNR  +{snr_final-snr_dist:.2f} dB\n"
        f"  SSIM +{ssim_final-ssim_dist:.4f}"
    )
    ax_summary.text(0.05, 0.95, metrics_text, transform=ax_summary.transAxes,
                    fontsize=10, verticalalignment='top', fontfamily='monospace',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # 阶段图像
    stage_positions = [(0,0), (0,1), (0,2), (1,0)]
    for idx, (title, img, psnr_val) in enumerate(stages):
        if idx >= len(stage_positions):
            break
        r, c = stage_positions[idx]
        axes2[r, c].imshow(img, cmap='gray', vmin=0, vmax=255)
        label = title
        if psnr_val is not None:
            label += f'\nPSNR={psnr_val:.1f}dB'
        axes2[r, c].set_title(label, fontweight='bold')
        axes2[r, c].axis('off')

    plt.tight_layout()
    plt.savefig(os.path.join(TEMP, "task1_pipeline_stages.png"), dpi=150, bbox_inches='tight')
    plt.close()
    print("  [OK] task1_pipeline_stages.png")

    # ==========================================
    # 保存最终结果到处理后图像文件夹
    # ==========================================
    for label, arr in [
        ("original", original),
        ("distorted", distorted),
        ("denoised_final", img_final),
        ("residual_before", noise_dist + 128),  # 偏移以便可视化
        ("residual_after", noise_final + 128),
    ]:
        img_save = np.clip(arr, 0, 255).astype(np.uint8)
        Image.fromarray(img_save, mode='L').save(os.path.join(OUT, f"task1_{label}.bmp"))
    print(f"  [OK] 结果图像已保存至 {OUT}")

    print(f"\n最终评估指标:")
    print(f"  MSE  = {mse_final:.2f}  (原始: {mse_dist:.1f})")
    print(f"  PSNR = {psnr_final:.2f} dB  (原始: {psnr_dist:.2f} dB, 提升 +{psnr_final-psnr_dist:.2f} dB)")
    print(f"  SNR  = {snr_final:.2f} dB  (原始: {snr_dist:.2f} dB, 提升 +{snr_final-snr_dist:.2f} dB)")
    print(f"  SSIM = {ssim_final:.4f}  (原始: {ssim_dist:.4f}, 提升 +{ssim_final-ssim_dist:.4f})")
    print("Done.")


if __name__ == '__main__':
    main()

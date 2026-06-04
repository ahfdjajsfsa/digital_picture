"""
精细调优：在最佳参数附近搜索 + 不同Sigma对比MSE vs SSIM权衡
"""
import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter, uniform_filter
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
        Dp = np.maximum(np.sqrt((u_rel - uc)**2 + (v_rel - vc)**2), 1e-10)
        Dn = np.maximum(np.sqrt((u_rel + uc)**2 + (v_rel + vc)**2), 1e-10)
        H_f *= 1.0 / (1.0 + (D0**2 / (Dp * Dn))**n)
    return np.real(np.fft.ifft2(np.fft.ifftshift(fft_s * H_f)))


def compute_ssim(img1, img2, L=255.0):
    K1, K2 = 0.01, 0.03
    C1, C2 = (K1*L)**2, (K2*L)**2
    ws = 11
    mu1 = uniform_filter(img1, size=ws)
    mu2 = uniform_filter(img2, size=ws)
    s1_sq = np.maximum(uniform_filter(img1**2, size=ws) - mu1**2, 0)
    s2_sq = np.maximum(uniform_filter(img2**2, size=ws) - mu2**2, 0)
    s12 = uniform_filter(img1*img2, size=ws) - mu1*mu2
    num = (2*mu1*mu2 + C1) * (2*s12 + C2)
    den = (mu1**2 + mu2**2 + C1) * (s1_sq + s2_sq + C2)
    return np.mean(num / (den + 1e-10))


def evaluate(processed, original):
    diff = processed - original
    mse = np.mean(diff**2)
    psnr = 10 * np.log10(255**2 / mse) if mse > 0 else float('inf')
    snr = 10 * np.log10(np.var(original) / np.var(diff)) if np.var(diff) > 0 else float('inf')
    ssim = compute_ssim(processed, original)
    return {'MSE': mse, 'PSNR': psnr, 'SNR': snr, 'SSIM': ssim}


def main():
    print("=" * 60)
    print("精细调优：陷波+高斯滤波最佳参数")
    print("=" * 60)

    distorted = np.array(Image.open(os.path.join(SRC, "dogDistorted.bmp")).convert('L'), dtype=np.float64)
    original = np.array(Image.open(os.path.join(SRC, "dogOriginal.bmp")).convert('L'), dtype=np.float64)
    img_dc = distorted - (distorted.mean() - original.mean())

    # 周期噪声频率
    f_u, f_v = 71, 74
    notch_peaks = [
        (f_u, 0), (2*f_u, 0),
        (0, f_v), (0, 2*f_v),
        (f_u, f_v), (f_u, -f_v),
        (2*f_u, f_v), (2*f_u, -f_v),
        (f_u, 2*f_v), (f_u, -2*f_v),
        (2*f_u, 2*f_v), (2*f_u, -2*f_v),
    ]

    # ============================================================
    # 精细搜索: D0=5,6,7,8; n=1; Gaussian sigma=1.0~2.0
    # ============================================================
    print("\n精细网格搜索 (D0, sigma) 组合...\n")
    print(f"{'D0':>4} {'sigma':>7} {'MSE':>10} {'PSNR(dB)':>10} {'SNR(dB)':>9} {'SSIM':>8}")
    print("-" * 55)

    best_mse = float('inf')
    best_result = None
    results_table = []

    for D0 in [5, 6, 7, 8]:
        img_notch = butterworth_notch_filter(img_dc, notch_peaks, D0=D0, n=1)
        for sigma in [0.8, 1.0, 1.1, 1.2, 1.3, 1.5, 1.8]:
            img_final = gaussian_filter(img_notch, sigma=sigma)
            m = evaluate(img_final, original)
            results_table.append({'D0': D0, 'sigma': sigma, 'metrics': m, 'image': img_final})
            marker = ""
            if m['MSE'] < best_mse:
                best_mse = m['MSE']
                best_result = {'D0': D0, 'sigma': sigma, 'metrics': m, 'image': img_final}
                marker = " <-- BEST MSE"
            print(f"{D0:>4} {sigma:>7.1f} {m['MSE']:>10.2f} {m['PSNR']:>10.2f} {m['SNR']:>9.2f} {m['SSIM']:>8.4f}{marker}")

    print(f"\n最佳MSE组合: D0={best_result['D0']}, sigma={best_result['sigma']}")
    print(f"  PSNR={best_result['metrics']['PSNR']:.2f} dB, SSIM={best_result['metrics']['SSIM']:.4f}")

    # 找最佳SSIM
    best_ssim = max(results_table, key=lambda x: x['metrics']['SSIM'])
    print(f"\n最佳SSIM组合: D0={best_ssim['D0']}, sigma={best_ssim['sigma']}")
    print(f"  PSNR={best_ssim['metrics']['PSNR']:.2f} dB, SSIM={best_ssim['metrics']['SSIM']:.4f}")

    # ============================================================
    # 尝试：分频带不同D0（基频用小D0，谐波用大D0）
    # ============================================================
    print("\n" + "-" * 60)
    print("实验：分频带陷波（基频紧/谐波宽）")
    print("-" * 60)

    # 基频: D0=3, 谐波和交叉项: D0=6
    def multi_D0_notch(img, peaks_with_D0):
        H, W = img.shape
        ch, cw = H // 2, W // 2
        fft = np.fft.fft2(img)
        fft_s = np.fft.fftshift(fft)
        v, u = np.indices((H, W))
        u_rel, v_rel = u - cw, v - ch
        H_f = np.ones((H, W), dtype=np.float64)
        for uc, vc, D0 in peaks_with_D0:
            Dp = np.maximum(np.sqrt((u_rel-uc)**2 + (v_rel-vc)**2), 1e-10)
            Dn = np.maximum(np.sqrt((u_rel+uc)**2 + (v_rel+vc)**2), 1e-10)
            H_f *= 1.0 / (1.0 + (D0**2 / (Dp * Dn))**1)
        return np.real(np.fft.ifft2(np.fft.ifftshift(fft_s * H_f)))

    # 分组：基频用D0=3，谐波用D0=7
    peaks_multi = [
        (f_u, 0, 3), (2*f_u, 0, 7),          # 水平
        (0, f_v, 3), (0, 2*f_v, 7),           # 垂直
        (f_u, f_v, 5), (f_u, -f_v, 5),        # 交叉 (f1±f2)
        (2*f_u, f_v, 7), (2*f_u, -f_v, 7),
        (f_u, 2*f_v, 7), (f_u, -2*f_v, 7),
        (2*f_u, 2*f_v, 8), (2*f_u, -2*f_v, 8),
    ]

    img_multi_notch = multi_D0_notch(img_dc, peaks_multi)
    m_multi = evaluate(img_multi_notch, original)
    print(f"分频带陷波: MSE={m_multi['MSE']:.2f}, PSNR={m_multi['PSNR']:.2f} dB, SSIM={m_multi['SSIM']:.4f}")

    for sigma in [1.0, 1.2, 1.5]:
        img_multi_final = gaussian_filter(img_multi_notch, sigma=sigma)
        m_mf = evaluate(img_multi_final, original)
        print(f"分频带陷波+Gaussian(sigma={sigma}): PSNR={m_mf['PSNR']:.2f} dB, SSIM={m_mf['SSIM']:.4f}")

    # ============================================================
    # 最终总结
    # ============================================================
    print("\n" + "=" * 60)
    print("最终方案确定")
    print("=" * 60)

    # 比较所有方案
    all_best = min(results_table + [{'D0': 'multi', 'sigma': 1.2, 'metrics': m_mf, 'image': img_multi_final}],
                   key=lambda x: x['metrics']['MSE'])

    # 保存全局最佳
    best_img = np.clip(best_result['image'], 0, 255).astype(np.uint8)
    Image.fromarray(best_img, mode='L').save(os.path.join(OUT, "task1_final_result.bmp"))

    print(f"最终算法参数: 巴特沃斯陷波(D0={best_result['D0']}, n=1) + 高斯滤波(sigma={best_result['sigma']})")
    m = best_result['metrics']
    print(f"最终指标: MSE={m['MSE']:.2f}, PSNR={m['PSNR']:.2f} dB, SNR={m['SNR']:.2f} dB, SSIM={m['SSIM']:.4f}")
    print(f"相较于原始失真图: PSNR提升 +{m['PSNR']-11.48:.2f} dB, MSE降低 {(1-m['MSE']/4629.17)*100:.1f}%")
    print(f"\n最终结果已保存: {os.path.join(OUT, 'task1_final_result.bmp')}")
    print("Done.")


if __name__ == '__main__':
    main()

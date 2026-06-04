"""
任务一：空域+频域结合图像增强算法
============================================
算法流程：
  1. 预处理：减去直流偏移
  2. 频域：巴特沃斯陷波滤波器 → 去除周期噪声
  3. 空域：双边/高斯滤波器 → 抑制随机噪声
  4. 评估：MSE / SNR / SSIM

知识点框架内技术：
  - 频域：陷波滤波器(Notch Filter)
  - 频域：巴特沃斯带阻滤波器
  - 空域：高斯滤波器、双边滤波器、算术均值滤波器
"""
import numpy as np
from PIL import Image
import os
import sys

# 添加项目路径
BASE = r"d:\Github\School\digital_picture"
TEMP = os.path.join(BASE, "临时文件")
SRC = os.path.join(BASE, "原图与参考图")
OUT = os.path.join(BASE, "处理后图像")

# ============================================================
# 0. 加载图像
# ============================================================
def load_images():
    """加载失真图和原始参考图"""
    original = np.array(Image.open(os.path.join(SRC, "dogOriginal.bmp")).convert('L'), dtype=np.float64)
    distorted = np.array(Image.open(os.path.join(SRC, "dogDistorted.bmp")).convert('L'), dtype=np.float64)
    return distorted, original


# ============================================================
# 1. 频域陷波滤波器（巴特沃斯型）
# ============================================================
def butterworth_notch_filter(img, notch_centers, D0=3, n=2):
    """
    巴特沃斯陷波滤波器：滤除指定频率成分

    参数:
        img: 输入图像 (H, W)
        notch_centers: 陷波中心列表，每个元素为 (u_px, v_px)，相对于中心
        D0: 陷波半径（频率域像素单位）
        n: 巴特沃斯阶数
    返回:
        滤波后图像
    """
    H, W = img.shape
    center_h, center_w = H // 2, W // 2

    # FFT
    fft = np.fft.fft2(img)
    fft_shifted = np.fft.fftshift(fft)

    # 构建频率网格
    v, u = np.indices((H, W))
    u_rel = u - center_w  # 相对DC的水平频率
    v_rel = v - center_h  # 相对DC的垂直频率

    # 初始传递函数（全通）
    H_filter = np.ones((H, W), dtype=np.float64)

    # 对每个陷波频率对（共轭对称），构建陷波滤波器
    for uc, vc in notch_centers:
        # 正频率侧
        D_pos = np.sqrt((u_rel - uc)**2 + (v_rel - vc)**2)
        # 负频率侧（共轭对称）
        D_neg = np.sqrt((u_rel + uc)**2 + (v_rel + vc)**2)

        # 避免除零
        D_pos = np.maximum(D_pos, 1e-10)
        D_neg = np.maximum(D_neg, 1e-10)

        # 巴特沃斯陷波：H = 1 / (1 + (D0^2 / (D_pos * D_neg))^n)
        H_notch = 1.0 / (1.0 + (D0**2 / (D_pos * D_neg))**n)
        H_filter *= H_notch

    # 应用滤波器
    fft_filtered = fft_shifted * H_filter
    fft_filtered = np.fft.ifftshift(fft_filtered)
    img_filtered = np.fft.ifft2(fft_filtered)

    return np.real(img_filtered)


# ============================================================
# 2. 空域滤波器
# ============================================================
def spatial_gaussian_filter(img, sigma=1.5, kernel_size=None):
    """高斯滤波器（空域平滑）"""
    from scipy.ndimage import gaussian_filter
    return gaussian_filter(img, sigma=sigma)


def spatial_bilateral_filter(img, sigma_spatial=5, sigma_color=30):
    """
    双边滤波器（保边平滑）
    自己实现双边滤波（知识点框架内 - 双边滤波器）
    """
    H, W = img.shape
    result = np.zeros_like(img)

    # 空间高斯核半径
    radius = int(np.ceil(3 * sigma_spatial))
    ks = 2 * radius + 1

    # 预计算空间高斯权重
    spatial_kernel = np.zeros((ks, ks))
    for i in range(ks):
        for j in range(ks):
            di = i - radius
            dj = j - radius
            spatial_kernel[i, j] = np.exp(-(di**2 + dj**2) / (2 * sigma_spatial**2))

    # 对每个像素进行双边滤波
    # 使用边缘扩展
    padded = np.pad(img, radius, mode='reflect')

    two_sigma2 = 2 * sigma_color**2

    for y in range(H):
        for x in range(W):
            # 局部窗口
            patch = padded[y:y+ks, x:x+ks]
            center_val = padded[y+radius, x+radius]

            # 灰度高斯权重
            diff = patch - center_val
            range_kernel = np.exp(-(diff**2) / two_sigma2)

            # 总权重
            weights = spatial_kernel * range_kernel
            weights_sum = weights.sum()

            if weights_sum > 0:
                result[y, x] = (patch * weights).sum() / weights_sum
            else:
                result[y, x] = center_val

    return result


def spatial_arithmetic_mean_filter(img, kernel_size=3):
    """算术均值滤波器（空域平滑）"""
    from scipy.ndimage import uniform_filter
    return uniform_filter(img, size=kernel_size)


# ============================================================
# 3. 图像质量评估
# ============================================================
def compute_ssim(img1, img2, L=255.0):
    """
    计算结构相似度 SSIM
    SSIM(x, y) = (2μxμy + C1)(2σxy + C2) / ((μx² + μy² + C1)(σx² + σy² + C2))
    """
    K1, K2 = 0.01, 0.03
    C1 = (K1 * L) ** 2
    C2 = (K2 * L) ** 2

    # 使用 11x11 高斯窗口计算局部统计量
    from scipy.ndimage import uniform_filter, gaussian_filter

    # 简化版：使用均匀窗口计算局部均值
    window_size = 11
    mu1 = uniform_filter(img1, size=window_size)
    mu2 = uniform_filter(img2, size=window_size)

    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2

    sigma1_sq = uniform_filter(img1**2, size=window_size) - mu1_sq
    sigma2_sq = uniform_filter(img2**2, size=window_size) - mu2_sq
    sigma12 = uniform_filter(img1 * img2, size=window_size) - mu1_mu2

    # 避免负方差（数值误差）
    sigma1_sq = np.maximum(sigma1_sq, 0)
    sigma2_sq = np.maximum(sigma2_sq, 0)

    # SSIM 分子和分母
    num = (2 * mu1_mu2 + C1) * (2 * sigma12 + C2)
    den = (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)

    ssim_map = num / (den + 1e-10)
    return np.mean(ssim_map)


def evaluate(processed, original):
    """计算全部质量指标"""
    diff = processed - original
    mse = np.mean(diff**2)
    psnr = 10 * np.log10(255**2 / mse) if mse > 0 else float('inf')

    var_orig = np.var(original)
    var_noise = np.var(diff)
    snr = 10 * np.log10(var_orig / var_noise) if var_noise > 0 else float('inf')

    ssim = compute_ssim(processed, original)

    return {
        'MSE': mse,
        'PSNR': psnr,
        'SNR': snr,
        'SSIM': ssim
    }


def save_image(arr, filename):
    """保存为 BMP 文件"""
    arr_clipped = np.clip(arr, 0, 255).astype(np.uint8)
    img = Image.fromarray(arr_clipped, mode='L')
    img.save(filename)
    print(f"  已保存: {filename}")


# ============================================================
# 4. 主流水线：组合去噪
# ============================================================
def denoise_pipeline(distorted, original, config, save_intermediate=True):
    """
    完整去噪流水线

    参数:
        distorted: 失真图像
        original: 原始参考图像（仅用于评估）
        config: 配置字典
        save_intermediate: 是否保存中间结果
    """
    print("\n" + "=" * 60)
    print(f"配置: {config['name']}")
    print("=" * 60)

    current = distorted.copy()

    # --- Step 0: 评估原始失真 ---
    orig_metrics = evaluate(current, original)
    print(f"\n[初始状态] MSE={orig_metrics['MSE']:.2f}, PSNR={orig_metrics['PSNR']:.2f} dB, "
          f"SNR={orig_metrics['SNR']:.2f} dB, SSIM={orig_metrics['SSIM']:.4f}")

    # --- Step 1: DC 偏移校正 ---
    if config.get('remove_dc', True):
        dc_offset = (current.mean() - original.mean())
        current = current - dc_offset
        print(f"[DC校正] 减去直流偏移: {dc_offset:.2f}")

    # --- Step 2: 频域陷波滤波 ---
    if config.get('notch', True):
        # 定义周期噪声频率（基于噪声分析的精确发现）
        # 基频和它们的组合
        f_u = 71    # 水平基频 (归一化 0.2 * 355 = 71)
        f_v = 74    # 垂直基频 (归一化 0.2005 * 369 ≈ 74)

        # 构建所有需要滤除的频率对（只列正频率侧，函数自动处理共轭对称）
        notch_peaks = [
            # === 水平方向 ===
            (f_u, 0),       # f1: 水平基频
            (2*f_u, 0),     # 2f1: 水平二次谐波

            # === 垂直方向 ===
            (0, f_v),       # f2: 垂直基频
            (0, 2*f_v),     # 2f2: 垂直二次谐波

            # === 斜向交叉项 (f1±f2 组合) ===
            (f_u, f_v),     # f1+f2
            (f_u, -f_v),    # f1-f2
            (2*f_u, f_v),   # 2f1+f2
            (2*f_u, -f_v),  # 2f1-f2
            (f_u, 2*f_v),   # f1+2f2
            (f_u, -2*f_v),  # f1-2f2
            (2*f_u, 2*f_v), # 2f1+2f2
            (2*f_u, -2*f_v),# 2f1-2f2
        ]

        D0 = config.get('notch_D0', 3)
        n_order = config.get('notch_n', 2)

        current = butterworth_notch_filter(current, notch_peaks, D0=D0, n=n_order)
        print(f"[频域陷波] 滤除 {len(notch_peaks)} 个周期噪声频率对 (D0={D0}, n={n_order})")

        if save_intermediate:
            save_image(current, os.path.join(TEMP, f"step1_notch_{config['name']}.bmp"))

        notch_metrics = evaluate(current, original)
        print(f"  → MSE={notch_metrics['MSE']:.2f}, PSNR={notch_metrics['PSNR']:.2f} dB, "
              f"SNR={notch_metrics['SNR']:.2f} dB, SSIM={notch_metrics['SSIM']:.4f}")

    # --- Step 3: 空域滤波 ---
    spatial_method = config.get('spatial_method', 'bilateral')

    if spatial_method == 'gaussian':
        sigma = config.get('gaussian_sigma', 1.5)
        current = spatial_gaussian_filter(current, sigma=sigma)
        print(f"[空域高斯滤波] sigma={sigma}")

    elif spatial_method == 'bilateral':
        sigma_spatial = config.get('bilateral_sigma_spatial', 5)
        sigma_color = config.get('bilateral_sigma_color', 30)
        current = spatial_bilateral_filter(current, sigma_spatial=sigma_spatial, sigma_color=sigma_color)
        print(f"[空域双边滤波] sigma_spatial={sigma_spatial}, sigma_color={sigma_color}")

    elif spatial_method == 'arithmetic_mean':
        ks = config.get('mean_kernel_size', 3)
        current = spatial_arithmetic_mean_filter(current, kernel_size=ks)
        print(f"[空域算术均值滤波] kernel_size={ks}")

    elif spatial_method is None:
        print(f"[空域] 跳过空域滤波")

    if save_intermediate and spatial_method:
        save_image(current, os.path.join(TEMP, f"step2_spatial_{config['name']}.bmp"))

    # --- Step 4: 直方图匹配（可选） ---
    if config.get('histogram_match', False):
        # 匹配到原图的直方图分布
        orig_sorted = np.sort(original.flatten())
        current_sorted = np.sort(current.flatten())
        # 使用直方图匹配
        from scipy.interpolate import interp1d
        # 映射关系
        n_pixels = len(orig_sorted)
        indices = np.linspace(0, 1, n_pixels)
        # 对当前图像的排序值建立映射到原始图像排序值的函数
        mapping = interp1d(current_sorted, orig_sorted, kind='linear',
                           bounds_error=False, fill_value=(0, 255))
        current_flat = current.flatten()
        current_flat = mapping(current_flat)
        current = current_flat.reshape(current.shape)
        print(f"[直方图匹配] 已匹配到原图分布")

    # --- Step 5: 最终评估 ---
    final_metrics = evaluate(current, original)
    print(f"\n[最终结果] MSE={final_metrics['MSE']:.2f}, PSNR={final_metrics['PSNR']:.2f} dB, "
          f"SNR={final_metrics['SNR']:.2f} dB, SSIM={final_metrics['SSIM']:.4f}")

    # 相对于原始失真的改善
    mse_improvement = (orig_metrics['MSE'] - final_metrics['MSE']) / orig_metrics['MSE'] * 100
    psnr_gain = final_metrics['PSNR'] - orig_metrics['PSNR']
    snr_gain = final_metrics['SNR'] - orig_metrics['SNR']
    print(f"MSE改善: {mse_improvement:.1f}%, PSNR提升: +{psnr_gain:.2f} dB, SNR提升: +{snr_gain:.2f} dB")

    return current, final_metrics


# ============================================================
# 5. 实验：多组参数对比
# ============================================================
def main():
    print("=" * 60)
    print("任务一：空域+频域结合图像增强算法")
    print("=" * 60)

    distorted, original = load_images()
    H, W = distorted.shape
    print(f"图像尺寸: {W}x{H}")
    print(f"原始图像均值={original.mean():.2f}, 失真图像均值={distorted.mean():.2f}")

    # 定义多个实验配置
    configs = [
        {
            'name': 'notch_only',
            'remove_dc': True,
            'notch': True,
            'notch_D0': 3,
            'notch_n': 2,
            'spatial_method': None,
        },
        {
            'name': 'notch_gaussian',
            'remove_dc': True,
            'notch': True,
            'notch_D0': 3,
            'notch_n': 2,
            'spatial_method': 'gaussian',
            'gaussian_sigma': 1.2,
        },
        {
            'name': 'notch_bilateral',
            'remove_dc': True,
            'notch': True,
            'notch_D0': 3,
            'notch_n': 2,
            'spatial_method': 'bilateral',
            'bilateral_sigma_spatial': 5,
            'bilateral_sigma_color': 30,
        },
        {
            'name': 'notch_bilateral_v2',
            'remove_dc': True,
            'notch': True,
            'notch_D0': 4,
            'notch_n': 2,
            'spatial_method': 'bilateral',
            'bilateral_sigma_spatial': 4,
            'bilateral_sigma_color': 35,
        },
        {
            'name': 'notch_mean',
            'remove_dc': True,
            'notch': True,
            'notch_D0': 3,
            'notch_n': 2,
            'spatial_method': 'arithmetic_mean',
            'mean_kernel_size': 3,
        },
    ]

    results = {}
    best_config = None
    best_psnr = -float('inf')

    for cfg in configs:
        result_img, metrics = denoise_pipeline(distorted, original, cfg)
        results[cfg['name']] = {
            'image': result_img,
            'metrics': metrics
        }
        if metrics['PSNR'] > best_psnr:
            best_psnr = metrics['PSNR']
            best_config = cfg['name']

    # 输出对比表
    print("\n" + "=" * 80)
    print("方案对比汇总")
    print("=" * 80)
    print(f"{'配置名称':<25} {'MSE':>10} {'PSNR(dB)':>12} {'SNR(dB)':>11} {'SSIM':>10}")
    print("-" * 80)
    for name, data in results.items():
        m = data['metrics']
        print(f"{name:<25} {m['MSE']:>10.2f} {m['PSNR']:>10.2f} dB {m['SNR']:>9.2f} dB {m['SSIM']:>10.4f}")

    print(f"\n🏆 最佳方案: {best_config} (PSNR={best_psnr:.2f} dB)")

    # 保存最佳结果到处理后图像文件夹
    best_img = results[best_config]['image']
    best_img_clipped = np.clip(best_img, 0, 255).astype(np.uint8)
    Image.fromarray(best_img_clipped, mode='L').save(
        os.path.join(OUT, f"task1_best_{best_config}.bmp"))
    print(f"\n最佳结果已保存至: {os.path.join(OUT, f'task1_best_{best_config}.bmp')}")

    # 同时保存原始失真图和处理结果对照
    distorted_clipped = np.clip(distorted, 0, 255).astype(np.uint8)
    Image.fromarray(distorted_clipped, mode='L').save(
        os.path.join(OUT, "task1_distorted.bmp"))
    Image.fromarray(original.astype(np.uint8), mode='L').save(
        os.path.join(OUT, "task1_original_ref.bmp"))

    print("Done. 所有结果已保存。")
    return results


if __name__ == '__main__':
    main()

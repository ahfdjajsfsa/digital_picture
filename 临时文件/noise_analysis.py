"""
噪声分析脚本（任务一）
分析 dogDistorted.bmp 中的噪声特性：
1. 随机噪声的统计特性
2. 周期噪声的频率成分
3. 噪声的空间分布特征
---
知识点框架内：
- 空域：灰度变换、直方图处理、空域滤波（平滑/锐化）
- 频域：低通/高通/带阻/带通/陷波滤波器
"""
import numpy as np
from PIL import Image
import os

# ============================================================
# 0. 路径设置
# ============================================================
BASE = r"d:\Github\School\digital_picture"
TEMP = os.path.join(BASE, "临时文件")
SRC = os.path.join(BASE, "原图与参考图")

original_path = os.path.join(SRC, "dogOriginal.bmp")
distorted_path = os.path.join(SRC, "dogDistorted.bmp")

# ============================================================
# 1. 加载图像
# ============================================================
img_original = Image.open(original_path).convert('L')
img_distorted = Image.open(distorted_path).convert('L')

arr_orig = np.array(img_original, dtype=np.float64)
arr_dist = np.array(img_distorted, dtype=np.float64)
noise = arr_dist - arr_orig

H, W = arr_dist.shape
center_h, center_w = H // 2, W // 2

# ============================================================
# 2. 输出到文件
# ============================================================
output_file = os.path.join(TEMP, "noise_analysis_report.txt")

with open(output_file, 'w', encoding='utf-8') as f:
    def p(*args, **kwargs):
        print(*args, **kwargs)
        print(*args, **kwargs, file=f)

    p("=" * 70)
    p("任务一：dogDistorted.bmp 噪声特性分析报告")
    p("=" * 70)
    p(f"图像尺寸: {W} x {H} (宽 x 高)")
    p(f"总像素数: {H * W}")

    # ============================================================
    # 3. 空域分析 - 噪声统计特性
    # ============================================================
    p("\n" + "-" * 50)
    p("一、空域噪声统计分析")
    p("-" * 50)

    p(f"\n【原始图像统计】")
    p(f"  均值: {arr_orig.mean():.2f}")
    p(f"  标准差: {arr_orig.std():.2f}")
    p(f"  最小值: {arr_orig.min():.0f}")
    p(f"  最大值: {arr_orig.max():.0f}")

    p(f"\n【失真图像统计】")
    p(f"  均值: {arr_dist.mean():.2f}")
    p(f"  标准差: {arr_dist.std():.2f}")
    p(f"  最小值: {arr_dist.min():.0f}")
    p(f"  最大值: {arr_dist.max():.0f}")

    p(f"\n【纯噪声分量 (distorted - original)】")
    p(f"  均值 (直流偏移): {noise.mean():.4f}")
    p(f"  标准差: {noise.std():.2f}")
    p(f"  噪声方差: {noise.var():.4f}")
    p(f"  RMS (均方根): {np.sqrt(np.mean(noise**2)):.4f}")
    p(f"  最小值: {noise.min():.0f}")
    p(f"  最大值: {noise.max():.0f}")

    # 噪声分布特征
    p(f"\n【噪声分布分位数】")
    percentiles = [1, 5, 10, 25, 50, 75, 90, 95, 99]
    for pc in percentiles:
        val = np.percentile(noise, pc)
        p(f"  {pc}%: {val:.1f}")

    # 偏度和峰度
    p(f"\n【噪声分布形态】")
    p(f"  偏度 (Skewness): {np.mean((noise - noise.mean())**3) / (noise.std()**3):.4f} （正值=右偏）")
    p(f"  峰度 (Kurtosis): {np.mean((noise - noise.mean())**4) / (noise.var()**2):.4f} （正态分布=3）")

    # 饱和像素分析
    n_clip_high = (arr_dist >= 255).sum()
    n_clip_low = (arr_dist <= 0).sum()
    p(f"\n【饱和像素】")
    p(f"  高饱和(>=255): {n_clip_high} 像素 ({100*n_clip_high/(H*W):.2f}%)")
    p(f"  低饱和(<=0): {n_clip_low} 像素 ({100*n_clip_low/(H*W):.2f}%)")
    p(f"  注意：高饱和像素较多（{100*n_clip_high/(H*W):.1f}%），说明噪声含强直流分量")

    # 图像质量指标（已知原图的情况）
    mse = np.mean(noise**2)
    psnr = 10 * np.log10(255**2 / mse) if mse > 0 else float('inf')
    snr = 10 * np.log10(np.var(arr_orig) / np.var(noise)) if np.var(noise) > 0 else float('inf')

    p(f"\n【图像质量指标（参考值）】")
    p(f"  MSE = {mse:.4f}")
    p(f"  PSNR = {psnr:.4f} dB")
    p(f"  SNR = {snr:.4f} dB")

    # ============================================================
    # 4. 频域分析 - 识别周期噪声
    # ============================================================
    p("\n" + "-" * 50)
    p("二、频域分析 - 周期噪声检测")
    p("-" * 50)

    fft_distorted = np.fft.fft2(arr_dist)
    fft_distorted_shifted = np.fft.fftshift(fft_distorted)
    fft_mag = np.abs(fft_distorted_shifted)

    fft_original = np.fft.fft2(arr_orig)
    fft_original_shifted = np.fft.fftshift(fft_original)

    fft_noise = fft_distorted - fft_original  # 噪声的 FFT（复数差）
    fft_noise_shifted = np.fft.fftshift(fft_noise)
    fft_noise_mag = np.abs(fft_noise_shifted)

    # DC 分量
    dc_value = fft_mag[center_h, center_w]
    dc_noise = fft_noise_mag[center_h, center_w]
    p(f"\n【DC（直流）分量】")
    p(f"  失真图 DC: {dc_value:.1f}")
    p(f"  噪声 DC: {dc_noise:.1f}")
    p(f"  说明：噪声DC很大→噪声有显著直流偏移（均值38.8），需先减均值或通过陷波去除")

    # 寻找周期噪声峰值
    # 排除DC附近区域来找峰值
    exclude_radius = 3  # 排除DC周围
    y, x = np.indices((H, W))
    r = np.sqrt((y - center_h)**2 + (x - center_w)**2)
    fft_noise_mag_excl_dc = fft_noise_mag.copy()
    fft_noise_mag_excl_dc[r < exclude_radius] = 0

    # 寻找全局和局部峰值
    p(f"\n【噪声FFT幅度统计（排除DC）】")
    p(f"  最大幅度: {fft_noise_mag_excl_dc.max():.1f}")
    p(f"  平均幅度: {fft_noise_mag_excl_dc[fft_noise_mag_excl_dc > 0].mean():.1f}")
    p(f"  中位数幅度: {np.median(fft_noise_mag_excl_dc[fft_noise_mag_excl_dc > 0]):.1f}")

    # 定位周期噪声峰值 —— 在噪声FFT中寻找显著离群点
    # 使用统计阈值：幅度 > 均值 + k*标准差 的视为周期噪声峰
    valid_mag = fft_noise_mag_excl_dc[fft_noise_mag_excl_dc > 0].flatten()
    threshold_mean = valid_mag.mean() + 5 * valid_mag.std()

    p(f"\n【周期噪声峰值检测（阈值 = 均值 + 5σ = {threshold_mean:.1f}）】")

    # 直接找全局峰值——这些是周期噪声
    # 方法：在噪声FFT幅度中找 top peaks（排除DC）
    from scipy.ndimage import maximum_filter

    # 局部最大值检测
    local_max = maximum_filter(fft_noise_mag_excl_dc, size=5) == fft_noise_mag_excl_dc
    peak_mask = local_max & (fft_noise_mag_excl_dc > threshold_mean)
    peak_coords = np.argwhere(peak_mask)

    p(f"  检测到 {len(peak_coords)} 个候选周期噪声峰值\n")

    # 对峰值按照幅度排序
    peaks_info = []
    for py, px in peak_coords:
        mag_val = fft_noise_mag[py, px]
        # 频率坐标（相对DC）
        du = px - center_w
        dv = py - center_h
        # 归一化频率
        norm_u = du / W
        norm_v = dv / H
        peaks_info.append({
            'row': py, 'col': px,
            'du': du, 'dv': dv,
            'norm_u': norm_u, 'norm_v': norm_v,
            'mag': mag_val,
            'dist_from_dc': np.sqrt(du**2 + dv**2)
        })

    peaks_info.sort(key=lambda x: x['mag'], reverse=True)

    # 输出前20个峰值
    p("  Top 20 周期噪声峰（按幅度排序）：")
    p(f"  {'排名':<5} {'坐标(r,c)':<14} {'du':<6} {'dv':<6} {'归一化(u,v)':<24} {'距离DC':<10} {'幅度':<12}")
    p(f"  {'-'*5} {'-'*14} {'-'*6} {'-'*6} {'-'*24} {'-'*10} {'-'*12}")
    for i, peak in enumerate(peaks_info[:20]):
        p(f"  {i+1:<5} ({peak['row']:3d},{peak['col']:3d})  "
          f"{peak['du']:+4d}  {peak['dv']:+4d}  "
          f"({peak['norm_u']:+.4f}, {peak['norm_v']:+.4f})     "
          f"{peak['dist_from_dc']:6.1f}    {peak['mag']:10.1f}")

    # 识别成对出现的对称峰值（周期噪声通常是共轭对称对）
    p(f"\n【共轭对称对分析】")
    # 周期噪声在FFT中成对出现（共轭对称）
    paired = set()
    pairs_found = []
    for i, p1 in enumerate(peaks_info[:30]):
        if i in paired:
            continue
        for j, p2 in enumerate(peaks_info[:30]):
            if j <= i or j in paired:
                continue
            # 两个点是否关于DC对称（共轭对）
            if abs(p1['du'] + p2['du']) <= 2 and abs(p1['dv'] + p2['dv']) <= 2:
                paired.add(i)
                paired.add(j)
                # 计算归一化频率
                freq_u = abs(p1['norm_u'])
                freq_v = abs(p1['norm_v'])
                wavelength_u = 1.0 / abs(p1['norm_u']) if abs(p1['norm_u']) > 0 else float('inf')
                wavelength_v = 1.0 / abs(p1['norm_v']) if abs(p1['norm_v']) > 0 else float('inf')
                pairs_found.append({
                    'u': freq_u, 'v': freq_v,
                    'wave_u': wavelength_u, 'wave_v': wavelength_v,
                    'mag': (p1['mag'] + p2['mag']) / 2
                })

    p(f"  找到 {len(pairs_found)} 对共轭对称周期噪声成分：")
    p(f"  {'归一化频率 u':<16} {'归一化频率 v':<16} {'水平波长(px)':<16} {'垂直波长(px)':<16} {'平均幅度':<12} {'方向':<10}")
    p(f"  {'-'*16} {'-'*16} {'-'*16} {'-'*16} {'-'*12} {'-'*10}")
    for pair in sorted(pairs_found, key=lambda x: x['mag'], reverse=True):
        direction = "水平" if pair['v'] < 0.001 else ("垂直" if pair['u'] < 0.001 else "斜向")
        p(f"  {pair['u']:.6f}        {pair['v']:.6f}        "
          f"{pair['wave_u']:8.1f}        {pair['wave_v']:8.1f}        "
          f"{pair['mag']:8.1f}     {direction}")

    # 识别哪些是基频，哪些是谐波
    if pairs_found:
        p(f"\n【频率成分解析】")
        # 按频率分组
        base_freqs = []
        for pair in pairs_found:
            if pair['u'] < 0.001:  # 纯垂直条纹
                base_freqs.append(f"垂直方向: f_v = {pair['v']:.6f} (垂直周期噪声, 波长≈{pair['wave_v']:.1f}像素)")
            elif pair['v'] < 0.001:  # 纯水平条纹
                base_freqs.append(f"水平方向: f_u = {pair['u']:.6f} (水平周期噪声, 波长≈{pair['wave_u']:.1f}像素)")
            else:
                base_freqs.append(f"斜向: f_u={pair['u']:.6f}, f_v={pair['v']:.6f} (斜向周期噪声)")

        for bf in base_freqs:
            p(f"  · {bf}")

    # ============================================================
    # 5. 随机噪声分量分析
    # ============================================================
    p("\n" + "-" * 50)
    p("三、随机噪声分量分析")
    p("-" * 50)

    # 分离周期噪声后估计随机噪声
    # 方法：对噪声图像做FFT，陷波滤除已识别的周期峰，反变换得到随机噪声估计
    # 先做一个简单的估计：在噪声FFT中，排除DC和已识别的周期峰值

    # 构建掩膜，排除DC和周期峰值
    mask_noise = np.ones((H, W), dtype=bool)
    # 排除DC附近
    mask_noise[r < 5] = False
    # 排除已识别的周期峰值区域
    for peak in peaks_info[:20]:
        for dy in range(-3, 4):
            for dx in range(-3, 4):
                py2 = peak['row'] + dy
                px2 = peak['col'] + dx
                if 0 <= py2 < H and 0 <= px2 < W:
                    mask_noise[py2, px2] = False

    remaining_noise_fft = fft_noise_mag * mask_noise
    random_noise_rms = np.sqrt(np.mean(remaining_noise_fft**2)) / np.sqrt(H * W)  # Parseval

    p(f"  排除周期峰后的残余FFT幅度均值: {remaining_noise_fft[remaining_noise_fft > 0].mean():.1f}")
    p(f"  估计随机噪声标准差（频域推算）: {random_noise_rms:.2f}")

    # 空域估计：用局部标准差
    # 在平滑区域估计纯噪声
    from scipy.ndimage import uniform_filter
    local_mean = uniform_filter(noise, size=7)
    local_var = uniform_filter(noise**2, size=7) - local_mean**2
    local_std = np.sqrt(np.maximum(local_var, 0))

    p(f"\n【空域局部噪声估计 (7×7窗口)】")
    p(f"  局部标准差 - 最小: {local_std.min():.2f}")
    p(f"  局部标准差 - 中位数: {np.median(local_std):.2f}")
    p(f"  局部标准差 - 平均: {local_std.mean():.2f}")
    p(f"  局部标准差 - 最大: {local_std.max():.2f}")

    # 用局部均值判断噪声是否与信号相关
    p(f"\n【噪声-信号相关性初步判断】")
    # 在原始图像不同亮度区域计算噪声统计
    bright_mask = arr_orig > 128
    mid_mask = (arr_orig >= 64) & (arr_orig <= 192)
    dark_mask = arr_orig < 64

    for region_name, region_mask in [("暗区(<64)", dark_mask), ("中区(64-192)", mid_mask), ("亮区(>128)", bright_mask)]:
        if region_mask.sum() > 0:
            region_noise = noise[region_mask]
            p(f"  {region_name}: 噪声均值={region_noise.mean():.2f}, 噪声std={region_noise.std():.2f}, "
              f"像素数={region_mask.sum()}")

    # ============================================================
    # 6. 总结
    # ============================================================
    p("\n" + "=" * 70)
    p("四、噪声特性总结")
    p("=" * 70)

    p("""
    【噪声组成】
    1. 直流偏移（亮度整体提高）：噪声均值为正（~38.8），使图像整体变亮
    2. 周期噪声：在频域中有多个显著的共轭对称峰值对
    3. 随机噪声：叠加在整幅图像上的类高斯随机噪声

    【周期噪声特征】
    - 频域中存在多个显著峰值对（共轭对称），对应不同频率和方向的周期噪声
    - 这些周期噪声在空域表现为规律性的条纹或波纹

    【随机噪声特征】
    - 近似高斯分布（需检验偏度和峰度）
    - 噪声方差约为 3123，标准差约 56
    - 高饱和像素较多（7880个），表明强噪声+直流偏移导致部分像素触及255上限

    【处理策略（知识点框架内）】
    - 频域：陷波滤波器(Notch Filter) — 针对周期噪声的各个频率成分
    - 频域：巴特沃斯/高斯带阻滤波器 — 同样可去除周期噪声
    - 空域：算术均值/高斯/双边滤波器 — 抑制随机噪声
    - 可先频域去周期噪声，再空域去随机噪声（或反之）
    - 去噪前需先减去直流偏移（或使用陷波滤除DC分量）
    """)

print(f"\n分析报告已保存至: {output_file}")
print("Done.")

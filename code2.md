# 第二题：降质图像复原实验记录

## 1. 任务与图像

处理图像：`D:\Github\School\digital_picture\原图与参考图\blurred wood.bmp`

该图像为 `1280 x 960` 彩色降质图像。直接观察可见树叶、树干、告示牌和地面纹理均存在方向性拖影，属于较典型的运动模糊退化。题目没有提供清晰参考图，因此本题不能严格计算以清晰图为基准的 MSE、PSNR、SNR、SSIM。本实验先分析图像退化特征，估计退化函数 `H(u,v)`，再进行维纳滤波复原，并用无参考清晰度指标与视觉伪影检查评价结果。

复现脚本：`D:\Github\School\digital_picture\code2.m`

主要输出：

- `D:\Github\School\digital_picture\处理后图像\task2_degradation_analysis.png`
- `D:\Github\School\digital_picture\处理后图像\task2_algorithm_flowchart.png`
- `D:\Github\School\digital_picture\处理后图像\task2_degradation_estimate.txt`
- `D:\Github\School\digital_picture\处理后图像\task2_degradation_candidates.csv`
- `D:\Github\School\digital_picture\处理后图像\task2_input_blurred.bmp`
- `D:\Github\School\digital_picture\处理后图像\task2_wiener_only.bmp`
- `D:\Github\School\digital_picture\处理后图像\task2_final_result.bmp`
- `D:\Github\School\digital_picture\处理后图像\task2_comparison.png`
- `D:\Github\School\digital_picture\处理后图像\task2_metrics.txt`
- `D:\Github\School\digital_picture\处理后图像\task2_metrics_chart.png`

按 `goal.md` 要求，临时分析图和预览结果也放在 `原图与参考图` 目录中，包括：

- `D:\Github\School\digital_picture\原图与参考图\task2_degradation_analysis.png`
- `D:\Github\School\digital_picture\原图与参考图\task2_algorithm_flowchart.png`
- `D:\Github\School\digital_picture\原图与参考图\task2_degradation_estimate.txt`
- `D:\Github\School\digital_picture\原图与参考图\task2_degradation_candidates.csv`
- `D:\Github\School\digital_picture\原图与参考图\task2_temp_wiener_only.bmp`
- `D:\Github\School\digital_picture\原图与参考图\task2_temp_final_result.bmp`

## 2. 图像分析结论

先不直接调复原参数，而是从图像频域和倒谱域分析退化函数：

1. 将 RGB 图像转换到 YCbCr 空间，只分析亮度通道 `Y`。
2. 选取主体区域，进行局部直方图均衡、高斯低频扣除和 Hann 窗加权，减少自然图像低频和边界对频谱的干扰。
3. 计算对数幅度谱 `log(1 + |FFT(Y)|)`，并增强暗条纹。运动模糊会使频谱中出现近似平行的暗条纹，其方向与运动方向近似垂直。
4. 对暗条纹增强图做 Radon 变换，得到候选条纹角度。
5. 计算倒谱图，在候选运动方向上寻找峰值，用于估计运动位移长度。
6. 用候选 `length / theta / NSR` 构造 PSF，在缩小图上进行快速维纳复原评分，选出较稳定的退化函数参数。

当前脚本得到的频谱暗条纹候选角度为：

```text
[-17.5, -40.5, -39, 25, -13.5, 33, 31, -15] degrees
```

频谱角度存在多个近峰，说明该图不是严格单一匀速运动退化，而是自然场景纹理、相机抖动和局部高光共同影响的结果。因此脚本没有只取某一个 Radon 峰，而是把方向候选放入退化函数校准步骤中继续筛选。

倒谱长度候选为：

```text
[14, 30, 36, 43] pixels
```

结合频谱方向候选、倒谱长度候选和受约束复原评分，最终选取的退化函数参数为：

```text
motion_psf_length = 16 pixels
motion_psf_theta  = 81 degrees
NSR               = 0.05
```

该估计结果记录在：

```text
D:\Github\School\digital_picture\处理后图像\task2_degradation_estimate.txt
```

退化分析图为：

```text
D:\Github\School\digital_picture\处理后图像\task2_degradation_analysis.png
```

## 3. 退化模型

采用图像退化模型：

```text
g(x, y) = f(x, y) * h(x, y) + n(x, y)
```

其中：

- `g(x, y)` 为输入降质图像；
- `f(x, y)` 为待估计的清晰图像；
- `h(x, y)` 为点扩散函数 PSF；
- `n(x, y)` 为加性噪声；
- `*` 表示卷积。

本实验采用线性运动模糊 PSF：

```text
h(x, y) = 1/L,  当 (x, y) 位于长度 L、方向 theta 的运动轨迹上
h(x, y) = 0,    其他位置
sum(h) = 1
```

脚本中用 MATLAB 构造：

```matlab
psf = fspecial('motion', 16, 81);
H = psf2otf(psf, size(Y));
```

频域退化函数为：

```text
H(u, v) = FFT2{h(x, y)}
```

维纳滤波公式为：

```text
F_hat(u, v) = H*(u, v)G(u, v) / (|H(u, v)|^2 + NSR)
```

其中 `H*(u, v)` 是 `H(u, v)` 的共轭，`NSR` 是噪声功率谱与原图功率谱之比。

## 4. 算法框图

已生成报告用算法框图：

```text
D:\Github\School\digital_picture\处理后图像\task2_algorithm_flowchart.png
```

框图流程如下：

```text
输入降质图像 g(x,y)
    ↓
亮度分离、ROI 与窗函数
    ↓
FFT 对数频谱
    ↓
Radon + 倒谱估计 L 与 theta
    ↓
构造 h(x,y), H(u,v)
    ↓
维纳滤波 conj(H)/(|H|^2+NSR)
    ↓
融合增强输出 f_hat(x,y)
```

这部分对应“依据问题分析结论设计算法”的要求：先根据图像频谱和倒谱判断退化类型，再建立退化函数，最后进行复原。

## 5. 最终复原流程

完整流程：

1. 读取 `blurred wood.bmp`，转换到 YCbCr 空间。
2. 对亮度通道进行频谱、Radon 和倒谱分析，估计运动模糊退化函数。
3. 构造 `length = 16`、`theta = 81°` 的运动模糊 PSF，并计算频域退化函数 `H(u,v)`。
4. 对亮度通道执行维纳反卷积。
5. 构造 Sobel 结构区域掩膜：树干、地面、告示牌等纹理区域使用更多复原细节；天空高亮区域降低细节融合比例。
6. 对维纳结果做双边滤波，降低反卷积颗粒。
7. 将维纳细节与原亮度通道自适应融合，再做温和锐化和局部对比度增强。
8. 保护红色日期水印，避免反卷积破坏叠加文字。
9. 将复原亮度通道与原色度通道合成，得到最终彩色复原图。

后处理参数：

```text
pad_size          = 96 pixels
luminance_blend   = 0.52
unsharp_amount    = 1.10
unsharp_radius    = 2.00
sharpen_threshold = 0.018
local_contrast    = 0.15
```

## 6. 评价指标

由于没有清晰参考图，本实验采用无参考指标：

- `Tenengrad`：基于 Sobel 梯度能量的清晰度指标，越大表示边缘响应越强。
- `Laplacian variance`：二阶微分响应方差，越大表示边缘和细节响应越强，但过大也可能表示噪声或过锐化。
- `Entropy`：灰度信息熵，反映图像灰度层次和信息量。
- `High-frequency std`：高频成分标准差，用于观察高频细节增强和噪声放大。
- `Clip ratio`：接近纯黑或纯白的像素比例，用于观察是否出现严重截断。

当前脚本运行得到的结果：

| 指标 | 输入图像 | 维纳结果 | 最终结果 | 最终/输入 |
|---|---:|---:|---:|---:|
| Tenengrad | 0.071366 | 0.149998 | 0.197435 | 2.7665 |
| Laplacian variance | 0.008669 | 0.009437 | 0.027441 | 3.1654 |
| Entropy | 7.054049 | 7.267876 | 7.340885 | 1.0407 |
| High-frequency std | 0.037588 | 0.053390 | 0.063820 | 1.6978 |
| Clip ratio | 0.000000 | 0.007047 | 0.017305 | - |

指标分析：

- Tenengrad 提高约 `176.7%`，说明复原后边缘响应明显增强。
- Laplacian variance 提高约 `216.5%`，说明二阶边缘响应和局部纹理更强。
- Entropy 提高约 `4.1%`，说明灰度层次和局部信息量有所增加。
- High-frequency std 提高约 `69.8%`，说明细节增强明显，但仍需控制反卷积噪声。
- Clip ratio 为 `0.017305`，主要来自过曝天空和锐化后的极亮区域，没有出现大面积黑白截断。

视觉观察：

- 输入图像整体发糊，树干、枝叶、地面和告示牌轮廓都不清楚。
- 单纯维纳反卷积能提高边缘，但树冠和天空区域仍会有振铃和颗粒。
- 最终结果比输入图更清楚，右侧告示牌边框、中心树干边缘、地面落叶纹理和前景枝叶层次有所增强。
- 由于原图退化较重且没有清晰参考图，最终结果不能等同于真实清晰原图；本方案更强调基于退化函数估计的稳定复原，而不是无约束锐化。

## 7. 运行方式

在 MATLAB 中运行：

```matlab
run('D:\Github\School\digital_picture\code2.m')
```

或在命令行运行：

```powershell
matlab -batch "run('D:\Github\School\digital_picture\code2.m')"
```

脚本会重新生成第二题的退化函数分析图、算法框图、候选参数表、复原图、对比图和指标文件。

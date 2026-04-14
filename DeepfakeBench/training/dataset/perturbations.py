"""
DeeperForensics-1.0 官方扰动实现 (适配 DeepfakeBench 框架)
============================================================
Source: https://github.com/EndlessSora/DeeperForensics-1.0/blob/master/perturbation/distortions.py

注意: DeepfakeBench 的 abstract_dataset.py 中 load_rgb() 返回的是 RGB 格式的图像,
      而 DeeperForensics 官方代码使用 BGR 格式。
      本文件对所有涉及颜色空间转换的函数做了适配 (RGB <-> YCbCr)。

7种扰动 × 5个强度等级:
  CS   - Color Saturation Change     饱和度  [0.4, 0.3, 0.2, 0.1, 0.0]
  CC   - Color Contrast Change       对比度  [0.85, 0.725, 0.6, 0.475, 0.35]
  BW   - Block Wise Distortion       块遮挡  [16, 32, 48, 64, 80]
  GNC  - Gaussian Noise (Color)      噪声    [0.001, 0.002, 0.005, 0.01, 0.05]
  GB   - Gaussian Blur               模糊    [7, 9, 13, 17, 21]
  JPEG - JPEG Compression            压缩    [2, 3, 4, 5, 6]
  VC   - Video Compression (CRF)     视频压缩 [30, 32, 35, 38, 40] (image-level不用)
"""

import math
import random
import cv2
import numpy as np


# ============================================================================
# 颜色空间转换 (适配 RGB 输入)
# ============================================================================

def rgb2ycbcr(img_rgb):
    """RGB -> YCbCr (从官方 bgr2ycbcr 适配)"""
    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    img_bgr = img_bgr.astype(np.float32)
    img_ycrcb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2YCR_CB)
    img_ycbcr = img_ycrcb[:, :, (0, 2, 1)].astype(np.float32)
    img_ycbcr[:, :, 0] = (img_ycbcr[:, :, 0] * (235 - 16) + 16) / 255.0
    img_ycbcr[:, :, 1:] = (img_ycbcr[:, :, 1:] * (240 - 16) + 16) / 255.0
    return img_ycbcr


def ycbcr2rgb(img_ycbcr):
    """YCbCr -> RGB (从官方 ycbcr2bgr 适配)"""
    img_ycbcr = img_ycbcr.astype(np.float32)
    img_ycbcr[:, :, 0] = (img_ycbcr[:, :, 0] * 255.0 - 16) / (235 - 16)
    img_ycbcr[:, :, 1:] = (img_ycbcr[:, :, 1:] * 255.0 - 16) / (240 - 16)
    img_ycrcb = img_ycbcr[:, :, (0, 2, 1)].astype(np.float32)
    img_bgr = cv2.cvtColor(img_ycrcb, cv2.COLOR_YCR_CB2BGR)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    return img_rgb


# ============================================================================
# 6种 Image-level 扰动函数 (严格遵循官方实现)
# ============================================================================

def color_saturation(img, param):
    """
    颜色饱和度变化 (CS)
    在 YCbCr 空间缩放 Cb/Cr 通道, param 越小饱和度越低
    """
    ycbcr = rgb2ycbcr(img)
    ycbcr[:, :, 1] = 0.5 + (ycbcr[:, :, 1] - 0.5) * param
    ycbcr[:, :, 2] = 0.5 + (ycbcr[:, :, 2] - 0.5) * param
    img = ycbcr2rgb(ycbcr)
    return np.clip(img * 255, 0, 255).astype(np.uint8) if img.max() <= 1.0 else np.clip(img, 0, 255).astype(np.uint8)


def color_contrast(img, param):
    """
    颜色对比度变化 (CC)
    直接对像素值乘以缩放因子, param 越小对比度越低
    """
    img = img.astype(np.float32) * param
    return np.clip(img, 0, 255).astype(np.uint8)


def block_wise(img, param):
    """
    局部块状失真 (BW)
    随机位置覆盖 8×8 灰色块, param 控制块数量
    """
    img = img.copy()
    width = 8
    block = np.ones((width, width, 3), dtype=np.uint8) * 128
    num_blocks = min(img.shape[0], img.shape[1]) // 256 * param
    for _ in range(int(num_blocks)):
        r_w = random.randint(0, max(0, img.shape[1] - 1 - width))
        r_h = random.randint(0, max(0, img.shape[0] - 1 - width))
        img[r_h:r_h + width, r_w:r_w + width, :] = block
    return img


def gaussian_noise_color(img, param):
    """
    YCbCr 空间高斯噪声 (GNC)
    在 YCbCr 颜色空间添加白高斯噪声, param = 噪声方差
    """
    ycbcr = rgb2ycbcr(img) / 255.0
    noise = math.sqrt(param) * np.random.randn(*ycbcr.shape)
    b = (ycbcr + noise) * 255.0
    img = ycbcr2rgb(b.astype(np.float32))
    return np.clip(img * 255 if img.max() <= 1.0 else img, 0, 255).astype(np.uint8)


def gaussian_blur(img, param):
    """
    高斯模糊 (GB)
    param = 高斯核大小 (奇数), sigma = param / 6
    """
    # 需要先转BGR再转回来，因为GaussianBlur对通道顺序不敏感，可直接处理
    return cv2.GaussianBlur(img, (param, param), param * 1.0 / 6)


def jpeg_compression(img, param):
    """
    JPEG 压缩/像素化 (JPEG)
    先缩小再放大, param = 缩放因子
    """
    h, w = img.shape[:2]
    s_h = max(1, h // param)
    s_w = max(1, w // param)
    img = cv2.resize(img, (s_w, s_h))
    img = cv2.resize(img, (w, h))
    return img


# ============================================================================
# 参数表 & 统一接口
# ============================================================================

PERTURBATION_PARAMS = {
    #           level-1    level-2    level-3    level-4    level-5
    'CS':   [   0.4,       0.3,       0.2,       0.1,       0.0     ],
    'CC':   [   0.85,      0.725,     0.6,       0.475,     0.35    ],
    'BW':   [   16,        32,        48,        64,        80      ],
    'GNC':  [   0.001,     0.002,     0.005,     0.01,      0.05    ],
    'GB':   [   7,         9,         13,        17,        21      ],
    'JPEG': [   2,         3,         4,         5,         6       ],
}

PERTURBATION_FUNCTIONS = {
    'CS':   color_saturation,
    'CC':   color_contrast,
    'BW':   block_wise,
    'GNC':  gaussian_noise_color,
    'GB':   gaussian_blur,
    'JPEG': jpeg_compression,
}

PERTURBATION_NAMES = {
    'CS':   'Color Saturation',
    'CC':   'Color Contrast',
    'BW':   'Block Wise',
    'GNC':  'Gaussian Noise',
    'GB':   'Gaussian Blur',
    'JPEG': 'JPEG Compression',
}

# 默认使用6种 image-level 扰动 (不含 VC)
DEFAULT_PERTURBATION_TYPES = ['CS', 'CC', 'BW', 'GNC', 'GB', 'JPEG']


def apply_perturbation(img_rgb, perturbation_type, severity_level):
    """
    对 RGB 图像施加 DeeperForensics-1.0 官方扰动

    Args:
        img_rgb: np.ndarray, shape (H, W, 3), uint8, RGB format
        perturbation_type: str, 'CS'|'CC'|'BW'|'GNC'|'GB'|'JPEG'
        severity_level: int, 1-5 (0 表示不施加扰动)

    Returns:
        perturbed image: np.ndarray, same shape, uint8, RGB format
    """
    if severity_level == 0 or perturbation_type is None:
        return img_rgb

    assert perturbation_type in PERTURBATION_PARAMS, \
        f"Unknown perturbation type: {perturbation_type}. Choose from {list(PERTURBATION_PARAMS.keys())}"
    assert 1 <= severity_level <= 5, \
        f"Severity level must be 1-5, got {severity_level}"

    param = PERTURBATION_PARAMS[perturbation_type][severity_level - 1]
    func = PERTURBATION_FUNCTIONS[perturbation_type]

    return func(img_rgb.copy(), param)
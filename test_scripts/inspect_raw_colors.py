#!/usr/bin/env python3
"""
检查原始文字区域的RGB颜色值
不使用Otsu二值化,直接分析文字像素的真实颜色
"""
import cv2
import numpy as np
import easyocr
import sys
import os

def inspect_text_colors(image_path):
    """检查文字区域的原始RGB颜色"""

    # 1. 读取图像
    img = cv2.imread(image_path)
    if img is None:
        print(f"错误: 无法读取图像 {image_path}")
        return

    h, w = img.shape[:2]
    print(f"图像尺寸: {w}x{h}\n")

    # 2. 初始化 EasyOCR
    print("初始化 EasyOCR...")
    reader = easyocr.Reader(['ch_sim', 'en'], gpu=False, verbose=False)
    print("检测文字区域...\n")

    # 3. 检测文字
    results = reader.readtext(image_path)
    print(f"检测到 {len(results)} 个文字区域\n")
    print("=" * 80)

    # 4. 分析每个文字区域的颜色
    all_colors = []

    for i, (bbox, text, conf) in enumerate(results):
        points = np.array(bbox, dtype=np.int32)

        # 过滤异常宽高比
        box_w = np.max(points[:, 0]) - np.min(points[:, 0])
        box_h = np.max(points[:, 1]) - np.min(points[:, 1])
        aspect_ratio = box_w / max(box_h, 1)

        if aspect_ratio > 5 or aspect_ratio < 0.2:
            continue

        # 提取 ROI
        x_min = max(0, np.min(points[:, 0]))
        y_min = max(0, np.min(points[:, 1]))
        x_max = min(w, np.max(points[:, 0]))
        y_max = min(h, np.max(points[:, 1]))

        roi = img[y_min:y_max, x_min:x_max]

        if roi.size == 0:
            continue

        # 方法1: 直接分析所有像素的RGB值
        all_pixels_bgr = roi.reshape(-1, 3)
        all_pixels_rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB).reshape(-1, 3)

        # 计算整个ROI的颜色统计
        mean_rgb = np.mean(all_pixels_rgb, axis=0).astype(int)
        median_rgb = np.median(all_pixels_rgb, axis=0).astype(int)
        std_rgb = np.std(all_pixels_rgb, axis=0).astype(int)

        # 方法2: 只分析最暗的像素(可能是墨迹)
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        dark_threshold = np.percentile(gray, 25)  # 最暗的25%像素
        dark_mask = gray < dark_threshold

        if np.sum(dark_mask) > 0:
            dark_pixels_rgb = all_pixels_rgb[dark_mask.flatten()]
            dark_mean_rgb = np.mean(dark_pixels_rgb, axis=0).astype(int)
            dark_median_rgb = np.median(dark_pixels_rgb, axis=0).astype(int)
        else:
            dark_mean_rgb = mean_rgb
            dark_median_rgb = median_rgb

        # 计算HSV饱和度
        hsv = cv2.cvtColor(median_rgb.reshape(1,1,3).astype(np.uint8), cv2.COLOR_RGB2HSV)[0,0]

        all_colors.append({
            'mean_rgb': mean_rgb,
            'median_rgb': median_rgb,
            'dark_mean_rgb': dark_mean_rgb,
            'dark_median_rgb': dark_median_rgb,
            'hsv': hsv,
            'text': text
        })

        # 每10个区域输出一次详细信息
        if i % 10 == 0 or i < 5:
            print(f"\n区域 {i}: 文字='{text[:10]}...' (置信度={conf:.2f})")
            print(f"  所有像素均值RGB:  {mean_rgb}  (标准差={std_rgb})")
            print(f"  所有像素中位数RGB: {median_rgb}")
            print(f"  最暗25%像素均值RGB: {dark_mean_rgb}")
            print(f"  最暗25%像素中位RGB: {dark_median_rgb}")
            print(f"  中位数HSV: {hsv} (饱和度={hsv[1]})")

    print("\n" + "=" * 80)
    print("\n整体颜色统计分析:")
    print("-" * 80)

    if len(all_colors) == 0:
        print("未检测到有效文字区域")
        return

    # 统计所有区域的颜色分布
    all_mean_rgb = np.array([c['mean_rgb'] for c in all_colors])
    all_median_rgb = np.array([c['median_rgb'] for c in all_colors])
    all_dark_median_rgb = np.array([c['dark_median_rgb'] for c in all_colors])
    all_saturations = np.array([c['hsv'][1] for c in all_colors])

    print(f"\n1. 所有像素均值RGB分布:")
    print(f"   R范围: [{all_mean_rgb[:,0].min()}, {all_mean_rgb[:,0].max()}]")
    print(f"   G范围: [{all_mean_rgb[:,1].min()}, {all_mean_rgb[:,1].max()}]")
    print(f"   B范围: [{all_mean_rgb[:,2].min()}, {all_mean_rgb[:,2].max()}]")

    print(f"\n2. 所有像素中位数RGB分布:")
    print(f"   R范围: [{all_median_rgb[:,0].min()}, {all_median_rgb[:,0].max()}]")
    print(f"   G范围: [{all_median_rgb[:,1].min()}, {all_median_rgb[:,1].max()}]")
    print(f"   B范围: [{all_median_rgb[:,2].min()}, {all_median_rgb[:,2].max()}]")

    print(f"\n3. 最暗像素中位数RGB分布(可能是真实墨迹):")
    print(f"   R范围: [{all_dark_median_rgb[:,0].min()}, {all_dark_median_rgb[:,0].max()}]")
    print(f"   G范围: [{all_dark_median_rgb[:,1].min()}, {all_dark_median_rgb[:,1].max()}]")
    print(f"   B范围: [{all_dark_median_rgb[:,2].min()}, {all_dark_median_rgb[:,2].max()}]")

    print(f"\n4. 饱和度分布:")
    print(f"   范围: [{all_saturations.min()}, {all_saturations.max()}]")
    print(f"   平均值: {all_saturations.mean():.1f}")
    print(f"   中位数: {np.median(all_saturations):.1f}")

    # 分析颜色聚集情况
    print(f"\n5. 颜色区分度分析:")

    # 计算RGB最暗像素的方差
    rgb_variance = np.var(all_dark_median_rgb, axis=0)
    print(f"   最暗像素RGB方差: R={rgb_variance[0]:.1f}, G={rgb_variance[1]:.1f}, B={rgb_variance[2]:.1f}")

    # 判断是否有区分度
    if np.max(rgb_variance) < 100:
        print(f"   ⚠️  RGB方差很小(<100),颜色高度相似,难以区分")
    elif np.max(rgb_variance) < 500:
        print(f"   ⚠️  RGB方差较小(<500),颜色略有差异,但区分度不高")
    else:
        print(f"   ✓  RGB方差较大(≥500),颜色有明显差异,可以区分")

    if all_saturations.mean() < 30:
        print(f"   ⚠️  平均饱和度很低(<30),大部分是灰色调")
    elif all_saturations.mean() < 60:
        print(f"   ⚠️  平均饱和度较低(<60),彩色不明显")
    else:
        print(f"   ✓  平均饱和度较高(≥60),有明显彩色")

    print("\n" + "=" * 80)
    print("\n结论:")
    print("-" * 80)

    # 给出诊断结论
    if np.max(rgb_variance) < 100 and all_saturations.mean() < 30:
        print("❌ 文档墨迹严重褪色,所有颜色都是灰色调,无法通过RGB颜色区分")
        print("   建议: 使用亮度分类或墨迹深度分析")
    elif all_saturations.mean() < 30:
        print("⚠️  文档墨迹褪色成灰色,但RGB值略有差异,可尝试聚类")
        print("   建议: 使用Lab色彩空间 + K-Means聚类")
    else:
        print("✓ 文档颜色保存良好,可以使用RGB颜色分类")
        print("   建议: 直接使用RGB或HSV色彩空间 + K-Means聚类")

    print("=" * 80)

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("用法: python inspect_raw_colors.py <图像路径>")
        sys.exit(1)

    image_path = sys.argv[1]

    if not os.path.exists(image_path):
        print(f"错误: 文件不存在 - {image_path}")
        sys.exit(1)

    inspect_text_colors(image_path)

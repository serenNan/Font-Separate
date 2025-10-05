#!/usr/bin/env python3
"""
基于色调(Hue)分类 - 忽略明度差异，只看颜色倾向
适合褪色文档
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2
import numpy as np
import easyocr
import json

def hex_to_hsv(hex_color):
    """将十六进制颜色转换为HSV"""
    hex_color = hex_color.lstrip('#')
    rgb = tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
    # RGB -> BGR -> HSV
    bgr = np.array([[[rgb[2], rgb[1], rgb[0]]]], dtype=np.uint8)
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)[0, 0]
    return hsv

def is_warm_color(rgb, min_warmth=5):
    """
    判断是否为暖色调(橙/褐/米色)

    Args:
        rgb: RGB颜色
        min_warmth: 最小暖度(R-B差值)
    """
    r, g, b = rgb
    # 暖色特征: R > B, 且R-B差距够大
    warmth = r - b
    return warmth >= min_warmth

def classify_by_warmth(image_path, target_hex, min_warmth=5, output_dir='results'):
    """
    根据暖色调分类(识别橙/褐/米色系)

    Args:
        image_path: 图像路径
        target_hex: 参考颜色(用于确定色调倾向)
        min_warmth: 最小暖度(R-B差值阈值)
        output_dir: 输出目录
    """
    # 目标颜色的HSV
    target_hsv = hex_to_hsv(target_hex)
    target_rgb = tuple(int(target_hex.lstrip('#')[i:i+2], 16) for i in (0, 2, 4))

    print(f"参考颜色: #{target_hex.lstrip('#')}")
    print(f"  RGB: {target_rgb}")
    print(f"  HSV: H={target_hsv[0]}° S={target_hsv[1]} V={target_hsv[2]}")
    print(f"  色调范围: {target_hsv[0]-10}° ~ {target_hsv[0]+10}° (橙/褐色系)")
    print(f"暖度阈值(R-B): {min_warmth}")
    print("=" * 80)

    # 读取图像
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"无法读取图像: {image_path}")

    h, w = img.shape[:2]
    print(f"图片尺寸: {w}x{h}")

    # 初始化 EasyOCR
    print("初始化 EasyOCR...")
    reader = easyocr.Reader(['ch_sim', 'en'], gpu=False, verbose=False)
    print("检测文字区域...\n")

    # 检测文字
    results = reader.readtext(image_path)
    print(f"检测到 {len(results)} 个文字区域\n")
    print("=" * 80)

    # 分类
    warm_boxes = []
    cool_boxes = []

    warm_colors = []
    cool_colors = []

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

        # 提取颜色(所有像素的中位数RGB)
        all_pixels = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB).reshape(-1, 3)
        median_rgb = tuple(np.median(all_pixels, axis=0).astype(int))

        # 计算暖度
        warmth = median_rgb[0] - median_rgb[2]  # R - B

        # 转HSV看色调
        bgr = np.array([[[median_rgb[2], median_rgb[1], median_rgb[0]]]], dtype=np.uint8)
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)[0, 0]

        # 分类: 暖色调(橙/褐/米色)
        if is_warm_color(median_rgb, min_warmth):
            warm_boxes.append(bbox)
            warm_colors.append(median_rgb)
            print(f"区域 {i}: RGB{median_rgb} 暖度={warmth:+3d} H={hsv[0]:3d}° → ✓ 暖色系 (文字: {text[:10]}...)")
        else:
            cool_boxes.append(bbox)
            cool_colors.append(median_rgb)
            if i % 20 == 0:  # 只打印部分
                print(f"区域 {i}: RGB{median_rgb} 暖度={warmth:+3d} H={hsv[0]:3d}° → ✗ 冷色/中性")

    print("\n" + "=" * 80)
    print(f"分类结果: 暖色系(橙褐米)={len(warm_boxes)}个, 冷色/中性={len(cool_boxes)}个")

    # 生成结果图像
    os.makedirs(output_dir, exist_ok=True)
    base_name = os.path.splitext(os.path.basename(image_path))[0]

    # 标注图
    annotated = img.copy()

    # 绘制暖色区域(红色边框)
    for bbox in warm_boxes:
        points = np.array(bbox, dtype=np.int32)
        cv2.polylines(annotated, [points], True, (0, 0, 255), 3)

    # 绘制冷色区域(蓝色边框)
    for bbox in cool_boxes:
        points = np.array(bbox, dtype=np.int32)
        cv2.polylines(annotated, [points], True, (255, 0, 0), 1)

    annotated_path = os.path.join(output_dir, f"{base_name}_warm_annotated.jpg")
    cv2.imwrite(annotated_path, annotated)

    # 暖色掩码
    warm_mask = np.zeros((h, w), dtype=np.uint8)
    for bbox in warm_boxes:
        points = np.array(bbox, dtype=np.int32)
        cv2.fillPoly(warm_mask, [points], 255)

    warm_result = cv2.bitwise_and(img, img, mask=warm_mask)
    warm_path = os.path.join(output_dir, f"{base_name}_warm_colors.jpg")
    cv2.imwrite(warm_path, warm_result)

    # 冷色掩码
    cool_mask = np.zeros((h, w), dtype=np.uint8)
    for bbox in cool_boxes:
        points = np.array(bbox, dtype=np.int32)
        cv2.fillPoly(cool_mask, [points], 255)

    cool_result = cv2.bitwise_and(img, img, mask=cool_mask)
    cool_path = os.path.join(output_dir, f"{base_name}_cool_colors.jpg")
    cv2.imwrite(cool_path, cool_result)

    # 统计信息
    stats = {
        'reference_color_hex': f"#{target_hex.lstrip('#')}",
        'min_warmth': min_warmth,
        'warm_count': len(warm_boxes),
        'cool_count': len(cool_boxes),
        'warm_color_samples': [[int(x) for x in c] for c in warm_colors[:20]],
        'cool_color_samples': [[int(x) for x in c] for c in cool_colors[:10]]
    }

    stats_path = os.path.join(output_dir, f"{base_name}_warm_stats.json")
    with open(stats_path, 'w', encoding='utf-8') as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)

    print("\n输出文件:")
    print(f"  标注图(红框=暖色): {annotated_path}")
    print(f"  暖色区域: {warm_path}")
    print(f"  冷色区域: {cool_path}")
    print(f"  统计信息: {stats_path}")
    print("=" * 80)

    # 打印颜色统计
    if len(warm_colors) > 0:
        warm_colors_arr = np.array(warm_colors)
        print(f"\n暖色系区域RGB统计:")
        print(f"  平均值: R={warm_colors_arr[:,0].mean():.1f} "
              f"G={warm_colors_arr[:,1].mean():.1f} "
              f"B={warm_colors_arr[:,2].mean():.1f}")
        print(f"  范围: R[{warm_colors_arr[:,0].min()}-{warm_colors_arr[:,0].max()}] "
              f"G[{warm_colors_arr[:,1].min()}-{warm_colors_arr[:,1].max()}] "
              f"B[{warm_colors_arr[:,2].min()}-{warm_colors_arr[:,2].max()}]")
        print(f"  暖度范围: {(warm_colors_arr[:,0] - warm_colors_arr[:,2]).min():.0f} ~ "
              f"{(warm_colors_arr[:,0] - warm_colors_arr[:,2]).max():.0f}")

    print("\n✓ 完成")

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("用法: python color_hue_classify.py <图像路径> [参考颜色] [暖度阈值]")
        print("\n示例:")
        print("  python color_hue_classify.py Pictures/原始.jpg")
        print("  python color_hue_classify.py Pictures/原始.jpg A16D48")
        print("  python color_hue_classify.py Pictures/原始.jpg A16D48 10")
        sys.exit(1)

    image_path = sys.argv[1]
    target_color = sys.argv[2] if len(sys.argv) > 2 else 'A16D48'
    min_warmth = int(sys.argv[3]) if len(sys.argv) > 3 else 5

    if not os.path.exists(image_path):
        print(f"错误: 文件不存在 - {image_path}")
        sys.exit(1)

    classify_by_warmth(image_path, target_color, min_warmth)

#!/usr/bin/env python3
"""
目标颜色分类 - 根据指定的颜色进行分类
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2
import numpy as np
import easyocr
import json

def hex_to_rgb(hex_color):
    """将十六进制颜色转换为RGB"""
    hex_color = hex_color.lstrip('#')
    return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))

def color_distance(color1, color2):
    """计算两个RGB颜色的欧氏距离"""
    return np.sqrt(np.sum((np.array(color1) - np.array(color2)) ** 2))

def classify_by_target_color(image_path, target_hex, threshold=50, output_dir='results'):
    """
    根据目标颜色分类文字

    Args:
        image_path: 图像路径
        target_hex: 目标颜色(十六进制,如 'A16D48' 或 '#A16D48')
        threshold: 颜色距离阈值,小于此值认为是目标颜色
        output_dir: 输出目录
    """
    # 转换目标颜色
    target_rgb = hex_to_rgb(target_hex)
    print(f"目标颜色: #{target_hex.lstrip('#')} = RGB{target_rgb}")
    print(f"颜色距离阈值: {threshold}")
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
    target_boxes = []
    other_boxes = []

    target_colors = []
    other_colors = []

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
        median_rgb = np.median(all_pixels, axis=0).astype(int)

        # 计算与目标颜色的距离
        distance = color_distance(median_rgb, target_rgb)

        # 分类
        if distance <= threshold:
            target_boxes.append(bbox)
            target_colors.append(median_rgb)
            print(f"区域 {i}: RGB{list(median_rgb)} 距离={distance:.1f} → ✓ 目标颜色 (文字: {text[:10]}...)")
        else:
            other_boxes.append(bbox)
            other_colors.append(median_rgb)
            if i % 20 == 0:  # 只打印部分
                print(f"区域 {i}: RGB{list(median_rgb)} 距离={distance:.1f} → ✗ 其他颜色")

    print("\n" + "=" * 80)
    print(f"分类结果: 目标颜色={len(target_boxes)}个, 其他颜色={len(other_boxes)}个")

    # 生成结果图像
    os.makedirs(output_dir, exist_ok=True)
    base_name = os.path.splitext(os.path.basename(image_path))[0]

    # 标注图
    annotated = img.copy()

    # 绘制目标颜色区域(红色边框)
    for bbox in target_boxes:
        points = np.array(bbox, dtype=np.int32)
        cv2.polylines(annotated, [points], True, (0, 0, 255), 3)

    # 绘制其他颜色区域(蓝色边框)
    for bbox in other_boxes:
        points = np.array(bbox, dtype=np.int32)
        cv2.polylines(annotated, [points], True, (255, 0, 0), 1)

    annotated_path = os.path.join(output_dir, f"{base_name}_target_{target_hex.lstrip('#')}_annotated.jpg")
    cv2.imwrite(annotated_path, annotated)

    # 目标颜色掩码
    target_mask = np.zeros((h, w), dtype=np.uint8)
    for bbox in target_boxes:
        points = np.array(bbox, dtype=np.int32)
        cv2.fillPoly(target_mask, [points], 255)

    target_result = cv2.bitwise_and(img, img, mask=target_mask)
    target_path = os.path.join(output_dir, f"{base_name}_target_{target_hex.lstrip('#')}.jpg")
    cv2.imwrite(target_path, target_result)

    # 其他颜色掩码
    other_mask = np.zeros((h, w), dtype=np.uint8)
    for bbox in other_boxes:
        points = np.array(bbox, dtype=np.int32)
        cv2.fillPoly(other_mask, [points], 255)

    other_result = cv2.bitwise_and(img, img, mask=other_mask)
    other_path = os.path.join(output_dir, f"{base_name}_other_colors.jpg")
    cv2.imwrite(other_path, other_result)

    # 统计信息
    stats = {
        'target_color_hex': f"#{target_hex.lstrip('#')}",
        'target_color_rgb': list(target_rgb),
        'threshold': threshold,
        'target_count': len(target_boxes),
        'other_count': len(other_boxes),
        'target_color_samples': [[int(x) for x in c] for c in target_colors[:10]],  # 最多10个样本
        'other_color_samples': [[int(x) for x in c] for c in other_colors[:10]]
    }

    stats_path = os.path.join(output_dir, f"{base_name}_target_{target_hex.lstrip('#')}_stats.json")
    with open(stats_path, 'w', encoding='utf-8') as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)

    print("\n输出文件:")
    print(f"  标注图(红框=目标): {annotated_path}")
    print(f"  目标颜色区域: {target_path}")
    print(f"  其他颜色区域: {other_path}")
    print(f"  统计信息: {stats_path}")
    print("=" * 80)

    # 打印颜色统计
    if len(target_colors) > 0:
        target_colors_arr = np.array(target_colors)
        print(f"\n目标颜色区域RGB统计:")
        print(f"  平均值: R={target_colors_arr[:,0].mean():.1f} "
              f"G={target_colors_arr[:,1].mean():.1f} "
              f"B={target_colors_arr[:,2].mean():.1f}")
        print(f"  范围: R[{target_colors_arr[:,0].min()}-{target_colors_arr[:,0].max()}] "
              f"G[{target_colors_arr[:,1].min()}-{target_colors_arr[:,1].max()}] "
              f"B[{target_colors_arr[:,2].min()}-{target_colors_arr[:,2].max()}]")

    print("\n✓ 完成")

if __name__ == '__main__':
    if len(sys.argv) < 3:
        print("用法: python color_target_classify.py <图像路径> <目标颜色> [阈值]")
        print("\n示例:")
        print("  python color_target_classify.py Pictures/原始.jpg A16D48")
        print("  python color_target_classify.py Pictures/原始.jpg A16D48 50")
        print("  python color_target_classify.py Pictures/原始.jpg '#A16D48' 30")
        sys.exit(1)

    image_path = sys.argv[1]
    target_color = sys.argv[2]
    threshold = int(sys.argv[3]) if len(sys.argv) > 3 else 50

    if not os.path.exists(image_path):
        print(f"错误: 文件不存在 - {image_path}")
        sys.exit(1)

    classify_by_target_color(image_path, target_color, threshold)

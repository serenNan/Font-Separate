#!/usr/bin/env python3
"""
基于色相(Hue)范围分类 - 指定目标颜色,自动识别相同色系(不论深浅)
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2
import numpy as np
import easyocr
import json
from sklearn.cluster import KMeans

def hex_to_hsv(hex_color):
    """将十六进制颜色转换为HSV"""
    hex_color = hex_color.lstrip('#')
    rgb = tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
    bgr = np.array([[[rgb[2], rgb[1], rgb[0]]]], dtype=np.uint8)
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)[0, 0]
    return hsv

def is_white_pixel(pixel_rgb, threshold=200):
    """检测白色像素 - RGB三通道都≥阈值"""
    r, g, b = pixel_rgb
    return r >= threshold and g >= threshold and b >= threshold

def classify_by_hue(image_path, target_hex='FFFFFF', hue_tolerance=15, min_saturation=10, output_dir='results'):
    """
    检测白色区域

    Args:
        image_path: 图像路径
        output_dir: 输出目录
    """
    print(f"检测白色区域(RGB≥200,200,200)")
    print("=" * 80)

    # 读取图像
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"无法读取图像: {image_path}")

    h, w = img.shape[:2]
    print(f"图片尺寸: {w}x{h}")
    print(f"总像素数: {h*w}")
    print("逐像素检测白色(255,255,255)...")

    # 转RGB
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # 创建白色掩码
    white_mask = np.zeros((h, w), dtype=np.uint8)

    # 逐像素检测并统计RGB值
    white_count = 0
    rgb_count = {}  # 统计每个RGB值的出现次数

    for y in range(h):
        for x in range(w):
            pixel = img_rgb[y, x]

            # 统计RGB值
            rgb_key = tuple(pixel)
            rgb_count[rgb_key] = rgb_count.get(rgb_key, 0) + 1

            # 检测白色
            if is_white_pixel(pixel):
                white_mask[y, x] = 255
                white_count += 1

        # 每处理100行打印进度
        if (y + 1) % 100 == 0:
            print(f"进度: {y+1}/{h} 行, 已检测到 {white_count} 个白色像素, {len(rgb_count)} 种RGB值")

    total_pixels = h * w
    white_ratio = white_count / total_pixels * 100

    print("\n" + "=" * 80)
    print(f"检测完成: 白色像素={white_count}/{total_pixels} ({white_ratio:.2f}%)")
    print(f"总计发现 {len(rgb_count)} 种不同的RGB颜色")

    # 收集非白色像素进行颜色聚类
    print("\n对非白色像素进行颜色聚类...")
    non_white_colors = []
    for (r, g, b), count in rgb_count.items():
        if not is_white_pixel((r, g, b), threshold=200):
            non_white_colors.extend([(r, g, b)] * min(count, 100))  # 限制每种颜色最多100个样本

    print(f"非白色样本数: {len(non_white_colors)}")

    if len(non_white_colors) > 0:
        # K-Means聚类(自动确定最佳K值,2-8类)
        non_white_array = np.array(non_white_colors)

        best_k = 3
        best_score = -1

        print("寻找最佳聚类数...")
        for k in range(2, min(9, len(non_white_colors))):
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            labels = kmeans.fit_predict(non_white_array)
            from sklearn.metrics import silhouette_score
            score = silhouette_score(non_white_array, labels)
            print(f"  K={k}: silhouette={score:.3f}")
            if score > best_score:
                best_score = score
                best_k = k

        print(f"最佳聚类数: K={best_k} (silhouette={best_score:.3f})")

        # 使用最佳K值进行聚类
        kmeans = KMeans(n_clusters=best_k, random_state=42, n_init=10)
        kmeans.fit(non_white_array)

        # 聚类中心
        centers = kmeans.cluster_centers_.astype(int)
        print(f"\n聚类中心颜色:")
        for i, center in enumerate(centers):
            hex_color = f"#{center[0]:02X}{center[1]:02X}{center[2]:02X}"
            print(f"  类别{i}: RGB{tuple(center)} = {hex_color}")

        # 构建RGB到类别的映射表(优化查找速度)
        print("\n构建颜色分类映射表...")
        rgb_to_class = {}
        for (r, g, b) in rgb_count.keys():
            if r >= 200 and g >= 200 and b >= 200:
                rgb_to_class[(r, g, b)] = 0  # 白色
            else:
                pixel = np.array([r, g, b])
                distances = [np.linalg.norm(pixel - center) for center in centers]
                class_id = np.argmin(distances)
                rgb_to_class[(r, g, b)] = class_id + 1

        # 向量化分配类别
        print("为图像像素分配颜色类别(优化算法)...")
        color_class_map = np.zeros((h, w), dtype=np.uint8)

        # 先分配白色
        white_mask = (img_rgb[:,:,0] >= 200) & (img_rgb[:,:,1] >= 200) & (img_rgb[:,:,2] >= 200)
        color_class_map[white_mask] = 0

        # 对非白色像素批量分类
        non_white_mask = ~white_mask
        non_white_pixels = img_rgb[non_white_mask]

        print(f"  对{len(non_white_pixels)}个非白色像素进行分类...")
        # 使用kmeans.predict批量预测
        labels = kmeans.predict(non_white_pixels) + 1  # +1因为0是白色

        # 分配类别
        color_class_map[non_white_mask] = labels

        # 为每个类别生成独立图像
        print(f"\n生成{best_k}个颜色类别的分离图像...")
        for class_id in range(best_k):
            class_mask = (color_class_map == class_id + 1).astype(np.uint8) * 255
            class_result = cv2.bitwise_and(img, img, mask=class_mask)
            class_path = os.path.join(output_dir, f"{base_name}_color_class_{class_id}.jpg")
            cv2.imwrite(class_path, class_result)

            pixel_count = np.sum(color_class_map == class_id + 1)
            center = centers[class_id]
            hex_color = f"#{center[0]:02X}{center[1]:02X}{center[2]:02X}"
            print(f"  类别{class_id}: RGB{tuple(center)} ({hex_color}) - {pixel_count}像素 -> {class_path}")
    else:
        print("没有非白色像素,跳过聚类")

    # 生成结果图像
    os.makedirs(output_dir, exist_ok=True)
    base_name = os.path.splitext(os.path.basename(image_path))[0]

    # 白色像素结果
    white_result = cv2.bitwise_and(img, img, mask=white_mask)
    white_path = os.path.join(output_dir, f"{base_name}_white_pixels.jpg")
    cv2.imwrite(white_path, white_result)

    # 非白色像素结果(反转掩码)
    non_white_mask = cv2.bitwise_not(white_mask)
    non_white_result = cv2.bitwise_and(img, img, mask=non_white_mask)
    non_white_path = os.path.join(output_dir, f"{base_name}_non_white_pixels.jpg")
    cv2.imwrite(non_white_path, non_white_result)

    # 白色掩码图(白色显示为白色,其他为黑色)
    mask_path = os.path.join(output_dir, f"{base_name}_white_mask.jpg")
    cv2.imwrite(mask_path, white_mask)

    # 按出现次数排序RGB值
    sorted_rgb = sorted(rgb_count.items(), key=lambda x: x[1], reverse=True)

    # 保存RGB统计到文本文件
    rgb_txt_path = os.path.join(output_dir, f"{base_name}_rgb_colors.txt")
    with open(rgb_txt_path, 'w', encoding='utf-8') as f:
        f.write(f"图片: {image_path}\n")
        f.write(f"总像素数: {total_pixels}\n")
        f.write(f"不同颜色数: {len(rgb_count)}\n")
        f.write("=" * 80 + "\n\n")
        f.write("RGB值统计 (按出现次数排序):\n")
        f.write("-" * 80 + "\n")
        f.write(f"{'RGB值':<20} {'十六进制':<12} {'出现次数':<12} {'占比':<10}\n")
        f.write("-" * 80 + "\n")

        for (r, g, b), count in sorted_rgb:
            hex_color = f"#{r:02X}{g:02X}{b:02X}"
            percentage = count / total_pixels * 100
            f.write(f"RGB({r:3d},{g:3d},{b:3d})   {hex_color:<12} {count:<12} {percentage:>6.2f}%\n")

    # JSON格式保存(只保存前1000个最常见的颜色)
    rgb_json = []
    for (r, g, b), count in sorted_rgb[:1000]:
        rgb_json.append({
            'rgb': [int(r), int(g), int(b)],
            'hex': f"#{r:02X}{g:02X}{b:02X}",
            'count': int(count),
            'percentage': float(count / total_pixels * 100)
        })

    rgb_json_path = os.path.join(output_dir, f"{base_name}_rgb_colors.json")
    with open(rgb_json_path, 'w', encoding='utf-8') as f:
        json.dump({
            'total_pixels': int(total_pixels),
            'unique_colors': len(rgb_count),
            'white_pixels': int(white_count),
            'white_percentage': float(white_ratio),
            'top_colors': rgb_json
        }, f, indent=2, ensure_ascii=False)

    print("\n输出文件:")
    print(f"  白色像素图: {white_path}")
    print(f"  非白色像素图: {non_white_path}")
    print(f"  白色掩码图: {mask_path}")
    print(f"  RGB统计(文本): {rgb_txt_path}")
    print(f"  RGB统计(JSON): {rgb_json_path}")
    print("=" * 80)

    # 显示前20个最常见的颜色
    print("\n前20个最常见的RGB颜色:")
    print("-" * 80)
    print(f"{'RGB值':<20} {'十六进制':<12} {'出现次数':<12} {'占比':<10}")
    print("-" * 80)
    for (r, g, b), count in sorted_rgb[:20]:
        hex_color = f"#{r:02X}{g:02X}{b:02X}"
        percentage = count / total_pixels * 100
        print(f"RGB({r:3d},{g:3d},{b:3d})   {hex_color:<12} {count:<12} {percentage:>6.2f}%")

    print("\n✓ 完成")

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("用法: python color_hue_range_classify.py <图像路径> [目标颜色] [色相容差] [最小饱和度]")
        print("\n示例:")
        print("  python color_hue_range_classify.py Pictures/原始.jpg")
        print("  python color_hue_range_classify.py Pictures/原始.jpg A16D48")
        print("  python color_hue_range_classify.py Pictures/原始.jpg A16D48 15")
        print("  python color_hue_range_classify.py Pictures/原始.jpg A16D48 15 10")
        sys.exit(1)

    image_path = sys.argv[1]
    target_color = sys.argv[2] if len(sys.argv) > 2 else 'A16D48'
    hue_tolerance = int(sys.argv[3]) if len(sys.argv) > 3 else 15
    min_saturation = int(sys.argv[4]) if len(sys.argv) > 4 else 10

    if not os.path.exists(image_path):
        print(f"错误: 文件不存在 - {image_path}")
        sys.exit(1)

    classify_by_hue(image_path, target_color, hue_tolerance, min_saturation)

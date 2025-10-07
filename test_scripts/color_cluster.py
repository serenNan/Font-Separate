#!/usr/bin/env python3
"""
非白色内容颜色聚类分类
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

def classify_colors(image_path, white_threshold=200, grid_size=20, output_dir='results'):
    """
    颜色聚类分类

    Args:
        image_path: 图像路径
        white_threshold: 白色阈值(RGB≥此值为白色)
        grid_size: 网格大小(像素),默认20x20像素网格(约1mm×1mm)
        output_dir: 输出目录
    """
    print(f"颜色聚类分类(白色阈值RGB≥{white_threshold}, 网格大小={grid_size}×{grid_size}像素)")
    print("=" * 80)

    # 读取图像
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"无法读取图像: {image_path}")

    h, w = img.shape[:2]
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    print(f"图片尺寸: {w}x{h}, 总像素: {h*w}")

    # 将图像分成网格,每个网格取中位数颜色
    print(f"将图像分成{grid_size}×{grid_size}像素的网格...")
    grid_h = h // grid_size
    grid_w = w // grid_size
    print(f"网格数量: {grid_w}×{grid_h} = {grid_w * grid_h}个网格")

    grid_colors = []
    grid_positions = []

    for i in range(grid_h):
        for j in range(grid_w):
            y_start = i * grid_size
            y_end = min((i + 1) * grid_size, h)
            x_start = j * grid_size
            x_end = min((j + 1) * grid_size, w)

            # 提取网格区域
            grid_region = img_rgb[y_start:y_end, x_start:x_end]

            # 过滤掉白色像素(RGB≥white_threshold)
            pixels = grid_region.reshape(-1, 3)
            non_white_pixels = pixels[
                (pixels[:, 0] < white_threshold) |
                (pixels[:, 1] < white_threshold) |
                (pixels[:, 2] < white_threshold)
            ]

            # 如果网格内有足够的非白色像素,计算中位数颜色
            if len(non_white_pixels) > 10:  # 至少10个非白色像素
                median_color = np.median(non_white_pixels, axis=0).astype(np.uint8)
                grid_colors.append(median_color)
                grid_positions.append((i, j))

    grid_colors = np.array(grid_colors)
    print(f"提取了{len(grid_colors)}个非白色网格(已过滤网格内白色背景)")

    if len(grid_colors) == 0:
        print("没有非白色网格!")
        return

    # 转换为HSV(只使用色相H进行聚类)
    print("转换为HSV颜色空间...")
    grids_bgr = cv2.cvtColor(grid_colors.reshape(1, -1, 3), cv2.COLOR_RGB2BGR)
    hsv = cv2.cvtColor(grids_bgr, cv2.COLOR_BGR2HSV).reshape(-1, 3)

    # 只提取色相H (0-180度)
    hues = hsv[:, 0].reshape(-1, 1)
    saturations = hsv[:, 1]

    # 过滤掉饱和度太低的灰色像素(S<30)
    colored_mask = saturations >= 30
    colored_hues = hues[colored_mask]
    colored_indices = np.where(colored_mask)[0]

    print(f"有色网格数: {len(colored_hues)} (饱和度≥30)")
    print(f"灰色网格数: {len(grid_colors) - len(colored_hues)} (饱和度<30)")

    if len(colored_hues) < 10:
        print("警告: 几乎没有彩色网格,可能都是黑/灰色")
        # 降低饱和度阈值
        colored_mask = saturations >= 10
        colored_hues = hues[colored_mask]
        colored_indices = np.where(colored_mask)[0]
        print(f"降低饱和度阈值到10后,有色网格数: {len(colored_hues)}")

    # 采样
    max_samples = 20000
    if len(colored_hues) > max_samples:
        print(f"采样{max_samples}个有色网格进行聚类...")
        sample_indices = np.random.choice(len(colored_hues), max_samples, replace=False)
        samples = colored_hues[sample_indices]
    else:
        samples = colored_hues

    # 寻找最佳聚类数(基于色相)
    print("\n寻找最佳聚类数(基于色相H,K=2-6)...")
    best_k = 2
    best_score = -1

    for k in range(2, min(7, len(samples)//100)):
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(samples)
        if len(np.unique(labels)) > 1:
            score = silhouette_score(samples, labels)
            print(f"  K={k}: silhouette={score:.3f}")
            if score > best_score:
                best_score = score
                best_k = k

    print(f"\n最佳聚类数: K={best_k} (silhouette={best_score:.3f})")

    # 使用最佳K值聚类
    print("执行K-Means聚类(基于色相)...")
    kmeans = KMeans(n_clusters=best_k, random_state=42, n_init=10)
    kmeans.fit(colored_hues)

    # 聚类中心(色相)
    hue_centers = kmeans.cluster_centers_.flatten().astype(int)
    print(f"\n聚类中心色相:")
    for i, hue in enumerate(hue_centers):
        print(f"  类别{i}: 色相H={hue}° (0=红, 60=黄, 120=绿, 180=青)")

    # 为所有有色网格分配类别
    print("\n为所有有色网格分配类别...")
    colored_labels = kmeans.predict(colored_hues)

    # 为所有网格分配类别(灰色先暂时单独一类)
    labels_all = np.full(len(grid_colors), best_k, dtype=np.uint8)  # 默认灰色类
    labels_all[colored_indices] = colored_labels

    # 对灰色类进行二次聚类(分为深灰和浅灰)
    print("\n对灰色/黑色网格进行二次聚类...")
    gray_mask = ~colored_mask
    gray_grids = grid_colors[gray_mask]
    gray_indices = np.where(gray_mask)[0]

    if len(gray_grids) > 10:
        # 使用明度V进行聚类(区分深灰和浅灰)
        gray_hsv = hsv[gray_mask]
        gray_values = gray_hsv[:, 2].reshape(-1, 1)  # V通道(明度)

        # 灰色分为2类
        print(f"  对{len(gray_grids)}个灰色网格按明度分为2类...")
        gray_kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
        gray_labels = gray_kmeans.fit_predict(gray_values)

        # 确定哪个是深色,哪个是浅色
        gray_centers = gray_kmeans.cluster_centers_.flatten()
        if gray_centers[0] < gray_centers[1]:
            dark_gray_id = 0  # 深色
            light_gray_id = 1  # 浅色
        else:
            dark_gray_id = 1
            light_gray_id = 0

        print(f"    深灰明度: V={int(gray_centers[dark_gray_id])}")
        print(f"    浅灰明度: V={int(gray_centers[light_gray_id])}")

        # 更新labels_all(best_k=深灰, best_k+1=浅灰)
        for i, idx in enumerate(gray_indices):
            if gray_labels[i] == dark_gray_id:
                labels_all[idx] = best_k  # 深灰
            else:
                labels_all[idx] = best_k + 1  # 浅灰

        total_classes = best_k + 2  # 彩色类 + 深灰 + 浅灰
    else:
        total_classes = best_k + 1

    # 计算每个类别的代表颜色(取该类别网格的中位数RGB)
    centers = []
    for i in range(total_classes):
        class_grids = grid_colors[labels_all == i]
        if len(class_grids) > 0:
            center_rgb = np.median(class_grids, axis=0).astype(int)
            centers.append(center_rgb)
        else:
            centers.append(np.array([128, 128, 128]))

    centers = np.array(centers)

    print(f"\n各类别代表颜色:")
    for i, center in enumerate(centers):
        hex_color = f"#{center[0]:02X}{center[1]:02X}{center[2]:02X}"
        if i < best_k:
            print(f"  类别{i}(色相{hue_centers[i]}°): RGB{tuple(center)} = {hex_color}")
        elif i == best_k:
            print(f"  类别{i}(深灰/黑色): RGB{tuple(center)} = {hex_color}")
        else:
            print(f"  类别{i}(浅灰色): RGB{tuple(center)} = {hex_color}")

    # 生成分类图
    print("生成分类图...")
    os.makedirs(output_dir, exist_ok=True)
    base_name = os.path.splitext(os.path.basename(image_path))[0]

    # 从网格级别映射回像素级别
    color_class_map = np.zeros((h, w), dtype=np.uint8)
    for idx, (i, j) in enumerate(grid_positions):
        y_start = i * grid_size
        y_end = min((i + 1) * grid_size, h)
        x_start = j * grid_size
        x_end = min((j + 1) * grid_size, w)
        color_class_map[y_start:y_end, x_start:x_end] = labels_all[idx] + 1

    # 为每个类别生成独立图像
    print(f"\n生成{total_classes}个颜色类别的分离图像:")
    for class_id in range(total_classes):
        class_mask = (color_class_map == class_id + 1).astype(np.uint8) * 255

        # 创建白色背景图像
        class_result = np.ones_like(img) * 255
        # 只保留该类别的像素
        class_result = cv2.bitwise_and(img, img, mask=class_mask, dst=class_result)
        # 将非该类别区域设为白色
        white_mask = (class_mask == 0).astype(np.uint8) * 255
        class_result[white_mask == 255] = [255, 255, 255]

        class_path = os.path.join(output_dir, f"{base_name}_color_class_{class_id}.jpg")
        cv2.imwrite(class_path, class_result)

        grid_count = np.sum(labels_all == class_id)
        pixel_count = np.sum(color_class_map == class_id + 1)
        center = centers[class_id]
        hex_color = f"#{center[0]:02X}{center[1]:02X}{center[2]:02X}"
        percentage = pixel_count / (h*w) * 100

        if class_id < best_k:
            print(f"  类别{class_id}(色相{hue_centers[class_id]}°): RGB{tuple(center)} ({hex_color})")
        elif class_id == best_k:
            print(f"  类别{class_id}(深灰/黑色): RGB{tuple(center)} ({hex_color})")
        else:
            print(f"  类别{class_id}(浅灰色): RGB{tuple(center)} ({hex_color})")
        print(f"    网格数={grid_count}, 像素数={pixel_count} ({percentage:.2f}%) -> {class_path}")

    print("\n✓ 完成")

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("用法: python color_cluster.py <图像路径>")
        sys.exit(1)

    classify_colors(sys.argv[1])

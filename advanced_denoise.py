#!/usr/bin/env python3
"""
高级图像去噪 - 专门用于历史文书扫描件
通过多级形态学操作和连通组件特征分析，精确区分文字和污渍
"""
import cv2
import numpy as np
import sys
import os

# ==================== 可调参数 ====================
# 连通组件过滤参数
MIN_AREA = 30              # 最小面积（过小的是噪点）
MAX_AREA = 5000            # 最大面积（过大的是污渍块）
MIN_ASPECT_RATIO = 0.2     # 最小宽高比（过小是竖线）
MAX_ASPECT_RATIO = 5.0     # 最大宽高比（过大是横线）
MIN_COMPACTNESS = 0.15     # 最小紧凑度（area/bbox_area，过低是稀疏污渍）
MAX_CIRCULARITY = 0.85     # 最大圆形度（过高是圆形污点）

# 形态学操作参数
OPEN_KERNEL_SIZE = 2       # 开运算核大小（去噪）
CLOSE_KERNEL_SIZE = 2      # 闭运算核大小（连接笔画）

# 二值化参数
ADAPTIVE_BLOCK_SIZE = 15   # 自适应阈值块大小
ADAPTIVE_C = 8             # 自适应阈值常数
# ==================================================


def advanced_denoise(image_path: str):
    """
    高级去噪主函数
    """
    print("=" * 70)
    print("高级图像去噪 - 历史文书专用")
    print("=" * 70)

    # 1. 读取图片
    print("\n[1/7] 读取图片...")
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"无法读取图像: {image_path}")

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    print(f"图片尺寸: {w}x{h}")

    # 2. 预处理增强
    print("\n[2/7] 图像增强...")
    # 形态学闭运算 - 填补小孔洞
    kernel_close_pre = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    closed = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel_close_pre)

    # 双边滤波 - 保边去噪
    denoised = cv2.bilateralFilter(closed, 9, 75, 75)

    # CLAHE 对比度增强
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(denoised)

    # 3. 自适应二值化
    print("\n[3/7] 自适应二值化...")
    binary = cv2.adaptiveThreshold(
        enhanced, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        blockSize=ADAPTIVE_BLOCK_SIZE,
        C=ADAPTIVE_C
    )

    original_pixels = np.sum(binary > 0)
    print(f"二值化前景像素: {original_pixels:,}")

    # 4. 形态学开运算 - 去除小噪点
    print("\n[4/7] 形态学去噪...")
    kernel_open = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (OPEN_KERNEL_SIZE, OPEN_KERNEL_SIZE)
    )
    opened = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel_open, iterations=1)

    # 5. 连通组件分析 + 特征过滤
    print("\n[5/7] 连通组件分析...")
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        opened, connectivity=8
    )

    # 创建干净图像
    clean = np.zeros_like(binary)

    # 统计计数器
    removed_small = 0
    removed_large = 0
    removed_thin = 0
    removed_sparse = 0
    removed_circular = 0
    kept_count = 0

    # 遍历每个组件（跳过背景）
    for i in range(1, num_labels):
        x = stats[i, cv2.CC_STAT_LEFT]
        y = stats[i, cv2.CC_STAT_TOP]
        width = stats[i, cv2.CC_STAT_WIDTH]
        height = stats[i, cv2.CC_STAT_HEIGHT]
        area = stats[i, cv2.CC_STAT_AREA]

        # 过滤条件1: 面积过小（噪点）
        if area < MIN_AREA:
            removed_small += 1
            continue

        # 过滤条件2: 面积过大（污渍块）
        if area > MAX_AREA:
            removed_large += 1
            continue

        # 计算宽高比
        aspect_ratio = max(width, height) / max(min(width, height), 1)

        # 过滤条件3: 宽高比异常（细长线条）
        if aspect_ratio < MIN_ASPECT_RATIO or aspect_ratio > MAX_ASPECT_RATIO:
            removed_thin += 1
            continue

        # 计算紧凑度（实际面积 / 外接矩形面积）
        bbox_area = width * height
        compactness = area / bbox_area if bbox_area > 0 else 0

        # 过滤条件4: 紧凑度过低（稀疏污渍）
        if compactness < MIN_COMPACTNESS:
            removed_sparse += 1
            continue

        # 计算圆形度（用于过滤圆形污点）
        # 提取该组件的轮廓
        component_mask = (labels == i).astype(np.uint8) * 255
        contours, _ = cv2.findContours(
            component_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        if contours:
            contour = contours[0]
            perimeter = cv2.arcLength(contour, True)
            # 圆形度: 4π*面积/周长²，圆形=1，不规则<1
            circularity = (4 * np.pi * area) / (perimeter * perimeter) if perimeter > 0 else 0

            # 过滤条件5: 圆形度过高（圆形污点）
            if circularity > MAX_CIRCULARITY:
                removed_circular += 1
                continue

        # 保留该区域
        clean[labels == i] = 255
        kept_count += 1

    print(f"\n过滤统计:")
    print(f"  去除过小噪点: {removed_small} 个")
    print(f"  去除过大污渍块: {removed_large} 个")
    print(f"  去除细长线条: {removed_thin} 个")
    print(f"  去除稀疏污渍: {removed_sparse} 个")
    print(f"  去除圆形污点: {removed_circular} 个")
    print(f"  保留文字区域: {kept_count} 个")

    # 6. 形态学闭运算 - 连接断裂笔画
    print("\n[6/7] 连接断裂笔画...")
    kernel_close = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (CLOSE_KERNEL_SIZE, CLOSE_KERNEL_SIZE)
    )
    final = cv2.morphologyEx(clean, cv2.MORPH_CLOSE, kernel_close, iterations=1)

    final_pixels = np.sum(final > 0)
    removed_pixels = original_pixels - final_pixels
    removal_ratio = (removed_pixels / original_pixels * 100) if original_pixels > 0 else 0

    print(f"\n总体统计:")
    print(f"  原始前景像素: {original_pixels:,}")
    print(f"  最终文字像素: {final_pixels:,}")
    print(f"  去除像素数: {removed_pixels:,}")
    print(f"  去除比例: {removal_ratio:.2f}%")

    # 7. 保存结果
    print("\n[7/7] 保存结果...")
    # 固定输出到 results 目录
    output_dir = "results"
    os.makedirs(output_dir, exist_ok=True)
    base_name = os.path.splitext(os.path.basename(image_path))[0]

    # 保存去噪后的图片
    cv2.imwrite(os.path.join(output_dir, f"{base_name}_denoised.jpg"), final)

    # 保存对比图（红色=删除，绿色=保留）
    comparison = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    comparison[binary > 0] = [100, 100, 100]  # 灰色显示原始前景
    comparison[final > 0] = [0, 255, 0]       # 绿色显示保留的文字
    removed_mask = cv2.subtract(binary, final)
    comparison[removed_mask > 0] = [0, 0, 255]  # 红色显示删除的噪点
    cv2.imwrite(os.path.join(output_dir, f"{base_name}_denoise_comparison.jpg"), comparison)

    # 保存标注图（绿框标注保留的区域）
    annotated = img.copy()
    for i in range(1, num_labels):
        if np.any(labels == i):
            # 检查该区域是否被保留
            if np.sum(clean[labels == i]) > 0:
                x = stats[i, cv2.CC_STAT_LEFT]
                y = stats[i, cv2.CC_STAT_TOP]
                width = stats[i, cv2.CC_STAT_WIDTH]
                height = stats[i, cv2.CC_STAT_HEIGHT]
                cv2.rectangle(annotated, (x, y), (x+width, y+height), (0, 255, 0), 1)
    cv2.imwrite(os.path.join(output_dir, f"{base_name}_denoise_annotated.jpg"), annotated)

    print("\n" + "=" * 70)
    print("去噪完成!")
    print(f"  去噪后图片: {output_dir}/{base_name}_denoised.jpg")
    print(f"  对比图（红=删除，绿=保留）: {output_dir}/{base_name}_denoise_comparison.jpg")
    print(f"  标注图: {output_dir}/{base_name}_denoise_annotated.jpg")
    print("=" * 70)

    return final


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("用法: python advanced_denoise.py <图片路径>")
        print("\n示例:")
        print("  python advanced_denoise.py 分离目标.jpg")
        print("\n可调参数（修改脚本顶部）:")
        print(f"  MIN_AREA = {MIN_AREA}")
        print(f"  MAX_AREA = {MAX_AREA}")
        print(f"  MIN_ASPECT_RATIO = {MIN_ASPECT_RATIO}")
        print(f"  MAX_ASPECT_RATIO = {MAX_ASPECT_RATIO}")
        print(f"  MIN_COMPACTNESS = {MIN_COMPACTNESS}")
        print(f"  MAX_CIRCULARITY = {MAX_CIRCULARITY}")
        sys.exit(1)

    image_path = sys.argv[1]
    advanced_denoise(image_path)

#!/usr/bin/env python3
"""
对预处理后的图片进行手写/印刷体分类
基于位置、笔画特征、密度等多维度判断
"""
import cv2
import numpy as np
from utils.text_classifier import TextClassifier
import sys
import os


def classify_by_features_and_position(image_path: str):
    """
    结合特征和位置信息进行分类
    """
    print("=" * 60)
    print("预处理图片的手写/印刷体分类")
    print("=" * 60)

    classifier = TextClassifier(debug=True)

    # 预处理图片
    img, gray, binary = classifier.preprocess(image_path)
    h, w = gray.shape

    # 提取文字区域
    regions = classifier.extract_text_regions(binary)
    print(f"\n检测到 {len(regions)} 个文字区域")

    # 分析区域位置分布，判断表格区域
    # 假设：左侧规整排列的是印刷体表格，右侧分散的是手写批注
    left_regions = []
    right_regions = []

    for region in regions:
        x, y, rw, rh = region['bbox']
        center_x = x + rw // 2

        # 简单按X坐标分区（中线为分界）
        if center_x < w * 0.5:
            left_regions.append(region)
        else:
            right_regions.append(region)

    print(f"左侧区域: {len(left_regions)} 个")
    print(f"右侧区域: {len(right_regions)} 个")

    # 分类结果
    handwritten_count = 0
    printed_count = 0

    classified_regions = []

    # 左侧区域 - 优先判断为印刷体，除非特征明显是手写
    print("\n分析左侧区域...")
    for i, region in enumerate(left_regions):
        x, y, rw, rh = region['bbox']
        roi = binary[y:y+rh, x:x+rw]

        # 简化特征提取（不使用骨架化）
        h_roi, w_roi = roi.shape
        density = np.sum(roi > 0) / (h_roi * w_roi) if (h_roi * w_roi) > 0 else 0

        # 计算边缘复杂度
        contours, _ = cv2.findContours(roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            perimeter = cv2.arcLength(contours[0], True)
            area_c = cv2.contourArea(contours[0])
            roughness = perimeter / np.sqrt(area_c) if area_c > 0 else 0
        else:
            roughness = 0

        # 左侧：粗糙度高(>15)或密度低(<0.3)认为是手写
        if roughness > 15 or density < 0.3:
            label = 'handwritten'
        else:
            label = 'printed'

        if label == 'handwritten':
            handwritten_count += 1
        else:
            printed_count += 1

        classified_regions.append({
            **region,
            'label': label
        })

        if i < 5:  # 显示前5个用于调试
            print(f"  区域 {i}: {label}, rough={roughness:.1f}, dens={density:.2f}")

    # 右侧区域 - 优先判断为手写，除非特征明显是印刷
    print("\n分析右侧区域...")
    for i, region in enumerate(right_regions):
        x, y, rw, rh = region['bbox']
        roi = binary[y:y+rh, x:x+rw]

        # 简化特征提取
        h_roi, w_roi = roi.shape
        density = np.sum(roi > 0) / (h_roi * w_roi) if (h_roi * w_roi) > 0 else 0

        contours, _ = cv2.findContours(roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            perimeter = cv2.arcLength(contours[0], True)
            area_c = cv2.contourArea(contours[0])
            roughness = perimeter / np.sqrt(area_c) if area_c > 0 else 0
        else:
            roughness = 0

        # 右侧：粗糙度低(<12)且密度高(>0.4)才认为是印刷
        if roughness < 12 and density > 0.4:
            label = 'printed'
        else:
            label = 'handwritten'

        if label == 'handwritten':
            handwritten_count += 1
        else:
            printed_count += 1

        classified_regions.append({
            **region,
            'label': label
        })

        if i < 5:  # 显示前5个用于调试
            print(f"  区域 {i}: {label}, rough={roughness:.1f}, dens={density:.2f}")

    print("\n" + "=" * 60)
    print(f"分类结果:")
    print(f"  手写体区域: {handwritten_count}")
    print(f"  印刷体区域: {printed_count}")
    print(f"  总区域数: {len(classified_regions)}")
    print("=" * 60)

    # 保存结果
    output_dir = os.path.dirname(image_path)
    base_name = os.path.splitext(os.path.basename(image_path))[0]

    # 创建分类图
    handwritten_img = np.zeros_like(gray)
    printed_img = np.zeros_like(gray)
    annotated_img = img.copy()

    for region in classified_regions:
        x, y, rw, rh = region['bbox']
        mask = binary[y:y+rh, x:x+rw]

        if region['label'] == 'handwritten':
            # 红色标注手写
            handwritten_img[y:y+rh, x:x+rw] = mask
            cv2.rectangle(annotated_img, (x, y), (x+rw, y+rh), (0, 0, 255), 2)
        else:
            # 绿色标注印刷
            printed_img[y:y+rh, x:x+rw] = mask
            cv2.rectangle(annotated_img, (x, y), (x+rw, y+rh), (0, 255, 0), 2)

    # 保存结果
    cv2.imwrite(os.path.join(output_dir, f"{base_name}_handwritten.jpg"), handwritten_img)
    cv2.imwrite(os.path.join(output_dir, f"{base_name}_printed.jpg"), printed_img)
    cv2.imwrite(os.path.join(output_dir, f"{base_name}_classified.jpg"), annotated_img)

    print(f"\n结果已保存到:")
    print(f"  手写体: {output_dir}/{base_name}_handwritten.jpg")
    print(f"  印刷体: {output_dir}/{base_name}_printed.jpg")
    print(f"  标注图: {output_dir}/{base_name}_classified.jpg")


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("用法: python classify_preprocessed.py <预处理后的图片>")
        sys.exit(1)

    image_path = sys.argv[1]
    classify_by_features_and_position(image_path)

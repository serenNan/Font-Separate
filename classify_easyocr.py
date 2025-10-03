#!/usr/bin/env python3
"""
使用 EasyOCR 进行文字检测和手写/印刷体分类
EasyOCR 兼容性好，不会出现 SIGILL 错误
"""
import cv2
import numpy as np
import easyocr
import sys
import os


def classify_with_easyocr(image_path: str):
    """
    使用 EasyOCR 检测文字并根据置信度分类
    """
    print("=" * 70)
    print("EasyOCR 手写/印刷体分类")
    print("=" * 70)

    # 1. 初始化 EasyOCR
    print("\n[1/4] 初始化 EasyOCR（中文+英文）...")
    reader = easyocr.Reader(['ch_sim', 'en'], gpu=False, verbose=False)
    print("✓ EasyOCR 初始化成功")

    # 2. 读取图片
    print("\n[2/4] 读取并检测文字...")
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"无法读取图像: {image_path}")

    h, w = img.shape[:2]
    print(f"图片尺寸: {w}x{h}")

    # 3. 检测文字
    print("正在检测文字（可能需要几分钟）...")
    results = reader.readtext(image_path)

    print(f"✓ 检测到 {len(results)} 个文字区域")

    # 4. 根据位置和排列规整度分类
    print("\n[3/4] 分析文字特征...")

    # 计算所有文字框的中心点
    centers = []
    for bbox, text, conf in results:
        points = np.array(bbox)
        center_x = np.mean(points[:, 0])
        center_y = np.mean(points[:, 1])
        centers.append((center_x, center_y))

    centers = np.array(centers)

    # 按位置分区：左侧（x < w*0.45）倾向印刷，右侧倾向手写
    split_x = w * 0.45

    handwritten_boxes = []
    printed_boxes = []

    for i, (bbox, text, conf) in enumerate(results):
        center_x, center_y = centers[i]

        # 计算文字框的宽高比
        points = np.array(bbox)
        box_w = np.max(points[:, 0]) - np.min(points[:, 0])
        box_h = np.max(points[:, 1]) - np.min(points[:, 1])
        aspect_ratio = box_w / max(box_h, 1)

        # 分类逻辑（改进版）：
        # 1. 左侧区域 → 大概率是印刷体表格
        # 2. 右侧区域 → 大概率是手写批注
        # 3. 宽高比极端（>5 或 <0.2）→ 可能是表格线残留

        if center_x < split_x:
            # 左侧：默认印刷体，除非宽高比异常
            if aspect_ratio > 5 or aspect_ratio < 0.2:
                continue  # 跳过异常框
            printed_boxes.append(bbox)
        else:
            # 右侧：默认手写体
            if aspect_ratio > 5 or aspect_ratio < 0.2:
                continue  # 跳过异常框
            handwritten_boxes.append(bbox)

    print(f"  印刷体区域（左侧）: {len(printed_boxes)} 个")
    print(f"  手写体区域（右侧）: {len(handwritten_boxes)} 个")

    # 显示部分结果用于调试
    print(f"\n分界线位置: x = {split_x:.0f}")
    print("部分检测结果（前 5 个）:")
    for i, (bbox, text, conf) in enumerate(results[:5]):
        center_x = centers[i][0]
        label = "印刷（左）" if center_x < split_x else "手写（右）"
        print(f"  {i+1}. [{label}] x={center_x:.0f}, 置信度={conf:.3f}, 文本='{text[:10]}...'")

    # 5. 生成结果图
    print("\n[4/4] 生成结果...")

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)

    handwritten_mask = np.zeros_like(binary)
    printed_mask = np.zeros_like(binary)
    annotated = img.copy()

    # 绘制手写体（红色）
    for bbox in handwritten_boxes:
        points = np.array(bbox, dtype=np.int32)
        cv2.fillPoly(handwritten_mask, [points], 255)
        cv2.polylines(annotated, [points], True, (0, 0, 255), 2)

    # 绘制印刷体（绿色）
    for bbox in printed_boxes:
        points = np.array(bbox, dtype=np.int32)
        cv2.fillPoly(printed_mask, [points], 255)
        cv2.polylines(annotated, [points], True, (0, 255, 0), 2)

    # 应用蒙版到二值图
    handwritten_result = cv2.bitwise_and(binary, handwritten_mask)
    printed_result = cv2.bitwise_and(binary, printed_mask)

    # 统计
    handwritten_pixels = np.sum(handwritten_mask > 0)
    printed_pixels = np.sum(printed_mask > 0)
    total_pixels = handwritten_pixels + printed_pixels

    if total_pixels > 0:
        print(f"\n像素统计:")
        print(f"  手写体像素: {handwritten_pixels:,} ({handwritten_pixels/total_pixels*100:.1f}%)")
        print(f"  印刷体像素: {printed_pixels:,} ({printed_pixels/total_pixels*100:.1f}%)")

    # 保存结果
    output_dir = "results"
    os.makedirs(output_dir, exist_ok=True)
    base_name = os.path.splitext(os.path.basename(image_path))[0]

    cv2.imwrite(os.path.join(output_dir, f"{base_name}_easyocr_handwritten.jpg"), handwritten_result)
    cv2.imwrite(os.path.join(output_dir, f"{base_name}_easyocr_printed.jpg"), printed_result)
    cv2.imwrite(os.path.join(output_dir, f"{base_name}_easyocr_annotated.jpg"), annotated)

    print("\n" + "=" * 70)
    print("分类完成!")
    print(f"  手写体: {output_dir}/{base_name}_easyocr_handwritten.jpg")
    print(f"  印刷体: {output_dir}/{base_name}_easyocr_printed.jpg")
    print(f"  标注图: {output_dir}/{base_name}_easyocr_annotated.jpg")
    print("=" * 70)


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("用法: python classify_easyocr.py <图片路径>")
        print("\n示例:")
        print("  python classify_easyocr.py results/分离目标_denoised.jpg")
        print("\n说明:")
        print("  - EasyOCR 兼容性好，不会出现 CPU 指令集错误")
        print("  - 根据 OCR 置信度分类：低置信度 = 手写，高置信度 = 印刷")
        print("  - 首次运行会下载模型（约 100MB）")
        sys.exit(1)

    image_path = sys.argv[1]
    classify_with_easyocr(image_path)

#!/usr/bin/env python3
"""
使用 PaddleOCR 检测文字区域，只保留真正的文字
去除噪点、线条、污渍等非文字内容
"""
import cv2
import numpy as np
from paddleocr import PaddleOCR
import sys
import os


def detect_text_with_paddleocr(image_path: str):
    """
    使用 PaddleOCR 检测文字区域
    """
    print("=" * 70)
    print("PaddleOCR 文字检测")
    print("=" * 70)

    # 1. 初始化 PaddleOCR (只使用检测模块)
    print("\n[1/4] 初始化 PaddleOCR...")
    ocr = PaddleOCR(
        use_angle_cls=False,  # 不使用方向分类器
        lang='ch',  # 中文
        det_model_dir=None,  # 使用默认检测模型
        rec=False,  # 不进行识别，只检测
        show_log=False
    )

    # 2. 读取图片
    print("\n[2/4] 读取图片...")
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"无法读取图像: {image_path}")

    h, w = img.shape[:2]
    print(f"图片尺寸: {w}x{h}")

    # 3. 检测文字区域
    print("\n[3/4] 检测文字区域...")
    result = ocr.ocr(image_path, cls=False, rec=False)

    if result is None or len(result) == 0 or result[0] is None:
        print("未检测到文字区域！")
        return

    text_boxes = result[0]
    print(f"检测到 {len(text_boxes)} 个文字区域")

    # 4. 生成文字蒙版
    print("\n[4/4] 生成文字蒙版...")
    text_mask = np.zeros((h, w), dtype=np.uint8)
    annotated_img = img.copy()

    for box in text_boxes:
        # box 是四个点的坐标 [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
        points = np.array(box, dtype=np.int32)

        # 填充文字区域
        cv2.fillPoly(text_mask, [points], 255)

        # 标注（绿色边框）
        cv2.polylines(annotated_img, [points], True, (0, 255, 0), 2)

    # 计算统计信息
    original_pixels = np.sum(text_mask > 0)
    total_pixels = h * w
    text_ratio = (original_pixels / total_pixels) * 100

    print(f"\n统计信息:")
    print(f"  文字像素: {original_pixels:,}")
    print(f"  图片总像素: {total_pixels:,}")
    print(f"  文字占比: {text_ratio:.2f}%")

    # 5. 保存结果
    output_dir = os.path.dirname(image_path)
    base_name = os.path.splitext(os.path.basename(image_path))[0]

    cv2.imwrite(os.path.join(output_dir, f"{base_name}_paddleocr_text.jpg"), text_mask)
    cv2.imwrite(os.path.join(output_dir, f"{base_name}_paddleocr_annotated.jpg"), annotated_img)

    # 应用蒙版到原图
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)
    text_only = cv2.bitwise_and(binary, text_mask)
    cv2.imwrite(os.path.join(output_dir, f"{base_name}_paddleocr_clean.jpg"), text_only)

    print("\n" + "=" * 70)
    print("检测完成!")
    print(f"  文字蒙版: {output_dir}/{base_name}_paddleocr_text.jpg")
    print(f"  干净文字: {output_dir}/{base_name}_paddleocr_clean.jpg")
    print(f"  标注图: {output_dir}/{base_name}_paddleocr_annotated.jpg")
    print("=" * 70)


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("用法: python detect_text_paddleocr.py <图片路径>")
        print("\n示例:")
        print("  python detect_text_paddleocr.py 分离目标.jpg")
        sys.exit(1)

    image_path = sys.argv[1]
    detect_text_with_paddleocr(image_path)

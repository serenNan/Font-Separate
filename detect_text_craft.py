#!/usr/bin/env python3
"""
使用 CRAFT 检测文字区域，只保留真正的文字
去除噪点、线条、污渍等非文字内容
"""
import cv2
import numpy as np
from craft_text_detector import Craft
import sys
import os


def detect_text_with_craft(image_path: str):
    """
    使用 CRAFT 检测文字区域
    """
    print("=" * 70)
    print("CRAFT 文字检测")
    print("=" * 70)

    # 1. 初始化 CRAFT
    print("\n[1/4] 初始化 CRAFT 模型...")
    craft = Craft(
        output_dir=None,
        crop_type="poly",
        cuda=False,  # 使用 CPU
        long_size=1280
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
    # CRAFT 返回预测结果
    prediction_result = craft.detect_text(image_path)

    # 提取检测到的文字框
    boxes = prediction_result["boxes"]
    print(f"检测到 {len(boxes)} 个文字区域")

    # 4. 生成文字蒙版
    print("\n[4/4] 生成文字蒙版...")
    text_mask = np.zeros((h, w), dtype=np.uint8)
    annotated_img = img.copy()

    for box in boxes:
        # box 是四个点的坐标 [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
        points = np.array(box, dtype=np.int32).reshape((-1, 1, 2))

        # 填充文字区域
        cv2.fillPoly(text_mask, [points], 255)

        # 标注（绿色边框）
        cv2.polylines(annotated_img, [points], True, (0, 255, 0), 2)

    # 计算统计信息
    text_pixels = np.sum(text_mask > 0)
    total_pixels = h * w
    text_ratio = (text_pixels / total_pixels) * 100

    print(f"\n统计信息:")
    print(f"  文字像素: {text_pixels:,}")
    print(f"  图片总像素: {total_pixels:,}")
    print(f"  文字占比: {text_ratio:.2f}%")

    # 5. 保存结果
    output_dir = os.path.dirname(image_path) if os.path.dirname(image_path) else "."
    base_name = os.path.splitext(os.path.basename(image_path))[0]

    cv2.imwrite(os.path.join(output_dir, f"{base_name}_craft_mask.jpg"), text_mask)
    cv2.imwrite(os.path.join(output_dir, f"{base_name}_craft_annotated.jpg"), annotated_img)

    # 应用蒙版到原图（如果原图是二值图）
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)
    text_only = cv2.bitwise_and(binary, text_mask)
    cv2.imwrite(os.path.join(output_dir, f"{base_name}_craft_clean.jpg"), text_only)

    print("\n" + "=" * 70)
    print("检测完成!")
    print(f"  文字蒙版: {output_dir}/{base_name}_craft_mask.jpg")
    print(f"  干净文字: {output_dir}/{base_name}_craft_clean.jpg")
    print(f"  标注图: {output_dir}/{base_name}_craft_annotated.jpg")
    print("=" * 70)

    # 卸载模型释放内存
    craft.unload_craftnet_model()
    craft.unload_refinenet_model()


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("用法: python detect_text_craft.py <图片路径>")
        print("\n示例:")
        print("  python detect_text_craft.py 分离目标.jpg")
        print("  python detect_text_craft.py results/分离目标_6_text_only.jpg")
        sys.exit(1)

    image_path = sys.argv[1]
    detect_text_with_craft(image_path)

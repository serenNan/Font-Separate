#!/usr/bin/env python3
"""
使用 YOLOv8 预训练模型进行手写/印刷体分类
模型: armvectores/yolov8n_handwritten_text_detection
"""
import cv2
import numpy as np
from ultralytics import YOLO
from huggingface_hub import hf_hub_download
import sys
import os


def classify_with_yolo(image_path: str):
    """
    使用 YOLOv8 模型检测手写文本区域
    """
    print("=" * 70)
    print("YOLOv8 手写/印刷体分类")
    print("=" * 70)

    # 1. 下载模型
    print("\n[1/4] 下载 YOLOv8 模型...")
    model_path = hf_hub_download(
        repo_id="armvectores/yolov8n_handwritten_text_detection",
        filename="best.pt"
    )
    print(f"模型已下载: {model_path}")

    # 2. 加载模型
    print("\n[2/4] 加载模型...")
    model = YOLO(model_path)

    # 3. 读取图片
    print("\n[3/4] 处理图片...")
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"无法读取图像: {image_path}")

    h, w = img.shape[:2]
    print(f"图片尺寸: {w}x{h}")

    # 4. 检测手写文本区域
    print("\n[4/4] 检测手写文本...")
    results = model.predict(source=image_path, conf=0.3, verbose=False)

    # 解析结果
    handwritten_regions = []
    for result in results:
        boxes = result.boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            conf = float(box.conf[0])
            handwritten_regions.append({
                'bbox': (int(x1), int(y1), int(x2-x1), int(y2-y1)),
                'confidence': conf
            })

    print(f"检测到 {len(handwritten_regions)} 个手写文本区域")

    # 5. 生成分类结果
    output_dir = os.path.dirname(image_path)
    base_name = os.path.splitext(os.path.basename(image_path))[0]

    # 创建手写/印刷蒙版
    handwritten_mask = np.zeros((h, w), dtype=np.uint8)
    annotated_img = img.copy()

    for region in handwritten_regions:
        x, y, rw, rh = region['bbox']
        conf = region['confidence']

        # 填充手写区域
        handwritten_mask[y:y+rh, x:x+rw] = 255

        # 标注（红色框）
        cv2.rectangle(annotated_img, (x, y), (x+rw, y+rh), (0, 0, 255), 2)
        cv2.putText(annotated_img, f"{conf:.2f}",
                    (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

    # 印刷体蒙版 = 全图 - 手写蒙版
    # 读取原始二值图
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)

    printed_mask = cv2.bitwise_and(binary, cv2.bitwise_not(handwritten_mask))

    # 标注印刷体区域（绿色）
    printed_contours, _ = cv2.findContours(
        printed_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    cv2.drawContours(annotated_img, printed_contours, -1, (0, 255, 0), 2)

    # 保存结果
    cv2.imwrite(os.path.join(output_dir, f"{base_name}_yolo_handwritten.jpg"),
                handwritten_mask)
    cv2.imwrite(os.path.join(output_dir, f"{base_name}_yolo_printed.jpg"),
                printed_mask)
    cv2.imwrite(os.path.join(output_dir, f"{base_name}_yolo_annotated.jpg"),
                annotated_img)

    print("\n" + "=" * 70)
    print("分类完成!")
    print(f"  手写文本区域: {len(handwritten_regions)} 个")
    print(f"  手写文本蒙版: {output_dir}/{base_name}_yolo_handwritten.jpg")
    print(f"  印刷文本蒙版: {output_dir}/{base_name}_yolo_printed.jpg")
    print(f"  标注图: {output_dir}/{base_name}_yolo_annotated.jpg")
    print("=" * 70)


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("用法: python classify_yolo.py <图片路径>")
        sys.exit(1)

    image_path = sys.argv[1]
    classify_with_yolo(image_path)

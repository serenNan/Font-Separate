"""
手写体与印刷体分类器
基于 EasyOCR 检测 + 位置分类策略
"""
import cv2
import numpy as np
import easyocr
import os
from typing import Dict, Tuple


class TextClassifier:
    """文字分类器: 使用 EasyOCR + 位置分类区分手写体和印刷体"""

    def __init__(self, debug=False, split_ratio=0.45):
        """
        初始化分类器
        Args:
            debug: 是否输出调试信息
            split_ratio: 位置分界线比例 (0-1)，左侧=印刷，右侧=手写。默认 0.45
        """
        self.debug = debug
        self.reader = None  # 延迟初始化
        self.split_ratio = split_ratio  # 位置分界线

    def _init_reader(self):
        """延迟初始化 EasyOCR (避免启动时加载)"""
        if self.reader is None:
            if self.debug:
                print("初始化 EasyOCR（中文+英文）...")
            self.reader = easyocr.Reader(['ch_sim', 'en'], gpu=False, verbose=False)
            if self.debug:
                print("✓ EasyOCR 初始化完成")

    def classify_and_separate(self, image_path: str, output_dir: str,
                              table_mask: np.ndarray = None) -> Dict:
        """
        分类并分离手写体和印刷体

        Args:
            image_path: 图像路径
            output_dir: 输出目录
            table_mask: 表格掩码 (可选, 避免重复处理表格区域)

        Returns:
            结果字典 {
                'handwritten_count': int,
                'printed_count': int,
                'handwritten_path': str,
                'printed_path': str,
                'annotated_path': str
            }
        """
        # 1. 初始化 EasyOCR
        self._init_reader()

        # 2. 读取图像
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"无法读取图像: {image_path}")

        h, w = img.shape[:2]
        if self.debug:
            print(f"图片尺寸: {w}x{h}")

        # 3. 检测文字
        if self.debug:
            print("正在检测文字...")
        results = self.reader.readtext(image_path)

        if self.debug:
            print(f"✓ 检测到 {len(results)} 个文字区域")

        # 4. 基于位置分类 (简单有效)
        split_x = w * self.split_ratio  # 分界线位置

        handwritten_boxes = []
        printed_boxes = []

        for i, (bbox, text, conf) in enumerate(results):
            points = np.array(bbox)
            center_x = np.mean(points[:, 0])

            # 计算宽高比过滤异常框
            box_w = np.max(points[:, 0]) - np.min(points[:, 0])
            box_h = np.max(points[:, 1]) - np.min(points[:, 1])
            aspect_ratio = box_w / max(box_h, 1)

            # 跳过异常宽高比 (表格线残留)
            if aspect_ratio > 5 or aspect_ratio < 0.2:
                continue

            # 位置分类
            if center_x < split_x:
                printed_boxes.append(bbox)
            else:
                handwritten_boxes.append(bbox)

        if self.debug:
            print(f"  印刷体区域（规整）: {len(printed_boxes)} 个")
            print(f"  手写体区域（歪斜）: {len(handwritten_boxes)} 个")

        # 5. 生成结果图
        # 创建蒙版（用于提取区域）
        handwritten_mask = np.zeros((h, w), dtype=np.uint8)
        printed_mask = np.zeros((h, w), dtype=np.uint8)
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

        # 应用蒙版到**原图**（保留彩色）
        handwritten_result = cv2.bitwise_and(img, img, mask=handwritten_mask)
        printed_result = cv2.bitwise_and(img, img, mask=printed_mask)

        # 6. 保存结果
        os.makedirs(output_dir, exist_ok=True)
        base_name = os.path.splitext(os.path.basename(image_path))[0]

        handwritten_path = os.path.join(output_dir, f"{base_name}_handwritten.jpg")
        printed_path = os.path.join(output_dir, f"{base_name}_printed.jpg")
        annotated_path = os.path.join(output_dir, f"{base_name}_text_annotated.jpg")

        cv2.imwrite(handwritten_path, handwritten_result)
        cv2.imwrite(printed_path, printed_result)
        cv2.imwrite(annotated_path, annotated)

        return {
            'handwritten_count': len(handwritten_boxes),
            'printed_count': len(printed_boxes),
            'handwritten_path': handwritten_path,
            'printed_path': printed_path,
            'annotated_path': annotated_path
        }

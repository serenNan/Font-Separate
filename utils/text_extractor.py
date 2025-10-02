"""
文字提取预处理器
去除表格线、印章、污渍等非文字内容,只保留纯文字
"""
import cv2
import numpy as np
from typing import Tuple
import os


class TextExtractor:
    """文字提取器 - 去除非文字元素"""

    def __init__(self, debug=False):
        self.debug = debug

    def remove_lines(self, binary: np.ndarray) -> np.ndarray:
        """
        去除表格线(横线和竖线)
        Args:
            binary: 二值图
        Returns:
            去除线条后的二值图
        """
        # 检测横线 - 使用更小的核以检测更多线条
        horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (30, 1))
        detect_horizontal = cv2.morphologyEx(binary, cv2.MORPH_OPEN, horizontal_kernel, iterations=2)

        # 检测竖线 - 使用更小的核以检测更多线条
        vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 30))
        detect_vertical = cv2.morphologyEx(binary, cv2.MORPH_OPEN, vertical_kernel, iterations=2)

        # 合并所有线条
        lines_mask = cv2.bitwise_or(detect_horizontal, detect_vertical)

        # 膨胀一下，确保线条完全去除
        dilate_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        lines_mask = cv2.dilate(lines_mask, dilate_kernel, iterations=1)

        # 从原图中去除线条
        result = cv2.subtract(binary, lines_mask)

        if self.debug:
            h_count = np.sum(detect_horizontal > 0)
            v_count = np.sum(detect_vertical > 0)
            print(f"检测到横线像素: {h_count}, 竖线像素: {v_count}")

        return result

    def remove_stamps(self, binary: np.ndarray, original: np.ndarray) -> np.ndarray:
        """
        去除印章(红色区域)
        Args:
            binary: 二值图
            original: 原始彩色图
        Returns:
            去除印章后的二值图
        """
        # 转换到HSV空间检测红色
        hsv = cv2.cvtColor(original, cv2.COLOR_BGR2HSV)

        # 红色的HSV范围(两个区间,因为红色跨越0度)
        lower_red1 = np.array([0, 50, 50])
        upper_red1 = np.array([10, 255, 255])
        lower_red2 = np.array([170, 50, 50])
        upper_red2 = np.array([180, 255, 255])

        # 创建红色掩码
        mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
        mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
        red_mask = cv2.bitwise_or(mask1, mask2)

        # 形态学闭运算,填补印章内部的空洞
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_CLOSE, kernel, iterations=2)

        # 从二值图中去除红色区域
        result = cv2.subtract(binary, red_mask)

        if self.debug:
            stamp_pixels = np.sum(red_mask > 0)
            print(f"检测到印章像素: {stamp_pixels}")

        return result

    def remove_noise(self, binary: np.ndarray, min_size: int = 200,
                     max_aspect_ratio: float = 15.0,
                     border_margin: int = 10) -> np.ndarray:
        """
        去除小噪点、细长线条和边缘噪点，智能保留文字
        Args:
            binary: 二值图
            min_size: 最小保留区域面积
            max_aspect_ratio: 最大长宽比(过滤细长噪点)
            border_margin: 边缘边距(去除边缘噪点)
        Returns:
            去噪后的二值图
        """
        h, w = binary.shape

        # 连通组件分析
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            binary, connectivity=8
        )

        # 创建干净的图像
        clean = np.zeros_like(binary)

        noise_count = 0
        edge_noise_count = 0
        thin_noise_count = 0
        sparse_noise_count = 0

        for i in range(1, num_labels):
            x = stats[i, cv2.CC_STAT_LEFT]
            y = stats[i, cv2.CC_STAT_TOP]
            width = stats[i, cv2.CC_STAT_WIDTH]
            height = stats[i, cv2.CC_STAT_HEIGHT]
            area = stats[i, cv2.CC_STAT_AREA]

            # 过滤条件1: 长宽比过大(细长线条残留) - 优先检查
            aspect_ratio = max(width, height) / max(min(width, height), 1)
            if aspect_ratio > max_aspect_ratio:
                thin_noise_count += 1
                continue

            # 过滤条件2: 边缘噪点(紧贴图像边缘)
            if (x < border_margin or y < border_margin or
                x + width > w - border_margin or
                y + height > h - border_margin):
                edge_noise_count += 1
                continue

            # 过滤条件3: 面积太小 + 密度太低(稀疏噪点)
            if area < min_size:
                # 计算区域密度(实际像素/外接矩形面积)
                bbox_area = width * height
                density = area / bbox_area if bbox_area > 0 else 0

                # 如果密度很低(<0.3)，可能是噪点；密度高(>=0.3)可能是小字符
                if density < 0.3 or area < 40:
                    noise_count += 1
                    continue

            # 过滤条件4: 区域过于稀疏(像素分散)
            bbox_area = width * height
            if bbox_area > 0:
                density = area / bbox_area
                # 大区域但密度极低，可能是污渍
                if area > min_size and density < 0.1:
                    sparse_noise_count += 1
                    continue

            # 保留该区域
            clean[labels == i] = 255

        if self.debug:
            print(f"去除小噪点: {noise_count} 个")
            print(f"去除细长线条: {thin_noise_count} 个")
            print(f"去除边缘噪点: {edge_noise_count} 个")
            print(f"去除稀疏噪点: {sparse_noise_count} 个")
            print(f"总计去除: {noise_count + thin_noise_count + edge_noise_count + sparse_noise_count} 个")

        return clean

    def enhance_text(self, binary: np.ndarray) -> np.ndarray:
        """
        增强文字,修复断裂
        Args:
            binary: 二值图
        Returns:
            增强后的二值图
        """
        # 形态学闭运算,连接断裂的笔画
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
        enhanced = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=1)

        return enhanced

    def extract_text_only(self, image_path: str, output_dir: str = None) -> Tuple[np.ndarray, dict]:
        """
        完整的文字提取流程
        Args:
            image_path: 输入图像路径
            output_dir: 输出目录(可选)
        Returns:
            (纯文字二值图, 统计信息)
        """
        if self.debug:
            print("=" * 60)
            print("开始文字提取预处理...")
            print("=" * 60)

        # 1. 读取图像
        original = cv2.imread(image_path)
        if original is None:
            raise ValueError(f"无法读取图像: {image_path}")

        gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)

        # 2. 图像增强
        if self.debug:
            print("\n[1/6] 图像增强...")

        # 形态学闭运算，先填补小孔洞
        kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        closed = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel_close)

        # 双边滤波去噪（参数加强）
        denoised = cv2.bilateralFilter(closed, 11, 80, 80)

        # 锐化
        kernel_sharpen = np.array([[-1,-1,-1], [-1, 9,-1], [-1,-1,-1]])
        sharpened = cv2.filter2D(denoised, -1, kernel_sharpen)

        # CLAHE对比度增强
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        enhanced = clahe.apply(sharpened)

        # 3. 自适应二值化
        if self.debug:
            print("[2/6] 二值化...")

        binary = cv2.adaptiveThreshold(
            enhanced, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            blockSize=21,  # 增大块大小，减少噪点
            C=10  # 增大常数，更保守的阈值
        )

        # 4. 去除表格线
        if self.debug:
            print("[3/6] 去除表格线...")

        no_lines = self.remove_lines(binary)

        # 5. 去除印章
        if self.debug:
            print("[4/6] 去除印章...")

        no_stamps = self.remove_stamps(no_lines, original)

        # 6. 去除噪点
        if self.debug:
            print("[5/6] 去除噪点...")

        no_noise = self.remove_noise(no_stamps, min_size=100, max_aspect_ratio=6.0, border_margin=10)

        # 7. 增强文字
        if self.debug:
            print("[6/6] 增强文字...")

        text_only = self.enhance_text(no_noise)

        # 统计信息
        stats = {
            'original_pixels': np.sum(binary > 0),
            'final_pixels': np.sum(text_only > 0),
            'removed_pixels': np.sum(binary > 0) - np.sum(text_only > 0),
            'removal_ratio': (np.sum(binary > 0) - np.sum(text_only > 0)) / np.sum(binary > 0) if np.sum(binary > 0) > 0 else 0
        }

        if self.debug:
            print("\n" + "=" * 60)
            print("预处理完成!")
            print(f"原始前景像素: {stats['original_pixels']}")
            print(f"最终文字像素: {stats['final_pixels']}")
            print(f"去除像素数: {stats['removed_pixels']}")
            print(f"去除比例: {stats['removal_ratio']*100:.2f}%")
            print("=" * 60)

        # 保存结果
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            base_name = os.path.splitext(os.path.basename(image_path))[0]

            # 保存各个阶段的结果
            cv2.imwrite(os.path.join(output_dir, f"{base_name}_1_enhanced.jpg"), enhanced)
            cv2.imwrite(os.path.join(output_dir, f"{base_name}_2_binary.jpg"), binary)
            cv2.imwrite(os.path.join(output_dir, f"{base_name}_3_no_lines.jpg"), no_lines)
            cv2.imwrite(os.path.join(output_dir, f"{base_name}_4_no_stamps.jpg"), no_stamps)
            cv2.imwrite(os.path.join(output_dir, f"{base_name}_5_no_noise.jpg"), no_noise)
            cv2.imwrite(os.path.join(output_dir, f"{base_name}_6_text_only.jpg"), text_only)

            # 保存纯文字叠加到原图
            text_overlay = original.copy()
            text_overlay[text_only > 0] = [0, 255, 0]  # 绿色标注文字
            cv2.imwrite(os.path.join(output_dir, f"{base_name}_text_overlay.jpg"), text_overlay)

            if self.debug:
                print(f"\n结果已保存到: {output_dir}")

        return text_only, stats

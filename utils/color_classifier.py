"""
颜色分类器 - 基于自适应颜色聚类
根据文档中实际出现的颜色动态分类，不预设固定颜色
"""
import cv2
import numpy as np
import easyocr
import os
from typing import Dict, List, Tuple, Optional
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


class ColorClassifier:
    """自适应颜色分类器：基于 K-Means 聚类区分不同颜色的文字"""

    def __init__(
        self,
        debug=False,
        n_clusters: Optional[int] = None,
        auto_k_range: Tuple[int, int] = (2, 6),
        color_space: str = 'lab',
        min_saturation: int = 10
    ):
        """
        初始化颜色分类器

        Args:
            debug: 是否输出调试信息
            n_clusters: 指定聚类数（None=自动检测最佳值）
            auto_k_range: 自动检测时的 K 值范围 (min, max)
            color_space: 颜色空间 'rgb' | 'hsv' | 'lab' (推荐lab)
            min_saturation: 最小饱和度阈值（过滤背景噪声）
        """
        self.debug = debug
        self.n_clusters = n_clusters
        self.auto_k_range = auto_k_range
        self.color_space = color_space.lower()
        self.min_saturation = min_saturation
        self.reader = None  # 延迟初始化 EasyOCR

    def _init_reader(self):
        """延迟初始化 EasyOCR"""
        if self.reader is None:
            if self.debug:
                print("初始化 EasyOCR（中文+英文）...")
            self.reader = easyocr.Reader(['ch_sim', 'en'], gpu=False, verbose=False)
            if self.debug:
                print("✓ EasyOCR 初始化完成")

    def extract_text_color(self, roi: np.ndarray) -> Optional[np.ndarray]:
        """
        从文字区域提取主色调（排除背景）

        Args:
            roi: 文字区域图像 (BGR)

        Returns:
            颜色向量 (3维) 或 None（如果提取失败）
        """
        if roi.size == 0:
            return None

        # 转灰度并二值化分离文字与背景
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        # 提取文字像素
        text_pixels = roi[mask > 0]

        if len(text_pixels) == 0:
            return None

        # 计算中位数颜色（比均值更鲁棒）
        median_color = np.median(text_pixels, axis=0).astype(np.uint8)

        # 转换到指定颜色空间
        color_bgr = median_color.reshape(1, 1, 3)

        if self.color_space == 'hsv':
            color_converted = cv2.cvtColor(color_bgr, cv2.COLOR_BGR2HSV)[0, 0]
        elif self.color_space == 'lab':
            color_converted = cv2.cvtColor(color_bgr, cv2.COLOR_BGR2Lab)[0, 0]
        else:  # rgb
            color_converted = cv2.cvtColor(color_bgr, cv2.COLOR_BGR2RGB)[0, 0]

        return color_converted

    def find_optimal_clusters(self, colors: np.ndarray) -> int:
        """
        使用轮廓系数法自动确定最佳聚类数

        Args:
            colors: 颜色向量数组 (N, 3)

        Returns:
            最佳聚类数 K
        """
        if len(colors) < self.auto_k_range[0]:
            return min(len(colors), self.auto_k_range[0])

        min_k, max_k = self.auto_k_range
        max_k = min(max_k, len(colors))  # K 不能超过样本数

        if self.debug:
            print(f"正在自动检测最佳聚类数（范围 {min_k}-{max_k}）...")

        scores = []
        for k in range(min_k, max_k + 1):
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            labels = kmeans.fit_predict(colors)
            score = silhouette_score(colors, labels)
            scores.append((k, score))
            if self.debug:
                print(f"  K={k}, 轮廓系数={score:.3f}")

        # 选择得分最高的 K
        best_k, best_score = max(scores, key=lambda x: x[1])

        if self.debug:
            print(f"✓ 最佳聚类数: K={best_k} (得分={best_score:.3f})")

        return best_k

    def classify_by_color(self, image_path: str, output_dir: str) -> Dict:
        """
        基于颜色分类并分离文字

        Args:
            image_path: 图像路径
            output_dir: 输出目录

        Returns:
            结果字典 {
                'n_clusters': int,
                'clusters': {
                    '0': {'count': int, 'color_rgb': [...], 'color_hsv': [...]},
                    ...
                },
                'annotated_path': str,
                'cluster_paths': [str, ...],
                'palette_path': str
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
            print(f"\n图片尺寸: {w}x{h}")

        # 3. 检测文字
        if self.debug:
            print("正在检测文字...")
        results = self.reader.readtext(image_path)

        if self.debug:
            print(f"✓ 检测到 {len(results)} 个文字区域")

        # 4. 提取颜色特征
        if self.debug:
            print("\n提取文字颜色特征...")

        colors = []
        valid_boxes = []

        for i, (bbox, text, conf) in enumerate(results):
            points = np.array(bbox, dtype=np.int32)

            # 计算宽高比过滤异常框（表格线残留）
            box_w = np.max(points[:, 0]) - np.min(points[:, 0])
            box_h = np.max(points[:, 1]) - np.min(points[:, 1])
            aspect_ratio = box_w / max(box_h, 1)

            if aspect_ratio > 5 or aspect_ratio < 0.2:
                continue

            # 提取 ROI
            x_min = max(0, np.min(points[:, 0]))
            y_min = max(0, np.min(points[:, 1]))
            x_max = min(w, np.max(points[:, 0]))
            y_max = min(h, np.max(points[:, 1]))

            roi = img[y_min:y_max, x_min:x_max]

            # 提取颜色
            color = self.extract_text_color(roi)
            if color is not None:
                colors.append(color)
                valid_boxes.append(bbox)

        if len(colors) == 0:
            raise ValueError("未检测到有效的文字颜色")

        colors = np.array(colors)

        if self.debug:
            print(f"✓ 提取到 {len(colors)} 个有效颜色特征")

        # 5. 颜色聚类
        if self.n_clusters is None:
            k = self.find_optimal_clusters(colors)
        else:
            k = min(self.n_clusters, len(colors))
            if self.debug:
                print(f"\n使用指定聚类数: K={k}")

        if self.debug:
            print(f"\n执行 K-Means 聚类 (K={k}, 颜色空间={self.color_space})...")

        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(colors)
        centers = kmeans.cluster_centers_

        if self.debug:
            print("✓ 聚类完成")

        # 6. 生成结果
        os.makedirs(output_dir, exist_ok=True)
        base_name = os.path.splitext(os.path.basename(image_path))[0]

        # 创建标注图和各类别掩码
        annotated = img.copy()
        cluster_masks = [np.zeros((h, w), dtype=np.uint8) for _ in range(k)]

        # 为每个聚类分配不同的边框颜色
        cluster_colors = self._generate_distinct_colors(k)

        cluster_counts = [0] * k

        for bbox, label in zip(valid_boxes, labels):
            points = np.array(bbox, dtype=np.int32)

            # 填充掩码
            cv2.fillPoly(cluster_masks[label], [points], 255)

            # 绘制边框
            cv2.polylines(annotated, [points], True, cluster_colors[label], 2)

            cluster_counts[label] += 1

        # 保存标注图
        annotated_path = os.path.join(output_dir, f"{base_name}_color_annotated.jpg")
        cv2.imwrite(annotated_path, annotated)

        # 保存各类别图像
        cluster_paths = []
        for i in range(k):
            cluster_result = cv2.bitwise_and(img, img, mask=cluster_masks[i])
            cluster_path = os.path.join(output_dir, f"{base_name}_cluster_{i}.jpg")
            cv2.imwrite(cluster_path, cluster_result)
            cluster_paths.append(cluster_path)

        # 7. 生成颜色统计
        cluster_info = {}
        for i in range(k):
            # 转换聚类中心颜色到 RGB 和 HSV
            center_color = centers[i].astype(np.uint8).reshape(1, 1, 3)

            if self.color_space == 'lab':
                center_bgr = cv2.cvtColor(center_color, cv2.COLOR_Lab2BGR)[0, 0]
            elif self.color_space == 'hsv':
                center_bgr = cv2.cvtColor(center_color, cv2.COLOR_HSV2BGR)[0, 0]
            else:  # rgb
                center_bgr = cv2.cvtColor(center_color, cv2.COLOR_RGB2BGR)[0, 0]

            center_rgb = cv2.cvtColor(center_bgr.reshape(1, 1, 3), cv2.COLOR_BGR2RGB)[0, 0]
            center_hsv = cv2.cvtColor(center_bgr.reshape(1, 1, 3), cv2.COLOR_BGR2HSV)[0, 0]

            cluster_info[str(i)] = {
                'count': int(cluster_counts[i]),
                'color_rgb': center_rgb.tolist(),
                'color_hsv': center_hsv.tolist(),
                'description': self._describe_color(center_hsv)
            }

        # 8. 生成颜色色板可视化
        palette_path = self._generate_color_palette(centers, cluster_counts, output_dir, base_name)

        if self.debug:
            print("\n=== 颜色分类结果 ===")
            for i in range(k):
                info = cluster_info[str(i)]
                print(f"类别 {i} ({info['description']}): {info['count']} 个区域")
                print(f"  RGB: {info['color_rgb']}")
                print(f"  HSV: {info['color_hsv']}")

        return {
            'n_clusters': k,
            'clusters': cluster_info,
            'annotated_path': annotated_path,
            'cluster_paths': cluster_paths,
            'palette_path': palette_path
        }

    def _generate_distinct_colors(self, n: int) -> List[Tuple[int, int, int]]:
        """生成 N 个视觉上明显区分的颜色（BGR）"""
        colors = []
        for i in range(n):
            hue = int(180 * i / n)
            hsv = np.uint8([[[hue, 255, 255]]])
            bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0, 0]
            colors.append(tuple(map(int, bgr)))
        return colors

    def _describe_color(self, hsv: np.ndarray) -> str:
        """根据 HSV 值生成颜色描述"""
        h, s, v = hsv

        # 低饱和度 -> 黑白灰
        if s < 30:
            if v < 50:
                return "black-like"
            elif v > 200:
                return "white-like"
            else:
                return "gray-like"

        # 高饱和度 -> 彩色
        if h < 10 or h > 170:
            return "red-like"
        elif 10 <= h < 25:
            return "orange-like"
        elif 25 <= h < 40:
            return "yellow-like"
        elif 40 <= h < 80:
            return "green-like"
        elif 80 <= h < 130:
            return "blue-like"
        elif 130 <= h < 170:
            return "purple-like"
        else:
            return "unknown"

    def _generate_color_palette(
        self,
        centers: np.ndarray,
        counts: List[int],
        output_dir: str,
        base_name: str
    ) -> str:
        """
        生成颜色色板可视化

        Args:
            centers: 聚类中心 (K, 3)
            counts: 各类别数量
            output_dir: 输出目录
            base_name: 基础文件名

        Returns:
            色板图像路径
        """
        k = len(centers)

        # 创建色板图像 (高度=100*k, 宽度=400)
        palette = np.ones((100 * k, 400, 3), dtype=np.uint8) * 255

        for i in range(k):
            y_start = i * 100
            y_end = (i + 1) * 100

            # 转换颜色到 BGR
            center_color = centers[i].astype(np.uint8).reshape(1, 1, 3)

            if self.color_space == 'lab':
                center_bgr = cv2.cvtColor(center_color, cv2.COLOR_Lab2BGR)[0, 0]
            elif self.color_space == 'hsv':
                center_bgr = cv2.cvtColor(center_color, cv2.COLOR_HSV2BGR)[0, 0]
            else:  # rgb
                center_bgr = cv2.cvtColor(center_color, cv2.COLOR_RGB2BGR)[0, 0]

            # 填充颜色块
            palette[y_start:y_end, :250] = center_bgr

            # 添加文本信息
            text = f"Cluster {i}: {counts[i]} regions"
            cv2.putText(palette, text, (260, y_start + 50),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

        # 保存色板
        palette_path = os.path.join(output_dir, f"{base_name}_color_palette.jpg")
        cv2.imwrite(palette_path, palette)

        return palette_path

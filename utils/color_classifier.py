"""
颜色分类器 - 基于网格聚类
将图像分成小网格(~1mm×1mm),对每个网格内非白色像素取中位数颜色进行聚类
"""
import cv2
import numpy as np
import os
from typing import Dict, List, Tuple, Optional
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


class ColorClassifier:
    """网格颜色分类器：基于网格聚类区分不同颜色"""

    def __init__(
        self,
        debug=False,
        white_threshold: int = 200,
        grid_size: int = 20,
        min_saturation: int = 30
    ):
        """
        初始化颜色分类器

        Args:
            debug: 是否输出调试信息
            white_threshold: 白色阈值(RGB≥此值为白色)
            grid_size: 网格大小(像素),默认20x20像素(约1mm×1mm)
            min_saturation: 彩色/灰色分界饱和度(HSV的S通道)
        """
        self.debug = debug
        self.white_threshold = white_threshold
        self.grid_size = grid_size
        self.min_saturation = min_saturation

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
                    '0': {'count': int, 'grid_count': int, 'color_rgb': [...], 'description': str},
                    ...
                },
                'annotated_path': str,
                'cluster_paths': [str, ...],
                'palette_path': str
            }
        """
        if self.debug:
            print(f"颜色聚类分类(白色阈值RGB≥{self.white_threshold}, 网格大小={self.grid_size}×{self.grid_size}像素)")
            print("=" * 80)

        # 1. 读取图像
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"无法读取图像: {image_path}")

        h, w = img.shape[:2]
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        if self.debug:
            print(f"图片尺寸: {w}x{h}, 总像素: {h*w}")

        # 2. 将图像分成网格,每个网格取中位数颜色
        if self.debug:
            print(f"将图像分成{self.grid_size}×{self.grid_size}像素的网格...")

        grid_h = h // self.grid_size
        grid_w = w // self.grid_size

        if self.debug:
            print(f"网格数量: {grid_w}×{grid_h} = {grid_w * grid_h}个网格")

        grid_colors = []
        grid_positions = []

        for i in range(grid_h):
            for j in range(grid_w):
                y_start = i * self.grid_size
                y_end = min((i + 1) * self.grid_size, h)
                x_start = j * self.grid_size
                x_end = min((j + 1) * self.grid_size, w)

                # 提取网格区域
                grid_region = img_rgb[y_start:y_end, x_start:x_end]

                # 过滤掉白色像素(RGB≥white_threshold)
                pixels = grid_region.reshape(-1, 3)
                non_white_pixels = pixels[
                    (pixels[:, 0] < self.white_threshold) |
                    (pixels[:, 1] < self.white_threshold) |
                    (pixels[:, 2] < self.white_threshold)
                ]

                # 如果网格内有足够的非白色像素,计算中位数颜色
                if len(non_white_pixels) > 10:  # 至少10个非白色像素
                    median_color = np.median(non_white_pixels, axis=0).astype(np.uint8)
                    grid_colors.append(median_color)
                    grid_positions.append((i, j))

        grid_colors = np.array(grid_colors)

        if self.debug:
            print(f"提取了{len(grid_colors)}个非白色网格(已过滤网格内白色背景)")

        if len(grid_colors) == 0:
            raise ValueError("没有非白色网格!")

        # 3. 转换为HSV(只使用色相H进行聚类)
        if self.debug:
            print("转换为HSV颜色空间...")

        grids_bgr = cv2.cvtColor(grid_colors.reshape(1, -1, 3), cv2.COLOR_RGB2BGR)
        hsv = cv2.cvtColor(grids_bgr, cv2.COLOR_BGR2HSV).reshape(-1, 3)

        # 只提取色相H (0-180度)
        hues = hsv[:, 0].reshape(-1, 1)
        saturations = hsv[:, 1]

        # 过滤掉饱和度太低的灰色网格(S<min_saturation)
        colored_mask = saturations >= self.min_saturation
        colored_hues = hues[colored_mask]
        colored_indices = np.where(colored_mask)[0]

        if self.debug:
            print(f"有色网格数: {len(colored_hues)} (饱和度≥{self.min_saturation})")
            print(f"灰色网格数: {len(grid_colors) - len(colored_hues)} (饱和度<{self.min_saturation})")

        # 4. 寻找最佳聚类数(基于色相)
        if len(colored_hues) < 10:
            if self.debug:
                print("警告: 几乎没有彩色网格,可能都是黑/灰色")
            # 降低饱和度阈值
            colored_mask = saturations >= 10
            colored_hues = hues[colored_mask]
            colored_indices = np.where(colored_mask)[0]
            if self.debug:
                print(f"降低饱和度阈值到10后,有色网格数: {len(colored_hues)}")

        # 采样
        max_samples = 20000
        if len(colored_hues) > max_samples:
            if self.debug:
                print(f"采样{max_samples}个有色网格进行聚类...")
            sample_indices = np.random.choice(len(colored_hues), max_samples, replace=False)
            samples = colored_hues[sample_indices]
        else:
            samples = colored_hues

        if self.debug:
            print("\n寻找最佳聚类数(基于色相H,K=2-6)...")

        best_k = 2
        best_score = -1

        for k in range(2, min(7, len(samples)//100 + 1)):
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            labels = kmeans.fit_predict(samples)
            if len(np.unique(labels)) > 1:
                score = silhouette_score(samples, labels)
                if self.debug:
                    print(f"  K={k}: silhouette={score:.3f}")
                if score > best_score:
                    best_score = score
                    best_k = k

        if self.debug:
            print(f"\n最佳聚类数: K={best_k} (silhouette={best_score:.3f})")

        # 5. 使用最佳K值聚类
        if self.debug:
            print("执行K-Means聚类(基于色相)...")

        kmeans = KMeans(n_clusters=best_k, random_state=42, n_init=10)
        kmeans.fit(colored_hues)

        # 聚类中心(色相)
        hue_centers = kmeans.cluster_centers_.flatten().astype(int)

        if self.debug:
            print(f"\n聚类中心色相:")
            for i, hue in enumerate(hue_centers):
                print(f"  类别{i}: 色相H={hue}° (0=红, 60=黄, 120=绿, 180=青)")

        # 6. 为所有有色网格分配类别
        if self.debug:
            print("\n为所有有色网格分配类别...")

        colored_labels = kmeans.predict(colored_hues)

        # 为所有网格分配类别(灰色先暂时单独一类)
        labels_all = np.full(len(grid_colors), best_k, dtype=np.uint8)  # 默认灰色类
        labels_all[colored_indices] = colored_labels

        # 7. 对灰色类进行二次聚类(分为深灰和浅灰)
        if self.debug:
            print("\n对灰色/黑色网格进行二次聚类...")

        gray_mask = ~colored_mask
        gray_grids = grid_colors[gray_mask]
        gray_indices = np.where(gray_mask)[0]

        if len(gray_grids) > 10:
            # 使用明度V进行聚类(区分深灰和浅灰)
            gray_hsv = hsv[gray_mask]
            gray_values = gray_hsv[:, 2].reshape(-1, 1)  # V通道(明度)

            # 灰色分为2类
            if self.debug:
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

            if self.debug:
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

        # 8. 计算每个类别的代表颜色(取该类别网格的中位数RGB)
        centers = []
        for i in range(total_classes):
            class_grids = grid_colors[labels_all == i]
            if len(class_grids) > 0:
                center_rgb = np.median(class_grids, axis=0).astype(np.uint8)  # 使用 uint8 而非 int 避免 CV_32S
                centers.append(center_rgb)
            else:
                centers.append(np.array([128, 128, 128], dtype=np.uint8))

        centers = np.array(centers)

        if self.debug:
            print(f"\n各类别代表颜色:")
            for i, center in enumerate(centers):
                hex_color = f"#{center[0]:02X}{center[1]:02X}{center[2]:02X}"
                if i < best_k:
                    print(f"  类别{i}(色相{hue_centers[i]}°): RGB{tuple(center)} = {hex_color}")
                elif i == best_k:
                    print(f"  类别{i}(深灰/黑色): RGB{tuple(center)} = {hex_color}")
                else:
                    print(f"  类别{i}(浅灰色): RGB{tuple(center)} = {hex_color}")

        # 9. 生成分类图
        if self.debug:
            print("生成分类图...")

        os.makedirs(output_dir, exist_ok=True)
        base_name = os.path.splitext(os.path.basename(image_path))[0]

        # 从网格级别映射回像素级别
        color_class_map = np.zeros((h, w), dtype=np.uint8)
        for idx, (i, j) in enumerate(grid_positions):
            y_start = i * self.grid_size
            y_end = min((i + 1) * self.grid_size, h)
            x_start = j * self.grid_size
            x_end = min((j + 1) * self.grid_size, w)
            color_class_map[y_start:y_end, x_start:x_end] = labels_all[idx] + 1

        # 为每个类别生成独立图像(白色背景)
        if self.debug:
            print(f"\n生成{total_classes}个颜色类别的分离图像:")

        cluster_paths = []
        cluster_info = {}

        for class_id in range(total_classes):
            class_mask = (color_class_map == class_id + 1).astype(np.uint8) * 255

            # 创建白色背景图像（使用 ones_like 而非 full_like 避免 dtype 冲突）
            class_result = np.ones_like(img) * 255
            # 只保留该类别的像素
            class_result = cv2.bitwise_and(img, img, mask=class_mask, dst=class_result)
            # 将非该类别区域设为白色
            white_mask = (class_mask == 0).astype(np.uint8) * 255
            class_result[white_mask == 255] = [255, 255, 255]

            class_path = os.path.join(output_dir, f"{base_name}_cluster_{class_id}.jpg")
            cv2.imwrite(class_path, class_result)
            cluster_paths.append(class_path)

            grid_count = np.sum(labels_all == class_id)
            pixel_count = np.sum(color_class_map == class_id + 1)
            center = centers[class_id]
            hex_color = f"#{center[0]:02X}{center[1]:02X}{center[2]:02X}"
            percentage = pixel_count / (h*w) * 100

            # 描述
            if class_id < best_k:
                description = f"色相{hue_centers[class_id]}°"
            elif class_id == best_k:
                description = "深灰/黑色"
            else:
                description = "浅灰色"

            # 计算HSV值
            center_bgr = cv2.cvtColor(center.reshape(1, 1, 3), cv2.COLOR_RGB2BGR)[0, 0]
            center_hsv = cv2.cvtColor(center_bgr.reshape(1, 1, 3), cv2.COLOR_BGR2HSV)[0, 0]

            cluster_info[str(class_id)] = {
                'count': int(pixel_count),
                'grid_count': int(grid_count),
                'color_rgb': center.tolist(),
                'color_hsv': center_hsv.tolist(),
                'description': description
            }

            if self.debug:
                print(f"  类别{class_id}({description}): RGB{tuple(center)} ({hex_color})")
                print(f"    网格数={grid_count}, 像素数={pixel_count} ({percentage:.2f}%) -> {class_path}")

        # 10. 生成标注图(可选,显示网格边界)
        annotated = img.copy()
        # 为每个类别生成不同颜色
        cluster_colors = self._generate_distinct_colors(total_classes)

        for idx, (i, j) in enumerate(grid_positions):
            y_start = i * self.grid_size
            y_end = min((i + 1) * self.grid_size, h)
            x_start = j * self.grid_size
            x_end = min((j + 1) * self.grid_size, w)

            label = labels_all[idx]
            color = cluster_colors[label]
            cv2.rectangle(annotated, (x_start, y_start), (x_end, y_end), color, 1)

        annotated_path = os.path.join(output_dir, f"{base_name}_color_annotated.jpg")
        cv2.imwrite(annotated_path, annotated)

        # 11. 生成颜色色板
        palette_path = self._generate_color_palette(centers, cluster_info, output_dir, base_name, best_k)

        if self.debug:
            print("\n✓ 完成")

        return {
            'n_clusters': total_classes,
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

    def _generate_color_palette(
        self,
        centers: np.ndarray,
        cluster_info: Dict,
        output_dir: str,
        base_name: str,
        n_colored: int
    ) -> str:
        """
        生成颜色色板可视化

        Args:
            centers: 聚类中心 RGB (K, 3)
            cluster_info: 类别信息字典
            output_dir: 输出目录
            base_name: 基础文件名
            n_colored: 彩色类别数

        Returns:
            色板图像路径
        """
        k = len(centers)

        # 创建色板图像 (高度=100*k, 宽度=400)
        palette = np.ones((100 * k, 400, 3), dtype=np.uint8) * 255

        for i in range(k):
            y_start = i * 100
            y_end = (i + 1) * 100

            # RGB转BGR
            center_bgr = cv2.cvtColor(centers[i].reshape(1, 1, 3), cv2.COLOR_RGB2BGR)[0, 0]

            # 填充颜色块
            palette[y_start:y_end, :250] = center_bgr

            # 添加文本信息
            info = cluster_info[str(i)]
            text = f"Class {i}: {info['grid_count']} grids"
            cv2.putText(palette, text, (260, y_start + 40),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

            desc = info['description']
            cv2.putText(palette, desc, (260, y_start + 65),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

        # 保存色板
        palette_path = os.path.join(output_dir, f"{base_name}_color_palette.jpg")
        cv2.imwrite(palette_path, palette)

        return palette_path

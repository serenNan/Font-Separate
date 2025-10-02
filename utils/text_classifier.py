"""
手写体与印刷体分类器
基于形态学特征和连通组件分析区分手写体和印刷体文字
"""
import cv2
import numpy as np
from typing import List, Tuple, Dict
import os


class TextClassifier:
    """文字分类器:区分手写体和印刷体"""

    def __init__(self, debug=False):
        self.debug = debug

    def preprocess(self, image_path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        图像预处理
        Returns:
            original: 原始彩色图像
            gray: 灰度图
            binary: 二值图
        """
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"无法读取图像: {image_path}")

        # 转灰度
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # 二值化 - 使用Otsu自动阈值
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        return img, gray, binary

    def extract_text_regions(self, binary: np.ndarray, min_area: int = 200) -> List[Dict]:
        """
        提取文字区域
        Args:
            binary: 二值图
            min_area: 最小区域面积(过滤噪点)
        Returns:
            文字区域列表 [{'bbox': (x,y,w,h), 'pixels': count}]
        """
        # 使用连通组件分析找到文字区域
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            binary, connectivity=8
        )

        regions = []
        for i in range(1, num_labels):  # 跳过背景(0)
            area = stats[i, cv2.CC_STAT_AREA]
            if area < min_area:
                continue

            x = stats[i, cv2.CC_STAT_LEFT]
            y = stats[i, cv2.CC_STAT_TOP]
            w = stats[i, cv2.CC_STAT_WIDTH]
            h = stats[i, cv2.CC_STAT_HEIGHT]

            regions.append({
                'bbox': (x, y, w, h),
                'area': area,
                'label': i
            })

        if self.debug:
            print(f"检测到 {len(regions)} 个文字区域")

        return regions

    def calculate_stroke_features(self, roi_binary: np.ndarray) -> Dict:
        """
        计算笔画特征
        手写体特征:
        - 笔画粗细变化大 (标准差高)
        - 笔画连续性强
        - 墨迹浓淡变化

        印刷体特征:
        - 笔画均匀
        - 字形规整
        - 边缘锐利

        Args:
            roi_binary: 区域二值图
        Returns:
            特征字典
        """
        h, w = roi_binary.shape

        # 1. 笔画宽度变化 (Stroke Width Variation)
        # 使用骨架化+距离变换分析笔画宽度
        skeleton = cv2.ximgproc.thinning(roi_binary)
        dist_transform = cv2.distanceTransform(roi_binary, cv2.DIST_L2, 5)

        # 提取骨架点的距离值(代表笔画宽度的一半)
        stroke_widths = dist_transform[skeleton > 0]

        if len(stroke_widths) > 0:
            stroke_width_mean = float(np.mean(stroke_widths))
            stroke_width_std = float(np.std(stroke_widths))
            stroke_width_cv = stroke_width_std / stroke_width_mean if stroke_width_mean > 0 else 0
        else:
            stroke_width_mean = 0
            stroke_width_std = 0
            stroke_width_cv = 0

        # 2. 边缘复杂度
        contours, _ = cv2.findContours(roi_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        if contours:
            contour = max(contours, key=cv2.contourArea)
            perimeter = cv2.arcLength(contour, True)
            area = cv2.contourArea(contour)
            # 圆形度: 4π*面积/周长^2, 圆形=1, 不规则<1
            circularity = (4 * np.pi * area) / (perimeter * perimeter) if perimeter > 0 else 0
        else:
            circularity = 0

        # 3. 纵横比
        aspect_ratio = float(w) / h if h > 0 else 0

        # 4. 像素密度
        density = np.sum(roi_binary > 0) / (w * h) if (w * h) > 0 else 0

        return {
            'stroke_width_mean': stroke_width_mean,
            'stroke_width_std': stroke_width_std,
            'stroke_width_cv': stroke_width_cv,  # 变异系数:手写体通常>0.3
            'circularity': circularity,
            'aspect_ratio': aspect_ratio,
            'density': density
        }

    def classify_region(self, features: Dict) -> str:
        """
        根据特征分类文字区域
        Args:
            features: 特征字典
        Returns:
            'handwritten': 手写体
            'printed': 印刷体
        """
        stroke_cv = features['stroke_width_cv']
        circularity = features['circularity']
        density = features['density']

        # 分类规则(基于经验阈值):
        # 手写体: 笔画变化大(cv>0.3), 形状不规则(circularity<0.4)
        # 印刷体: 笔画均匀(cv<0.25), 形状规整

        handwritten_score = 0

        # 笔画宽度变异系数
        if stroke_cv > 0.35:
            handwritten_score += 3
        elif stroke_cv > 0.25:
            handwritten_score += 1
        else:
            handwritten_score -= 2

        # 圆形度(不规则度)
        if circularity < 0.3:
            handwritten_score += 2
        elif circularity < 0.5:
            handwritten_score += 1

        # 密度(手写体墨迹可能更浓)
        if density > 0.4:
            handwritten_score += 1

        if self.debug:
            print(f"  特征: stroke_cv={stroke_cv:.3f}, circ={circularity:.3f}, "
                  f"density={density:.3f}, score={handwritten_score}")

        return 'handwritten' if handwritten_score >= 2 else 'printed'

    def merge_nearby_regions(self, regions: List[Dict],
                            binary: np.ndarray,
                            distance_threshold: int = 50) -> List[Dict]:
        """
        合并相近的同类文字区域
        Args:
            regions: 分类后的区域列表
            binary: 二值图
            distance_threshold: 合并距离阈值
        Returns:
            合并后的大区域列表
        """
        if not regions:
            return []

        # 按类型分组
        handwritten_regions = [r for r in regions if r['type'] == 'handwritten']
        printed_regions = [r for r in regions if r['type'] == 'printed']

        def merge_group(group):
            if not group:
                return []

            # 简单合并:创建最小外接矩形
            merged = []
            for region_type in ['handwritten', 'printed']:
                type_regions = [r for r in group if r.get('type') == region_type]
                if not type_regions:
                    continue

                # 找到所有bbox的并集
                all_x = [r['bbox'][0] for r in type_regions]
                all_y = [r['bbox'][1] for r in type_regions]
                all_x2 = [r['bbox'][0] + r['bbox'][2] for r in type_regions]
                all_y2 = [r['bbox'][1] + r['bbox'][3] for r in type_regions]

                x_min, x_max = min(all_x), max(all_x2)
                y_min, y_max = min(all_y), max(all_y2)

                merged.append({
                    'bbox': (x_min, y_min, x_max - x_min, y_max - y_min),
                    'type': region_type,
                    'count': len(type_regions)
                })

            return merged

        # 分别合并手写和印刷区域
        h_merged = merge_group(handwritten_regions) if handwritten_regions else []
        p_merged = merge_group(printed_regions) if printed_regions else []

        return h_merged + p_merged

    def classify_and_separate(self, image_path: str, output_dir: str,
                             table_mask: np.ndarray = None) -> Dict:
        """
        分类并分离手写体和印刷体
        Args:
            image_path: 图像路径
            output_dir: 输出目录
            table_mask: 表格掩码(已分离的表格区域,不再处理)
        Returns:
            结果字典
        """
        # 预处理
        original_img, gray, binary = self.preprocess(image_path)
        h, w = original_img.shape[:2]

        # 如果提供了表格掩码,从binary中去除表格区域
        if table_mask is not None:
            # 表格区域置0(不处理)
            binary = cv2.bitwise_and(binary, cv2.bitwise_not(table_mask))

        # 提取文字区域
        text_regions = self.extract_text_regions(binary, min_area=200)

        if self.debug:
            print(f"\n开始分类 {len(text_regions)} 个文字区域...")

        # 对每个区域进行分类
        classified_regions = []
        for i, region in enumerate(text_regions):
            x, y, rw, rh = region['bbox']

            # 提取ROI
            roi_binary = binary[y:y+rh, x:x+rw]

            # 计算特征
            features = self.calculate_stroke_features(roi_binary)

            # 分类
            text_type = self.classify_region(features)

            region['type'] = text_type
            region['features'] = features
            classified_regions.append(region)

            if self.debug and i < 10:  # 只打印前10个
                print(f"区域 {i}: {text_type}")

        # 统计
        handwritten_count = sum(1 for r in classified_regions if r['type'] == 'handwritten')
        printed_count = sum(1 for r in classified_regions if r['type'] == 'printed')

        if self.debug:
            print(f"\n分类结果: 手写体={handwritten_count}, 印刷体={printed_count}")

        # 创建掩码
        handwritten_mask = np.zeros((h, w), dtype=np.uint8)
        printed_mask = np.zeros((h, w), dtype=np.uint8)

        # 标注图像
        annotated_img = original_img.copy()

        for region in classified_regions:
            x, y, rw, rh = region['bbox']

            if region['type'] == 'handwritten':
                # 手写体 - 红色
                cv2.rectangle(handwritten_mask, (x, y), (x+rw, y+rh), 255, -1)
                cv2.rectangle(annotated_img, (x, y), (x+rw, y+rh), (0, 0, 255), 2)
                cv2.putText(annotated_img, 'H', (x, y-5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
            else:
                # 印刷体 - 蓝色
                cv2.rectangle(printed_mask, (x, y), (x+rw, y+rh), 255, -1)
                cv2.rectangle(annotated_img, (x, y), (x+rw, y+rh), (255, 0, 0), 2)
                cv2.putText(annotated_img, 'P', (x, y-5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

        # 提取区域
        handwritten_img = cv2.bitwise_and(original_img, original_img, mask=handwritten_mask)
        printed_img = cv2.bitwise_and(original_img, original_img, mask=printed_mask)

        # 保存
        os.makedirs(output_dir, exist_ok=True)
        base_name = os.path.splitext(os.path.basename(image_path))[0]

        handwritten_path = os.path.join(output_dir, f"{base_name}_handwritten.jpg")
        printed_path = os.path.join(output_dir, f"{base_name}_printed.jpg")
        annotated_path = os.path.join(output_dir, f"{base_name}_text_annotated.jpg")

        cv2.imwrite(handwritten_path, handwritten_img)
        cv2.imwrite(printed_path, printed_img)
        cv2.imwrite(annotated_path, annotated_img)

        return {
            'handwritten_path': handwritten_path,
            'printed_path': printed_path,
            'annotated_path': annotated_path,
            'handwritten_count': handwritten_count,
            'printed_count': printed_count,
            'regions': classified_regions
        }

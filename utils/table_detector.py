"""
表格检测模块
使用Hough变换检测直线，识别表格结构
"""
import cv2
import numpy as np
from typing import List, Tuple, Dict
import os


class TableDetector:
    """表格检测器"""

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

    def detect_lines(self, binary: np.ndarray) -> Tuple[List, List]:
        """
        检测直线（表格线）
        Returns:
            horizontal_lines: 水平线列表
            vertical_lines: 垂直线列表
        """
        # 检测水平线
        horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 1))
        detect_horizontal = cv2.morphologyEx(binary, cv2.MORPH_OPEN, horizontal_kernel, iterations=2)
        horizontal_lines = cv2.HoughLinesP(
            detect_horizontal,
            rho=1,
            theta=np.pi/180,
            threshold=100,
            minLineLength=100,
            maxLineGap=10
        )

        # 检测垂直线
        vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 40))
        detect_vertical = cv2.morphologyEx(binary, cv2.MORPH_OPEN, vertical_kernel, iterations=2)
        vertical_lines = cv2.HoughLinesP(
            detect_vertical,
            rho=1,
            theta=np.pi/180,
            threshold=100,
            minLineLength=100,
            maxLineGap=10
        )

        if self.debug:
            h_count = len(horizontal_lines) if horizontal_lines is not None else 0
            v_count = len(vertical_lines) if vertical_lines is not None else 0
            print(f"检测到 {h_count} 条水平线, {v_count} 条垂直线")

        return horizontal_lines, vertical_lines

    def merge_nearby_lines(self, lines: np.ndarray, is_horizontal: bool, threshold: int = 15) -> List:
        """
        合并相近的线条
        Args:
            lines: 线条数组
            is_horizontal: 是否为水平线
            threshold: 合并阈值（像素）
        """
        if lines is None or len(lines) == 0:
            return []

        # 按位置排序
        if is_horizontal:
            # 水平线按y坐标排序
            lines_sorted = sorted(lines, key=lambda x: (x[0][1] + x[0][3]) / 2)
        else:
            # 垂直线按x坐标排序
            lines_sorted = sorted(lines, key=lambda x: (x[0][0] + x[0][2]) / 2)

        merged = []
        current_group = [lines_sorted[0]]

        for line in lines_sorted[1:]:
            if is_horizontal:
                # 比较y坐标
                last_y = (current_group[-1][0][1] + current_group[-1][0][3]) / 2
                curr_y = (line[0][1] + line[0][3]) / 2
                if abs(curr_y - last_y) < threshold:
                    current_group.append(line)
                else:
                    merged.append(self._average_line(current_group, is_horizontal))
                    current_group = [line]
            else:
                # 比较x坐标
                last_x = (current_group[-1][0][0] + current_group[-1][0][2]) / 2
                curr_x = (line[0][0] + line[0][2]) / 2
                if abs(curr_x - last_x) < threshold:
                    current_group.append(line)
                else:
                    merged.append(self._average_line(current_group, is_horizontal))
                    current_group = [line]

        # 添加最后一组
        merged.append(self._average_line(current_group, is_horizontal))

        return merged

    def _average_line(self, lines: List, is_horizontal: bool) -> np.ndarray:
        """计算一组线条的平均线"""
        lines_arr = np.array([line[0] for line in lines])

        if is_horizontal:
            # 水平线：取平均y坐标，x坐标取最大范围
            y_avg = int(np.mean(lines_arr[:, [1, 3]]))
            x_min = int(np.min(lines_arr[:, [0, 2]]))
            x_max = int(np.max(lines_arr[:, [0, 2]]))
            return np.array([[x_min, y_avg, x_max, y_avg]])
        else:
            # 垂直线：取平均x坐标，y坐标取最大范围
            x_avg = int(np.mean(lines_arr[:, [0, 2]]))
            y_min = int(np.min(lines_arr[:, [1, 3]]))
            y_max = int(np.max(lines_arr[:, [1, 3]]))
            return np.array([[x_avg, y_min, x_avg, y_max]])

    def calculate_content_density(self, region_img: np.ndarray, binary_img: np.ndarray,
                                   bbox: Tuple[int, int, int, int]) -> Dict:
        """
        计算区域的内容密度特征
        Args:
            region_img: 区域图像
            binary_img: 二值图
            bbox: 边界框 (x, y, w, h)
        Returns:
            密度特征字典
        """
        x, y, w, h = bbox
        roi_binary = binary_img[y:y+h, x:x+w]

        # 计算黑色像素比例（文字密度）
        total_pixels = w * h
        black_pixels = np.sum(roi_binary > 0)
        density = black_pixels / total_pixels if total_pixels > 0 else 0

        # 计算连通组件数量（文字块数量）
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(roi_binary, connectivity=8)
        # 过滤掉太小的组件（噪点）
        valid_components = [i for i in range(1, num_labels)
                          if stats[i, cv2.CC_STAT_AREA] > 50]

        return {
            'density': density,
            'component_count': len(valid_components),
            'area': total_pixels
        }

    def classify_table_type(self, content_info: Dict, h_lines: int, v_lines: int) -> str:
        """
        根据内容密度分类表格类型
        Returns:
            'data_table': 数据表格（保留）
            'empty_table': 空白表格（过滤）
            'title_box': 标题框（过滤）
        """
        density = content_info['density']
        component_count = content_info['component_count']

        # 空白表格：密度很低，组件很少
        if density < 0.01 and component_count < 3:
            return 'empty_table'

        # 标题框特征：
        # 1. 密度很高（>0.15）且组件少（文字密集但内容少）
        # 2. 横线很少（<=2）说明不是数据表格结构
        if density > 0.15 and component_count < 50:
            return 'title_box'

        if h_lines <= 2:
            return 'title_box'

        # 数据表格：有足够的横线形成行结构
        return 'data_table'

    def find_table_regions(self, h_lines: List, v_lines: List, img_shape: Tuple,
                          binary_img: np.ndarray = None) -> List[Dict]:
        """
        根据横竖线找到表格区域
        使用聚类方法识别独立的表格区域，并根据内容密度过滤
        Args:
            h_lines: 水平线列表
            v_lines: 垂直线列表
            img_shape: 图像尺寸
            binary_img: 二值图（用于内容密度分析）
        Returns:
            表格区域列表 [{'bbox': (x, y, w, h), 'type': 'table'}]
        """
        if not h_lines or not v_lines:
            return []

        # 找到所有交叉点
        intersections = []
        for h_line in h_lines:
            x1, y1, x2, y2 = h_line[0]
            for v_line in v_lines:
                x3, y3, x4, y4 = v_line[0]
                # 检查是否相交
                if x3 >= x1 and x3 <= x2 and y1 >= y3 and y1 <= y4:
                    intersections.append((x3, y1, h_line, v_line))

        if len(intersections) < 4:  # 至少需要4个交点才能形成表格
            return []

        # 使用线条聚类来识别独立的表格区域
        # 基于垂直线的x坐标进行聚类
        v_xs = sorted([v_line[0][0] for v_line in v_lines])

        # 找到垂直线的间隙（大间隙表示不同表格区域）
        v_gaps = []
        for i in range(len(v_xs) - 1):
            gap = v_xs[i + 1] - v_xs[i]
            if gap > 100:  # 间隙阈值：超过100像素认为是不同区域
                v_gaps.append((v_xs[i], v_xs[i + 1]))

        tables = []

        if len(v_gaps) == 0:
            # 没有大间隙，整个区域是一个表格
            table = self._create_table_from_lines(h_lines, v_lines, img_shape)
            tables.append(table)
        else:
            # 有大间隙，分割成多个表格区域
            # 左侧区域
            left_v_lines = [v for v in v_lines if v[0][0] <= v_gaps[0][0]]
            if len(left_v_lines) >= 2:
                left_h_lines = self._filter_h_lines_for_region(
                    h_lines, 0, v_gaps[0][0], img_shape
                )
                if len(left_h_lines) >= 2:
                    table = self._create_table_from_lines(
                        left_h_lines, left_v_lines, img_shape
                    )
                    tables.append(table)

            # 中间区域（如果有多个间隙）
            for i in range(len(v_gaps) - 1):
                mid_v_lines = [v for v in v_lines
                              if v_gaps[i][1] <= v[0][0] <= v_gaps[i + 1][0]]
                if len(mid_v_lines) >= 2:
                    mid_h_lines = self._filter_h_lines_for_region(
                        h_lines, v_gaps[i][1], v_gaps[i + 1][0], img_shape
                    )
                    if len(mid_h_lines) >= 2:
                        table = self._create_table_from_lines(
                            mid_h_lines, mid_v_lines, img_shape
                        )
                        tables.append(table)

            # 右侧区域
            right_v_lines = [v for v in v_lines if v[0][0] >= v_gaps[-1][1]]
            if len(right_v_lines) >= 2:
                right_h_lines = self._filter_h_lines_for_region(
                    h_lines, v_gaps[-1][1], img_shape[1], img_shape
                )
                if len(right_h_lines) >= 2:
                    table = self._create_table_from_lines(
                        right_h_lines, right_v_lines, img_shape
                    )
                    tables.append(table)

        # 如果提供了二值图，进行内容密度过滤
        if binary_img is not None and len(tables) > 0:
            filtered_tables = []
            for table in tables:
                x, y, w, h = table['bbox']
                # 确保bbox在图像范围内
                x = max(0, min(x, img_shape[1] - 1))
                y = max(0, min(y, img_shape[0] - 1))
                w = min(w, img_shape[1] - x)
                h = min(h, img_shape[0] - y)

                if w > 0 and h > 0:
                    # 创建临时区域图像用于密度分析
                    region_img = np.zeros((h, w, 3), dtype=np.uint8)
                    content_info = self.calculate_content_density(
                        region_img, binary_img, (x, y, w, h)
                    )

                    table_type = self.classify_table_type(
                        content_info,
                        table['h_lines'],
                        table['v_lines']
                    )

                    table['table_type'] = table_type
                    table['content_density'] = content_info['density']
                    table['component_count'] = content_info['component_count']

                    # 只保留数据表格
                    if table_type == 'data_table':
                        filtered_tables.append(table)
                    elif self.debug:
                        print(f"过滤掉 {table_type}: 密度={content_info['density']:.3f}, "
                              f"组件={content_info['component_count']}")

            return filtered_tables

        return tables

    def _filter_h_lines_for_region(self, h_lines: List, x_min: int, x_max: int, img_shape: Tuple) -> List:
        """过滤出指定x范围内的水平线"""
        filtered = []
        for h_line in h_lines:
            x1, y1, x2, y2 = h_line[0]
            # 线条的中点在范围内，或者与范围有重叠
            line_center = (x1 + x2) / 2
            if x_min <= line_center <= x_max or (x1 <= x_max and x2 >= x_min):
                filtered.append(h_line)
        return filtered

    def _create_table_from_lines(self, h_lines: List, v_lines: List, img_shape: Tuple) -> Dict:
        """从线条创建表格区域"""
        h_ys = [h_line[0][1] for h_line in h_lines]
        v_xs = [v_line[0][0] for v_line in v_lines]

        x_min = min(v_xs)
        x_max = max(v_xs)
        y_min = min(h_ys)
        y_max = max(h_ys)

        # 添加边距
        margin = 10
        x_min = max(0, x_min - margin)
        y_min = max(0, y_min - margin)
        x_max = min(img_shape[1], x_max + margin)
        y_max = min(img_shape[0], y_max + margin)

        return {
            'bbox': (x_min, y_min, x_max - x_min, y_max - y_min),
            'type': 'table',
            'h_lines': len(h_lines),
            'v_lines': len(v_lines)
        }

    def separate_and_save(self, image_path: str, output_dir: str) -> Dict:
        """
        分离表格和非表格区域并保存
        """
        # 预处理
        original_img, gray, binary = self.preprocess(image_path)
        h, w = original_img.shape[:2]

        # 检测线条
        h_lines, v_lines = self.detect_lines(binary)

        # 合并相近线条
        h_lines_merged = self.merge_nearby_lines(h_lines, is_horizontal=True)
        v_lines_merged = self.merge_nearby_lines(v_lines, is_horizontal=False)

        if self.debug:
            print(f"合并后: {len(h_lines_merged)} 条水平线, {len(v_lines_merged)} 条垂直线")

        # 找到表格区域（传入二值图进行内容密度分析）
        table_regions = self.find_table_regions(h_lines_merged, v_lines_merged, (h, w), binary)

        if self.debug:
            print(f"检测到 {len(table_regions)} 个表格区域")

        # 创建掩码
        table_mask = np.zeros((h, w), dtype=np.uint8)
        non_table_mask = np.ones((h, w), dtype=np.uint8) * 255

        # 标注图像
        annotated_img = original_img.copy()

        for table in table_regions:
            x, y, w, h = table['bbox']
            # 填充表格掩码
            cv2.rectangle(table_mask, (x, y), (x + w, y + h), 255, -1)
            # 从非表格掩码中去除
            cv2.rectangle(non_table_mask, (x, y), (x + w, y + h), 0, -1)
            # 标注
            cv2.rectangle(annotated_img, (x, y), (x + w, y + h), (0, 255, 0), 3)

            # 显示表格类型和密度信息
            label = f"Table ({table['h_lines']}x{table['v_lines']})"
            if 'content_density' in table:
                label += f" D:{table['content_density']:.2f}"

            cv2.putText(
                annotated_img,
                label,
                (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2
            )

        # 绘制检测到的线条
        lines_img = original_img.copy()
        if h_lines_merged:
            for line in h_lines_merged:
                x1, y1, x2, y2 = line[0]
                cv2.line(lines_img, (x1, y1), (x2, y2), (255, 0, 0), 2)
        if v_lines_merged:
            for line in v_lines_merged:
                x1, y1, x2, y2 = line[0]
                cv2.line(lines_img, (x1, y1), (x2, y2), (0, 0, 255), 2)

        # 提取区域
        table_img = cv2.bitwise_and(original_img, original_img, mask=table_mask)
        non_table_img = cv2.bitwise_and(original_img, original_img, mask=non_table_mask)

        # 保存
        os.makedirs(output_dir, exist_ok=True)
        base_name = os.path.splitext(os.path.basename(image_path))[0]

        table_path = os.path.join(output_dir, f"{base_name}_table.jpg")
        non_table_path = os.path.join(output_dir, f"{base_name}_non_table.jpg")
        annotated_path = os.path.join(output_dir, f"{base_name}_table_annotated.jpg")
        lines_path = os.path.join(output_dir, f"{base_name}_lines.jpg")

        cv2.imwrite(table_path, table_img)
        cv2.imwrite(non_table_path, non_table_img)
        cv2.imwrite(annotated_path, annotated_img)
        cv2.imwrite(lines_path, lines_img)

        return {
            'table_path': table_path,
            'non_table_path': non_table_path,
            'annotated_path': annotated_path,
            'lines_path': lines_path,
            'table_count': len(table_regions),
            'table_regions': table_regions
        }
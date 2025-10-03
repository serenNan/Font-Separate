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

    def __init__(self, debug=False, use_simple_mode=True):
        """
        Args:
            debug: 调试模式
            use_simple_mode: 简单模式（直接按布局分离，不检测表格线）
        """
        self.debug = debug
        self.use_simple_mode = use_simple_mode

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
        # 检测水平线（大幅降低阈值以适应历史文档的模糊线条）
        horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (30, 1))  # 减小核40→30
        detect_horizontal = cv2.morphologyEx(binary, cv2.MORPH_OPEN, horizontal_kernel, iterations=1)  # 减少迭代2→1
        horizontal_lines = cv2.HoughLinesP(
            detect_horizontal,
            rho=1,
            theta=np.pi/180,
            threshold=30,       # 大幅降低阈值 50→30
            minLineLength=30,   # 降低最小长度 50→30
            maxLineGap=30       # 增加允许间隙 20→30（连接断裂线条）
        )

        # 检测垂直线
        vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 30))  # 减小核40→30
        detect_vertical = cv2.morphologyEx(binary, cv2.MORPH_OPEN, vertical_kernel, iterations=1)  # 减少迭代2→1
        vertical_lines = cv2.HoughLinesP(
            detect_vertical,
            rho=1,
            theta=np.pi/180,
            threshold=30,       # 大幅降低阈值 50→30
            minLineLength=30,   # 降低最小长度 50→30
            maxLineGap=30       # 增加允许间隙 20→30
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

        # 数据表格特征（优先判断）：
        # 1. 竖线很多（>12）且组件很多（>300）→ 多列数据表格
        if v_lines > 12 and component_count > 300:
            return 'data_table'

        # 2. 横线很多（>15）→ 多行表格
        if h_lines > 15:
            return 'data_table'

        # 3. 横竖线都较多（横>8 且竖>15）
        if h_lines > 8 and v_lines > 15:
            return 'data_table'

        # 标题框判断（注意：标题框也需要保留，只是名称叫title_box）
        # 实际上所有带框线的区域都应该保留（表格+标题框）
        # 真正要过滤的是：组件少且线条很少的纯手写批注区域

        # 过滤条件：组件少（<100）且横线很少（<5）且竖线很少（<5）
        # → 这是纯手写批注，没有明显的表格结构
        if component_count < 100 and h_lines < 5 and v_lines < 5:
            return 'title_box'

        # 默认当作数据表格（宽松策略，避免漏检）
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

        # 使用间隙分析找到表格区域
        # 1. 按垂直线的x坐标排序
        v_xs = sorted([v_line[0][0] for v_line in v_lines])

        # 2. 计算相邻垂直线之间的间隙
        gaps = []
        for i in range(len(v_xs) - 1):
            gap = v_xs[i + 1] - v_xs[i]
            gaps.append((i, gap, v_xs[i], v_xs[i + 1]))

        # 3. 找到最大间隙作为分界点（分隔不同区域）
        if len(gaps) > 0:
            # 按间隙大小排序
            gaps_sorted = sorted(gaps, key=lambda x: x[1], reverse=True)

            # 找到明显的大间隙（超过平均间隙的2倍）
            avg_gap = sum(g[1] for g in gaps) / len(gaps)
            large_gaps = [g for g in gaps_sorted if g[1] > avg_gap * 2 and g[1] > 80]

            if self.debug and large_gaps:
                print(f"发现 {len(large_gaps)} 个大间隙（平均间隙={avg_gap:.1f}px）")
                for idx, gap, x1, x2 in large_gaps[:3]:
                    print(f"  间隙 {idx}: {gap:.0f}px at x={x1:.0f}-{x2:.0f}")

            # 根据大间隙分割区域
            merged_regions = []
            if large_gaps:
                # 取第一个大间隙作为分界点
                split_gap = large_gaps[0]
                split_x = split_gap[3]  # 间隙结束位置

                # 左侧区域（第一个大间隙之前）
                left_v_xs = [x for x in v_xs if x < split_x]
                if len(left_v_xs) >= 3:
                    merged_regions.append((min(left_v_xs), max(left_v_xs), len(left_v_xs)))

                # 右侧区域（第一个大间隙之后）
                right_v_xs = [x for x in v_xs if x >= split_x]
                if len(right_v_xs) >= 3:
                    merged_regions.append((min(right_v_xs), max(right_v_xs), len(right_v_xs)))
            else:
                # 没有明显间隙，整个区域是一个表格
                merged_regions.append((min(v_xs), max(v_xs), len(v_xs)))

            if self.debug:
                print(f"发现 {len(merged_regions)} 个表格区域")
        else:
            merged_regions = []

        tables = []

        # 4. 为每个密集区域创建表格
        for region_start, region_end, line_count in merged_regions:
            # 提取该区域的垂直线
            region_v_lines = [v for v in v_lines
                            if region_start <= v[0][0] <= region_end]

            if len(region_v_lines) >= 3:  # 至少3条垂直线
                # 提取该区域的水平线
                region_h_lines = self._filter_h_lines_for_region(
                    h_lines, region_start, region_end, img_shape
                )

                if len(region_h_lines) >= 2:  # 至少2条水平线
                    table = self._create_table_from_lines(
                        region_h_lines, region_v_lines, img_shape
                    )
                    tables.append(table)
                    if self.debug:
                        print(f"  区域 [{region_start:.0f}, {region_end:.0f}]: "
                              f"{len(region_v_lines)}条竖线, {len(region_h_lines)}条横线")

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
            # 只要线条有任何部分在x范围内就保留
            if (x1 <= x_max and x2 >= x_min):  # 有重叠即可
                filtered.append(h_line)
        return filtered

    def _create_table_from_lines(self, h_lines: List, v_lines: List, img_shape: Tuple) -> Dict:
        """从线条创建表格区域"""
        # 收集垂直线的x坐标和水平线的y坐标
        v_xs = [v_line[0][0] for v_line in v_lines]  # 只取x坐标（垂直线位置）
        h_ys = [h_line[0][1] for h_line in h_lines]  # 只取y坐标（水平线位置）

        # 同时收集垂直线的y范围和水平线的x范围
        v_y_min = min([v_line[0][1] for v_line in v_lines])
        v_y_max = max([v_line[0][3] for v_line in v_lines])
        h_x_min = min([h_line[0][0] for h_line in h_lines])
        h_x_max = max([h_line[0][2] for h_line in h_lines])

        # x范围：取垂直线的min/max
        x_min = min(v_xs)
        x_max = max(v_xs)

        # y范围：取水平线的min/max
        y_min = min(h_ys)
        y_max = max(h_ys)

        # 扩大边距以包含外框线条的宽度和可能遗漏的边缘
        margin = 30  # 增加边距从10→30像素
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

        # 合并相近线条（增加合并阈值，避免过度分割）
        h_lines_merged = self.merge_nearby_lines(h_lines, is_horizontal=True, threshold=20)  # 5→20
        v_lines_merged = self.merge_nearby_lines(v_lines, is_horizontal=False, threshold=20)  # 10→20

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

        # 创建擦除线条后的图像
        img_without_lines = original_img.copy()

        # 擦除检测到的水平线
        if h_lines_merged:
            for line in h_lines_merged:
                x1, y1, x2, y2 = line[0]
                cv2.line(img_without_lines, (x1, y1), (x2, y2), (255, 255, 255), 3)

        # 擦除检测到的垂直线
        if v_lines_merged:
            for line in v_lines_merged:
                x1, y1, x2, y2 = line[0]
                cv2.line(img_without_lines, (x1, y1), (x2, y2), (255, 255, 255), 3)

        # 提取区域
        # 表格区域：提取线条框架，去除文字内容
        table_img = original_img.copy()
        table_img[table_mask == 0] = 255  # 非表格区域填白

        # 转灰度并二值化
        gray_table = cv2.cvtColor(table_img, cv2.COLOR_BGR2GRAY)
        _, binary_table = cv2.threshold(gray_table, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        # 使用中等核提取横线和竖线
        # 核大小25-30在保留表格线和过滤文字笔画间取得平衡
        h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 1))
        v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 25))

        h_lines_mask = cv2.morphologyEx(binary_table, cv2.MORPH_OPEN, h_kernel, iterations=1)
        v_lines_mask = cv2.morphologyEx(binary_table, cv2.MORPH_OPEN, v_kernel, iterations=1)

        # 合并横竖线
        lines_mask = cv2.add(h_lines_mask, v_lines_mask)

        # 转回BGR（反转黑白）
        table_img = cv2.cvtColor(255 - lines_mask, cv2.COLOR_GRAY2BGR)

        # 非表格区域：使用擦除线条后的图像（保留文字）
        non_table_img = cv2.bitwise_and(img_without_lines, img_without_lines, mask=non_table_mask)

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
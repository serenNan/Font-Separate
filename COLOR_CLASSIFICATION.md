# 颜色分类功能使用说明

## 功能概述

**自适应颜色分类器** - 根据文档中实际出现的文字颜色动态分类，无需预设固定颜色。

### 核心特点

✅ **完全自适应** - 不预设任何颜色（黑/蓝/红），自动识别文档中的颜色类别
✅ **智能聚类** - 自动确定最佳分类数（2-6类）或手动指定
✅ **鲁棒性强** - 排除背景像素、使用中位数颜色、Lab色彩空间聚类
✅ **可视化输出** - 标注图、颜色色板、各类别独立图像、统计信息

---

## 快速开始

### 基本使用

```bash
# 激活conda环境
conda activate font-separate

# 自动检测颜色类别数（推荐）
python color_classify_demo.py Pictures/原始.jpg

# 调试模式（查看详细过程）
python color_classify_demo.py Pictures/原始.jpg --debug
```

### 输出结果

```
results/
├── 原始_color_annotated.jpg      # 标注图（各类别不同颜色边框）
├── 原始_cluster_0.jpg            # 类别0（如：深色文字）
├── 原始_cluster_1.jpg            # 类别1（如：浅色文字）
├── 原始_color_palette.jpg        # 颜色代表色卡
└── 原始_color_stats.json         # 颜色统计信息（JSON）
```

---

## 高级用法

### 指定颜色类别数

```bash
# 强制分为3类
python color_classify_demo.py Pictures/原始.jpg --n-clusters 3

# 强制分为5类
python color_classify_demo.py Pictures/原始.jpg --n-clusters 5
```

### 切换颜色空间

```bash
# 使用HSV颜色空间
python color_classify_demo.py Pictures/原始.jpg --color-space hsv

# 使用RGB颜色空间
python color_classify_demo.py Pictures/原始.jpg --color-space rgb

# 使用Lab颜色空间（默认，推荐）
python color_classify_demo.py Pictures/原始.jpg --color-space lab
```

### 自定义参数

```bash
# 自定义自动检测范围（3-8类之间选择）
python color_classify_demo.py Pictures/原始.jpg --auto-k-range 3 8

# 自定义输出目录
python color_classify_demo.py Pictures/原始.jpg --output my_results

# 调整饱和度阈值（过滤背景噪声）
python color_classify_demo.py Pictures/原始.jpg --min-saturation 15
```

### 完整参数示例

```bash
python color_classify_demo.py Pictures/原始.jpg \
    --n-clusters 3 \
    --color-space lab \
    --auto-k-range 2 6 \
    --min-saturation 10 \
    --output results_custom \
    --debug
```

---

## 命令行参数说明

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `image` | 位置参数 | 必填 | 输入图像路径 |
| `--n-clusters` | int | None（自动） | 指定颜色聚类数 |
| `--color-space` | rgb/hsv/lab | lab | 颜色空间选择 |
| `--auto-k-range` | int int | 2 6 | 自动检测时的聚类数范围 |
| `--min-saturation` | int | 10 | 最小饱和度阈值（过滤背景） |
| `--output` | str | results | 输出目录 |
| `--debug` | flag | False | 启用调试模式 |

---

## 技术原理

### 核心算法流程

1. **EasyOCR 文字检测** → 获取所有文字区域边界框
2. **颜色特征提取**：
   - 二值化分离文字/背景（Otsu阈值）
   - 提取文字像素的中位数颜色（排除背景干扰）
   - 转换到 Lab 色彩空间（更符合人类感知）
3. **智能聚类**：
   - 自动模式：轮廓系数法选择最佳 K 值（2-6）
   - 手动模式：使用指定 K 值
   - K-Means 聚类得到颜色类别
4. **结果生成**：
   - 为每个类别创建掩码图像
   - 生成彩色标注图（不同边框颜色）
   - 输出颜色统计和色板可视化

### 颜色空间对比

| 颜色空间 | 优点 | 缺点 | 适用场景 |
|---------|------|------|---------|
| **Lab** ⭐推荐 | 符合人类视觉感知、对光照变化鲁棒 | 计算稍复杂 | 通用场景 |
| **HSV** | 色调分离明确、易于理解 | 对灰度文字不友好 | 彩色文档 |
| **RGB** | 计算简单、直观 | 不符合人类感知、对光照敏感 | 调试用途 |

### 自动聚类数选择

使用**轮廓系数（Silhouette Score）**评估聚类质量：
- 范围：[-1, 1]
- 值越大，聚类越紧密、类别越明显
- 自动选择得分最高的 K 值

---

## 实战示例

### 示例1：历史文档（黑色文字 + 蓝色批注）

```bash
python color_classify_demo.py historical_doc.jpg --debug
```

**预期输出**：
- 类别0：黑色印刷文字（60个区域）
- 类别1：蓝色手写批注（15个区域）

### 示例2：彩色表单（多种颜色）

```bash
python color_classify_demo.py colorful_form.jpg --n-clusters 4
```

**预期输出**：
- 类别0：黑色文字
- 类别1：蓝色标注
- 类别2：红色印章
- 类别3：绿色签名

### 示例3：低质量扫描件

```bash
python color_classify_demo.py low_quality_scan.jpg \
    --color-space lab \
    --min-saturation 5 \
    --auto-k-range 2 4
```

**参数说明**：
- `--color-space lab`：对光照变化更鲁棒
- `--min-saturation 5`：降低阈值，保留更多低饱和度文字
- `--auto-k-range 2 4`：限制聚类数范围（避免过度分类）

---

## 输出文件详解

### 1. 标注图 (`*_color_annotated.jpg`)

在原图上用不同颜色边框标记各类别：
- 每个文字区域用边框标记
- 不同类别使用不同颜色（彩虹色）
- 可视化检查分类效果

### 2. 类别图像 (`*_cluster_N.jpg`)

每个颜色类别的独立图像：
- 仅保留该类别的文字
- 其他区域为黑色背景
- 适合单独处理或提取

### 3. 颜色色板 (`*_color_palette.jpg`)

可视化各类别的代表颜色：
- 左侧：颜色块（聚类中心颜色）
- 右侧：类别统计信息（区域数量）

### 4. 统计信息 (`*_color_stats.json`)

JSON格式的详细统计：
```json
{
  "n_clusters": 2,
  "clusters": {
    "0": {
      "count": 62,
      "color_rgb": [182, 177, 173],
      "color_hsv": [13, 13, 182],
      "description": "gray-like"
    },
    "1": {
      "count": 16,
      "color_rgb": [118, 120, 119],
      "color_hsv": [75, 4, 120],
      "description": "gray-like"
    }
  },
  "color_space": "lab",
  "input_image": "Pictures/原始.jpg"
}
```

---

## 常见问题

### Q1: 为什么检测到的类别数比预期少？

**原因**：自动模式根据轮廓系数选择最优聚类数，可能认为2类已足够。

**解决**：
```bash
# 手动指定类别数
python color_classify_demo.py image.jpg --n-clusters 3
```

### Q2: 为什么黑白文字被分成了多个类别？

**原因**：灰度文字在颜色上有细微差异（扫描噪声、墨水深浅）。

**解决**：
```bash
# 限制聚类数范围
python color_classify_demo.py image.jpg --auto-k-range 2 3

# 或使用HSV空间（对灰度更友好）
python color_classify_demo.py image.jpg --color-space hsv
```

### Q3: 如何处理低质量扫描件（泛黄、污渍）？

**解决**：
```bash
# 降低饱和度阈值 + 使用Lab空间
python color_classify_demo.py old_doc.jpg \
    --color-space lab \
    --min-saturation 5
```

### Q4: 运行速度慢怎么办？

**原因**：EasyOCR 首次加载模型需要3-5秒。

**解决**：
- 首次运行会慢，后续处理会快
- 考虑启用 GPU 模式（修改 `utils/color_classifier.py:31`，设置 `gpu=True`）

### Q5: 如何集成到Flask应用？

参考 `app.py` 的集成方式：
```python
from utils.color_classifier import ColorClassifier

# 初始化
color_classifier = ColorClassifier(debug=False, n_clusters=3)

# 分类
result = color_classifier.classify_by_color(image_path, output_dir)
```

---

## 与现有功能对比

| 功能 | 分类依据 | 适用场景 | 灵活性 |
|------|---------|---------|--------|
| **颜色分类** | 文字颜色（自适应） | 彩色文档、多色标注 | ⭐⭐⭐⭐⭐ |
| 位置分类 | 文字位置（左/右） | 固定布局文档 | ⭐⭐⭐ |
| 表格分离 | 线条检测（Hough） | 表格文档 | ⭐⭐⭐⭐ |

**推荐组合使用**：
1. 表格分离 → 去除表格线
2. 颜色分类 → 区分不同颜色文字
3. 位置分类 → 进一步细分（如有需要）

---

## 开发与扩展

### 修改聚类算法

编辑 `utils/color_classifier.py:105-120`，修改 `find_optimal_clusters()` 方法。

### 自定义颜色描述

编辑 `utils/color_classifier.py:319-345`，修改 `_describe_color()` 方法。

### 添加新的颜色空间

在 `extract_text_color()` 方法中添加转换逻辑。

---

## 依赖环境

- Python 3.x
- OpenCV 4.6+
- NumPy 1.26+
- scikit-learn 1.7+
- EasyOCR 1.7+

已在 `font-separate` conda 环境中完整配置。

---

## 技术支持

遇到问题请查看：
1. 使用 `--debug` 模式查看详细日志
2. 检查 `results/*_color_stats.json` 了解聚类结果
3. 调整 `--color-space` 和 `--n-clusters` 参数

---

**最后更新**: 2025-10-04
**版本**: 1.0.0

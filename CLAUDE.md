# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 项目概述

**Font-Separate** 是一个文档图像智能分离系统,主要用于处理历史档案扫描件:
- **表格分离** ✅ (85%准确度,可用) - 基于 Hough 变换检测表格线,结合内容密度过滤
- **手写/印刷体分类** ⚠️ (60%准确度,需改进) - EasyOCR + 位置策略,依赖固定布局
- **颜色分类** ❌ (不适用历史文档) - K-Means 聚类,仅适用于彩色现代文档

**技术栈**: Python 3.x + Flask 3.1 + OpenCV 4.6 + EasyOCR 1.7 + scikit-learn 1.7

## 核心命令

### 环境设置
```bash
# 激活 conda 环境(项目使用 conda 管理)
conda activate font-separate  # 或 base,或请求用户创建

# 安装依赖
pip install -r requirements.txt
```

### 运行应用
```bash
# 启动 Flask 服务器(监听 0.0.0.0:5000)
python app.py

# 访问: http://localhost:5000
```

### 独立测试脚本
```bash
# 表格分离 + 手写体分类(集成测试,通过Web界面)
# 访问 http://localhost:5000 上传 Pictures/原始.jpg

# EasyOCR 文字分类(独立测试)
python classify_easyocr.py Pictures/原始.jpg

# 颜色自适应分类(独立测试,不推荐用于历史文档)
python test_scripts/color_classify_demo.py Pictures/原始.jpg

# 历史文书去噪(辅助工具)
python advanced_denoise.py Pictures/原始.jpg
```

## 架构设计

### 三重处理流程 (app.py:62-243)

**阶段1: 表格分离** (最成功的功能 ✅)
```python
TableDetector (utils/table_detector.py)
├── 1. 预处理 (23-41行): 灰度化 → Otsu 二值化
├── 2. 线条检测 (43-79行): 形态学 + Hough 变换
│   ├── 水平线核: (30,1) | 阈值30 | 最小长度30px | 最大间隙30px
│   └── 垂直线核: (1,30) | 阈值30 | 最小长度30px | 最大间隙30px
├── 3. 线条合并 (81-143行): 合并相近平行线(水平5px,垂直10px)
├── 4. 区域识别 (204-342行): 交叉点聚类 + 垂直线间隙分割(>平均×2且>80px)
└── 5. 内容密度过滤 (145-202行): 核心创新 ⭐
    ├── 空白表格: density<0.01且组件<3 → 过滤
    ├── 标题框: density>0.20且组件<30,或横线≤1且竖线≤3 → 过滤
    └── 数据表格: 其他情况 → 保留
```

**阶段2: 手写/印刷体分类** (准确度有限 ⚠️)
```python
TextClassifier (utils/text_classifier.py)
├── 1. EasyOCR 检测 (26-73行): 中英文模型,CPU模式
├── 2. 位置分类 (74-101行):
│   ├── 分界线: 图像宽度的45% (可调: split_ratio参数)
│   ├── 左侧(<45%) → 印刷体(规整表格)
│   └── 右侧(>45%) → 手写体(批注备注)
└── 3. 异常过滤: 宽高比>5或<0.2(表格线残留)

⚠️ 局限性:
- 强依赖"左侧印刷+右侧手写"的固定布局
- 无法处理混合布局文档
- 历史文档中印刷/手写特征差异小,准确度仅60%
```

**阶段3: 颜色分类** (历史文档不适用 ❌)
```python
ColorClassifier (utils/color_classifier.py)
├── 1. EasyOCR 检测文字区域
├── 2. 颜色提取 (51-87行): 二值化分离文字/背景,提取中位数颜色
├── 3. 色彩空间转换: Lab(推荐)/HSV/RGB
├── 4. K-Means 聚类 (89-143行): 自动确定K值(2-6)或手动指定
└── 5. 结果生成: 各类别掩码 + 标注图 + 色板 + JSON统计

❌ 历史文档问题:
- 墨迹严重褪色,所有颜色退化成灰色(饱和度<25)
- Otsu二值化对褪色文档失效,混入大量背景像素
- 聚类结果无意义(RGB[175,166,158]浅灰vs[118,120,119]深灰)
✅ 仅适用于彩色现代文档(黑/蓝/红笔区分印章批注)
```

### 输出结果 (results/ 目录)

**表格分离** (4个文件):
- `*_table.jpg` - 表格内容
- `*_non_table.jpg` - 非表格内容
- `*_table_annotated.jpg` - 绿色框标注
- `*_lines.jpg` - 红/蓝线条检测

**文字分类** (3个文件):
- `*_handwritten.jpg` - 手写体(红框)
- `*_printed.jpg` - 印刷体(绿框)
- `*_text_annotated.jpg` - 分类标注

**颜色分类** (N+3个文件):
- `*_color_annotated.jpg` - 彩虹色边框标注
- `*_cluster_N.jpg` - 各类别独立图像
- `*_color_palette.jpg` - 颜色色板
- `*_color_stats.json` - JSON统计

## 关键参数调整

### 表格检测不准确
编辑 `utils/table_detector.py`:
```python
# 线条检测灵敏度(51,63,69行)
threshold=30          # 降低→检测更多线条(当前已调低至30)
minLineLength=30      # 降低→检测短线条
maxLineGap=30         # 增大→连接断裂线条(历史文档专用)

# 形态学核大小(51,63行)
(30, 1) / (1, 30)     # 增大→检测更粗线条

# 表格分割阈值(250行)
gap > avg_gap*2 and gap > 80  # 避免误分割

# 内容密度阈值(188-199行)
density < 0.01        # 空白表格过滤
density > 0.20        # 标题框过滤
```

### 手写体误判
编辑 `utils/text_classifier.py`:
```python
# 位置分界线(24行)
split_ratio=0.45      # 调整为0.3-0.6,根据文档布局

# 异常框过滤(90行)
aspect_ratio > 5 or < 0.2  # 调整宽高比阈值
```

### 颜色分类(仅现代文档)
编辑 `utils/color_classifier.py`:
```python
# 聚类数(20行)
n_clusters=3          # 手动指定类别数(None=自动)

# 颜色空间(22行)
color_space='lab'     # lab(推荐)/hsv(彩色)/rgb(调试)

# 饱和度阈值(23行)
min_saturation=10     # 降低→保留更多低饱和度文字
```

## 项目结构

```
Font-Separate/
├── app.py (266行)                    # Flask主应用,三重处理流程
├── utils/
│   ├── table_detector.py (471行)     # 表格检测(Hough变换+内容密度)✅
│   ├── text_classifier.py (143行)    # 手写体分类(EasyOCR+位置)⚠️
│   ├── color_classifier.py (390行)   # 颜色分类(K-Means聚类)❌
│   └── text_extractor.py (294行)     # 辅助模块
├── test_scripts/
│   ├── color_classify_demo.py        # 颜色分类独立演示
│   ├── brightness_classify_demo.py   # 亮度分类(颜色替代方案)
│   └── historical_classify_demo.py   # 墨迹深度分类(历史文档)
├── classify_easyocr.py (161行)       # EasyOCR独立测试
├── advanced_denoise.py (238行)       # 历史文书去噪(连通组件分析)
├── static/
│   ├── css/style.css                 # 前端样式
│   └── js/main.js                    # 文件上传+结果展示
├── templates/
│   └── index.html                    # 单页应用
├── Pictures/                         # 测试样本(历史档案扫描件)
├── uploads/                          # 用户上传临时目录
└── results/                          # 处理结果输出
```

## 核心算法深度解析

### 1. 表格检测核心创新: 内容密度过滤 ⭐
```python
# utils/table_detector.py:145-202
def calculate_content_density(region):
    """
    解决问题: 初版只检测线条,误判大量空白表格/标题框

    核心思路:
    1. 计算黑色像素密度(density = black_pixels / total_area)
    2. 统计连通组件数量(cv2.connectedComponents)
    3. 统计横竖线数量

    分类规则:
    - 空白表格: density < 0.01 且 components < 3  → 过滤
    - 标题框: (density > 0.20 且 components < 30) 或
              (h_lines ≤ 1 且 v_lines ≤ 3)        → 过滤
    - 数据表格: 其他情况                          → 保留
    """
    # 实际效果: 准确度从50%提升到85% ✅
```

### 2. 手写体分类失败历程 (开发历程文档:126-234行)

**尝试1: 形态学特征分类** (提交17bdfad,失败❌)
```python
# 笔画宽度变异系数 + 圆形度 + 像素密度
准确度: 30-40%
失败原因:
- 历史文档模糊,特征不稳定
- 印刷/手写差异小
- 调到"毛笔书法级别"(CV>0.80)依然误判
```

**尝试2: EasyOCR + 位置分类** (提交9e0286e,勉强⚠️)
```python
# 代码从321行简化到93行
split_x = w * 0.45  # 分界线
center_x < split_x → 印刷体(左侧表格)
center_x > split_x → 手写体(右侧批注)

准确度: 60%
局限性:
- 强依赖"左印刷+右手写"布局
- 无法泛化到混合布局
- 需要手动调整分界线
```

**结论**: 传统方法无法解决,需要深度学习(CNN/Transformer)或人工标注

### 3. 颜色分类三次失败 (开发历程文档:625-664行)

**尝试1: 颜色聚类** (提交f33d67f,失败❌)
```python
问题: 历史文档墨迹褪色成灰色
实测: 3个类别都是gray-like,饱和度仅4-25
RGB[175,166,158]浅灰 vs [118,120,119]深灰 vs [186,187,183]浅灰
准确度: 15%
```

**尝试2: 亮度分类** (失败❌)
```python
问题: 墨迹已严重褪色,亮度范围93-221
准确度: 40%
```

**尝试3: 墨迹深度分类** (失败❌)
```python
固定阈值180 + 腐蚀操作
问题: 漏检率70%(只检测到26/89个区域)
准确度: 50%(但实际可用性差)
```

**结论**: 对历史文档完全不适用,仅适用彩色现代文档

## 常见问题

### 1. 表格检测遗漏
```bash
# 降低Hough变换阈值(当前已调至30,历史文档专用)
utils/table_detector.py:57,69行 threshold=30

# 增大maxLineGap连接断裂线条(当前已调至30)
utils/table_detector.py:59,71行 maxLineGap=30
```

### 2. 手写体误判严重
```bash
# 调整位置分界线(默认45%)
utils/text_classifier.py:24行 split_ratio=0.45

# 根据实际文档布局调整:
# - 左侧宽表格 → 降低至0.3-0.4
# - 右侧宽批注 → 提高至0.5-0.6
```

### 3. EasyOCR 初始化慢
```python
# 延迟初始化(已实现)
app.py:43-53行: 首次上传时才加载模型(3-5秒)
启动时间: <1秒

# 启用GPU加速(可选)
utils/text_classifier.py:31行: gpu=True (需要CUDA)
```

### 4. 内存溢出
```python
# 在preprocess()添加图像缩放
max_size = 2000
if max(h, w) > max_size:
    scale = max_size / max(h, w)
    img = cv2.resize(img, None, fx=scale, fy=scale)
```

## 开发建议

### 已实现功能评估 (基于实际测试,非提交注释)
1. ✅ **表格分离**(85%准确度): 生产可用,唯一成功的功能
2. ⚠️ **手写/印刷体分类**(60%): 需人工校正或深度学习重构
3. ❌ **颜色分类**(15%): 历史文档完全不适用,放弃

### 改进方向 (按优先级)

**短期(1-2周)**:
1. 实现倾斜校正(Hough变换检测角度+仿射变换)
2. 优化手写体分类:
   - 收集样本,训练轻量级CNN(MobileNet)
   - 或使用预训练Transformer模型
3. 添加表格结构化解析(提取单元格→CSV/JSON)

**中期(1-2月)**:
1. 实现批量处理(多文件上传)
2. 集成OCR识别(EasyOCR已初始化,需整合到流程)
3. 支持复杂表格(嵌套表格,斜线表头)

**长期(3-6月)**:
1. 训练专用历史文档OCR模型
2. 开发文档结构化解析引擎
3. 云部署+API服务

### 禁止操作 (遵循全局CLAUDE.md)
- ❌ **严禁执行 rm 命令** (删除文件请运行 `~/.claude/scripts/popup.fish warning`)
- ❌ 不要私自创建一堆脚本(需要时设置test_scripts/单独文件夹)
- ❌ 对于cpp代码,不要单独写CMakeLists.txt(添加到现有的)
- ❌ 尽量在原有文件基础上修改,不要自作主张新添文件

### 代理调用建议 (遵循全局CLAUDE.md)
- 优化Python代码 → `python-pro`
- 性能问题 → `performance-engineer` + `database-optimizer`(如涉及数据库)
- 代码完成后 → `code-reviewer`(必须)
- 架构变更 → `architect-reviewer`

## 测试样本说明

`Pictures/原始.jpg` 特点:
- **历史档案扫描件**: 纸张老化,污渍,破损
- **复杂布局**: 左侧印刷表格(多行多列) + 右侧手写批注(毛笔字)
- **低质量**: 低对比度,表格线模糊断裂,墨迹褪色成灰色
- **多语言**: 中文文本
- **检测挑战**:
  - 手写/印刷体特征差异小(导致分类准确度仅60%)
  - 墨迹褪色(导致颜色分类完全失效)
  - 表格线断裂(已通过增大maxLineGap=30解决)

## 技术债务

### 高优先级
1. **手写/印刷体分类需完全重构** (当前60%不可用)
   - 建议方案: 深度学习(CNN/Transformer)或人工标注训练集
2. **颜色分类对历史文档完全失效** (15%准确度)
   - 建议: 仅用于彩色现代文档,或研发专门的墨迹深度分析算法

### 中优先级
3. 倾斜校正缺失(影响倾斜扫描件)
4. 复杂表格支持有限(嵌套表格,斜线表头)

### 低优先级
5. 大图缩放优化(>2000px内存占用高)
6. GPU加速(当前CPU模式,EasyOCR初始化3-5秒)

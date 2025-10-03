# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 项目概述

**Font-Separate** 是一个文档图像智能分离系统，实现双重处理流程：
1. **表格分离**：基于 Hough 变换检测表格线，结合内容密度过滤空白表格和标题框
2. **文字分类**：使用 EasyOCR 检测文字区域，基于位置策略区分手写体和印刷体
3. **Web 界面**：Flask 单页应用，支持文件上传和结果可视化

## 技术栈

- **后端**：Python 3.x + Flask 3.1
- **图像处理**：OpenCV 4.6（含 contrib）、NumPy 1.26、Pillow 11.3、imutils 0.5
- **OCR 引擎**：EasyOCR 1.7（中英文模型）
- **机器学习**：scikit-learn 1.7、scipy 1.16
- **前端**：原生 HTML/CSS/JavaScript（无依赖框架）

## 开发命令

### 环境设置
```bash
# 激活 conda 环境（推荐使用 base 或创建专用环境）
conda activate font-separate

# 安装依赖
pip install -r requirements.txt
```

### 运行应用
```bash
# 启动 Flask 开发服务器（自动监听 0.0.0.0:5000）
python app.py

# 访问地址：http://localhost:5000
```

### 测试
```bash
# 方法1: 通过 Web 界面上传 Pictures/ 目录的样本图像
# 方法2: 直接运行测试脚本（如果存在 test_classifier.py）
python test_classifier.py

# 样本图像特点（Pictures/1.jpg, 2.jpg, 3.jpg）:
# - 历史档案扫描件，包含表格 + 手写批注 + 印章
# - 复杂布局：左侧印刷表格，右侧手写备注
```

### 项目结构
```
Font-Separate/
├── app.py                    # Flask 主应用（双重处理流程：表格+文字分类）
├── utils/
│   ├── table_detector.py     # TableDetector：表格线检测与分离
│   └── text_classifier.py    # TextClassifier：手写体/印刷体分类（EasyOCR）
├── static/
│   ├── css/style.css         # 前端样式
│   └── js/main.js            # 文件上传和结果展示逻辑
├── templates/
│   └── index.html            # 单页应用界面
├── uploads/                  # 用户上传文件临时存储
├── results/                  # 处理结果输出（8张图：表格+文字分类）
├── Pictures/                 # 测试样本（历史文档扫描件）
└── test_classifier.py        # 测试脚本（可选）
```

## 核心架构

### 双重处理流程（app.py:55-154）

**阶段1: 表格分离**
1. **文件上传**：接收用户上传的图像文件（支持 JPG/PNG/BMP/TIFF，最大 16MB）
2. **初始化检测器**：延迟初始化 `TableDetector` 和 `TextClassifier` 实例
3. **图像预处理**（table_detector.py:23-41）：
   - 转灰度图
   - Otsu 自动阈值二值化
4. **线条检测**（table_detector.py:43-79）：
   - 使用形态学操作分离水平/垂直线条（核大小 40×1 和 1×40）
   - Hough 变换检测直线（阈值 50，最小线长 50px，最大间隙 20px）
5. **线条合并**（table_detector.py:81-143）：
   - 合并相近的平行线（水平线阈值 5px，垂直线 10px）
   - 扩展线条到最大范围
6. **表格区域识别**（table_detector.py:204-342）：
   - 通过线条交叉点聚类识别独立表格区域
   - 基于垂直线间隙分割多个表格（间隙阈值：平均间隙的 2 倍且 >80px）
7. **内容密度过滤**（table_detector.py:145-202）：
   - 计算区域的黑色像素密度和连通组件数量
   - 分类并过滤：
     - 空白表格（density < 0.01 且组件 < 3）
     - 标题框（density > 0.20 且组件 < 30，或横线 ≤ 1 且竖线 ≤ 3）
     - 数据表格（其他情况，保留）
8. **表格擦除**（table_detector.py:454-471）：
   - 在原图上擦除检测到的表格线（线宽 3px）
   - 为文字分类阶段提供干净的图像

**阶段2: 手写体/印刷体分类**
1. **EasyOCR 检测**（text_classifier.py:26-73）：
   - 初始化中英文 OCR 模型（ch_sim + en，CPU 模式）
   - 检测所有文字区域的边界框和置信度
2. **位置分类策略**（text_classifier.py:74-101）：
   - 根据文字中心点位置分类（默认分界线：图像宽度的 45%）
   - 左侧 → 印刷体（规整表格内容）
   - 右侧 → 手写体（批注、备注）
   - 过滤异常宽高比（>5 或 <0.2，排除表格线残留）
3. **结果生成**（text_classifier.py:103-143）：
   - 创建手写体/印刷体掩码（彩色原图 + 掩码保留彩色信息）
   - 生成标注图像（手写红色框、印刷绿色框）

**输出结果**（app.py:126-145）
- 8 张图像文件：
  1. 原图（uploads/）
  2-5. 表格分离：table, non_table, table_annotated, lines
  6-8. 文字分类：handwritten, printed, text_annotated
- JSON 统计信息：表格数量、表格区域、手写体/印刷体数量

### 关键算法参数

**表格检测（table_detector.py）**
- **形态学核大小**：水平线 (40, 1)，垂直线 (1, 40)
- **Hough 变换参数**：阈值 50，最小线长 50px，最大间隙 20px
- **线条合并阈值**：水平线 5px，垂直线 10px
- **表格分割阈值**：垂直线间隙 >平均间隙×2 且 >80px
- **边距扩展**：30 像素（包含外框线条）
- **内容密度阈值**：
  - 空白表格：density < 0.01 且组件 < 3
  - 标题框：density > 0.20 且组件 < 30，或横线 ≤ 1 且竖线 ≤ 3
  - 数据表格：其他情况（保留）

**文字分类（text_classifier.py）**
- **位置分界线**：图像宽度的 45%（可调整）
- **异常框过滤**：宽高比 >5 或 <0.2
- **OCR 配置**：EasyOCR，语言=['ch_sim', 'en']，CPU 模式

## 扩展开发建议

### 已实现功能 ✓
1. **表格分离** ✓：Hough 变换 + 内容密度过滤
2. **手写/印刷体分类** ✓：EasyOCR + 位置策略（基于文档布局特点）

### 可优化方向（按优先级）

1. **文字分类算法改进**：
   - 当前使用简单的位置分类（左侧=印刷，右侧=手写）
   - 可改进为基于特征的分类器：
     - 笔画宽度变化（Stroke Width Transform）
     - 文字倾斜角度统计
     - 轻量级 CNN 分类器（MobileNet）
   - 调整参数：text_classifier.py:24 的 `split_ratio`

2. **表格结构化解析**：
   - 基于检测到的横竖线定位单元格
   - 提取表格内容为 CSV/JSON 格式
   - 处理合并单元格、斜线表头等复杂情况

3. **图像预处理增强**：
   - 去噪：针对老化纸张的污渍和破损
   - 倾斜校正：检测和校正扫描件倾斜
   - 对比度增强：改善低质量扫描件的识别率

4. **性能优化**：
   - EasyOCR 首次加载较慢（约 3-5 秒），考虑模型预加载
   - 大图处理：添加图像缩放（保持宽高比，长边不超过 2000px）
   - 并发处理：使用 Celery 或 multiprocessing 处理多个上传请求

### 代码修改建议

**调整表格检测参数**：
- 文件：table_detector.py
- 关键位置：
  - 线条检测阈值：51-71 行（threshold, minLineLength, maxLineGap）
  - 线条合并阈值：400-401 行（水平线 5px，垂直线 10px）
  - 内容密度阈值：188-199 行（区分空白表格、标题框、数据表格）
- 调试模式：app.py:41 修改为 `TableDetector(debug=True)` 可输出详细日志

**调整文字分类参数**：
- 文件：text_classifier.py
- 关键位置：
  - 位置分界线：24 行 `split_ratio=0.45`（默认 45%）
  - 异常框过滤：90 行（宽高比阈值 >5 或 <0.2）
- 调试模式：app.py:45 修改为 `TextClassifier(debug=True)` 可输出详细日志

**添加新检测器**：
- 在 `utils/` 创建新模块，遵循 `TableDetector` 和 `TextClassifier` 的接口设计
- 必须实现的方法：
  - `__init__(debug=False)` - 初始化
  - `detect_*()` - 检测核心算法
  - `separate_and_save()` 或 `classify_and_separate()` - 保存结果

**前端修改**：
- JavaScript 逻辑：`static/js/main.js`
- CSS 样式：`static/css/style.css`
- HTML 结构：`templates/index.html`

## 测试数据说明

`Pictures/` 目录的样本特点：
- **历史档案扫描件**：纸张老化、污渍、破损
- **复杂内容**：印刷表格 + 手写批注 + 印章
- **多语言**：包含中文文本
- **检测挑战**：低对比度、模糊线条、倾斜扫描

## 常见问题排查

**表格检测问题**：
1. **检测不到表格**：
   - 调低 Hough 变换阈值（table_detector.py:57, 69）
   - 调整形态学核大小（table_detector.py:51, 63）
   - 降低线条合并阈值（table_detector.py:400-401）

2. **误检标题框为表格**：
   - 调整内容密度阈值（table_detector.py:188-199）
   - 增加最小横线数量要求（table_detector.py:198）

3. **表格分割错误**：
   - 修改垂直线间隙阈值（table_detector.py:250，当前为平均间隙×2 且 >80px）

**文字分类问题**：
1. **手写/印刷体分类错误**：
   - 调整位置分界线比例（text_classifier.py:24，默认 0.45）
   - 检查样本图像布局是否符合"左侧印刷、右侧手写"的假设
   - 考虑使用基于特征的分类器（见扩展开发建议）

2. **误检表格线为文字**：
   - 调整异常框过滤阈值（text_classifier.py:90）
   - 检查表格分离阶段是否正确擦除表格线（table_detector.py:457-467）

**性能问题**：
1. **EasyOCR 初始化慢**：
   - 首次加载模型需要 3-5 秒（正常现象）
   - 考虑在启动时预加载（app.py:43-46 去掉延迟初始化）

2. **内存溢出**：
   - 在 `preprocess()` 添加图像缩放逻辑
   - 限制上传文件大小（app.py:16 已设为 16MB）

3. **处理速度慢**：
   - 大图（>2000px）建议先缩放
   - 考虑启用 GPU 模式（text_classifier.py:31，修改 `gpu=True`）
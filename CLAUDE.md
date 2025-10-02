# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 项目概述

**Font-Separate** 是一个文档图像分析与分离项目，基于 Flask + OpenCV 实现：
- 从历史文档扫描件中自动检测和分离表格区域
- 提供 Web 界面进行图像上传和结果可视化
- **当前实现**：基于 Hough 变换的表格线检测，结合内容密度分析过滤空白表格和标题框

## 技术栈

- **后端**：Python 3.x + Flask 3.0
- **图像处理**：OpenCV 4.8、NumPy 1.26、Pillow 10.1
- **OCR 引擎**：PaddleOCR 2.7（已安装但未集成）
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
# 使用 Pictures/ 目录下的样本图像（1.jpg, 2.jpg, 3.jpg）进行测试
# 这些是真实的历史档案扫描件，包含表格、手写批注、印章等复杂内容
```

### 项目结构
```
Font-Separate/
├── app.py                    # Flask 主应用（路由、文件上传、检测器初始化）
├── utils/
│   └── table_detector.py     # TableDetector 类：表格检测核心算法
├── static/
│   ├── css/style.css         # 前端样式
│   └── js/main.js            # 文件上传和结果展示逻辑
├── templates/
│   └── index.html            # 单页应用界面
├── uploads/                  # 用户上传文件临时存储
├── results/                  # 处理结果输出（表格分离、标注图等）
└── Pictures/                 # 测试样本（历史文档扫描件）
```

## 核心架构

### 当前实现的处理流程（app.py:46-99）

1. **文件上传**：接收用户上传的图像文件（支持 JPG/PNG/BMP/TIFF）
2. **初始化检测器**：延迟初始化 `TableDetector` 实例
3. **图像预处理**（table_detector.py:17-35）：
   - 转灰度图
   - Otsu 自动阈值二值化
4. **线条检测**（table_detector.py:37-73）：
   - 使用形态学操作分离水平/垂直线条
   - Hough 变换检测直线
5. **线条合并**（table_detector.py:75-137）：
   - 合并相近的平行线（阈值 15 像素）
   - 扩展线条到最大范围
6. **表格区域识别**（table_detector.py:197-320）：
   - 通过线条交叉点聚类识别独立表格区域
   - 基于垂直线间隙分割多个表格（间隙阈值 100px）
7. **内容密度过滤**（table_detector.py:139-195）：
   - 计算区域的黑色像素密度和连通组件数量
   - 分类并过滤：空白表格（density < 0.01）、标题框（density > 0.15 且横线 ≤ 2）
8. **结果输出**（table_detector.py:357-448）：
   - 生成 5 张图像：原图、线条检测、表格标注、表格内容、非表格内容
   - 返回 JSON 结果给前端展示

### 关键算法参数

- **形态学核大小**：水平线 (40, 1)，垂直线 (1, 40)
- **Hough 变换参数**：阈值 100，最小线长 100px，最大间隙 10px
- **线条合并阈值**：15 像素
- **表格分割阈值**：垂直线间隙 100 像素
- **内容密度阈值**：
  - 空白表格：density < 0.01，组件 < 3
  - 标题框：density > 0.15 或横线 ≤ 2
  - 数据表格：其他情况

## 扩展开发建议

### 未实现功能（按优先级）

1. **手写/印刷体分类**：
   - 可使用 Stroke Width Transform (SWT) 分析笔画特征
   - 或训练轻量级 CNN 分类器（建议使用 MobileNet）
   - 参考论文：区分规整印刷字体 vs 连续手写笔画

2. **OCR 集成**（PaddleOCR 已安装）：
   - 对分离出的表格内容进行结构化识别
   - 对非表格区域的文字进行 OCR
   - 注意中文支持：使用 `ch_PP-OCRv4` 模型

3. **表格结构化解析**：
   - 基于检测到的横竖线定位单元格
   - 提取表格内容为 CSV/JSON 格式
   - 处理合并单元格、斜线表头等复杂情况

4. **图像预处理增强**：
   - 去噪：针对老化纸张的污渍和破损
   - 倾斜校正：检测和校正扫描件倾斜
   - 对比度增强：改善低质量扫描件的识别率

### 代码修改建议

- **调整参数时**：修改 `table_detector.py` 中的类常量，而非硬编码
- **添加新检测器**：在 `utils/` 创建新模块，遵循 `TableDetector` 的接口设计
- **前端修改**：JavaScript 逻辑在 `static/js/main.js`，样式在 `static/css/style.css`
- **调试模式**：初始化 `TableDetector(debug=True)` 可输出详细日志

### 性能优化建议

- **大图处理**：考虑图像缩放（保持宽高比，长边不超过 2000px）
- **并发处理**：使用 Celery 或 multiprocessing 处理多个上传请求
- **缓存机制**：对相同图像避免重复计算（可使用文件 hash）

## 测试数据说明

`Pictures/` 目录的样本特点：
- **历史档案扫描件**：纸张老化、污渍、破损
- **复杂内容**：印刷表格 + 手写批注 + 印章
- **多语言**：包含中文文本
- **检测挑战**：低对比度、模糊线条、倾斜扫描

## 常见问题排查

1. **检测不到表格**：
   - 调低 Hough 变换阈值（table_detector.py:51, 63）
   - 调整形态学核大小（table_detector.py:45, 57）

2. **误检标题框为表格**：
   - 调整内容密度阈值（table_detector.py:182-192）
   - 增加最小横线数量要求

3. **表格分割错误**：
   - 修改垂直线间隙阈值（table_detector.py:234）

4. **内存溢出**：
   - 在 `preprocess()` 添加图像缩放逻辑
   - 限制上传文件大小（app.py:12 已设为 16MB）
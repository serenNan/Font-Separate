# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 项目概述

**Font-Separate** 是一个文档图像分析与分离项目，基于 Flask + OpenCV 实现：
- 从文稿图像中分离出表格、手写体、印刷体等不同内容类型
- 提供本地 Web 界面供用户上传和处理文档图像
- 当前实现：表格检测和分离（基于 Hough 变换的线条检测）

## 技术栈

- **后端**：Python 3.x + Flask 3.0
- **图像处理**：OpenCV 4.8、NumPy 1.26、Pillow 10.1
- **OCR 引擎**：PaddleOCR 2.7（已安装但未集成）
- **前端**：HTML + JavaScript（静态文件在 static/ 和 templates/）

## 开发命令

### 安装依赖
```bash
# 创建 conda 环境（如果需要）
conda create -n font-separate python=3.10
conda activate font-separate

# 安装依赖
pip install -r requirements.txt
```

### 运行应用
```bash
# 启动 Flask 开发服务器
python app.py

# 访问地址：http://localhost:5000
```

### 项目结构
```
Font-Separate/
├── app.py                    # Flask 主应用（路由、文件上传）
├── utils/
│   └── table_detector.py     # 表格检测核心逻辑
├── static/                   # 前端静态资源（CSS、JS）
├── templates/
│   └── index.html            # 上传界面
├── uploads/                  # 用户上传目录（临时文件）
├── results/                  # 处理结果输出目录
└── Pictures/                 # 测试图像（历史文档样本）
```

## 核心架构设计

### 处理流程
1. **图像预处理**：去噪、二值化、倾斜校正
2. **区域分割**：使用形态学操作或深度学习模型检测不同区域
3. **类型分类**：
   - 表格检测：检测表格边界和结构
   - 文字区域：区分手写与印刷
4. **内容提取**：
   - 表格：提取为结构化数据（CSV/JSON）
   - 手写体：OCR 识别或保存为独立图像
   - 印刷体：OCR 识别
5. **结果输出**：分别保存或在网页上展示

### 推荐模块结构
```
Font-Separate/
├── app.py                 # Web 应用主入口
├── requirements.txt       # Python 依赖
├── models/               # 预训练模型
├── utils/
│   ├── preprocess.py     # 图像预处理
│   ├── table_detector.py # 表格检测
│   ├── text_classifier.py # 手写/印刷分类
│   └── ocr_engine.py     # OCR 引擎封装
├── static/               # 静态资源
├── templates/            # HTML 模板
└── uploads/              # 用户上传目录
```

## 关键技术点

### 表格检测
- 使用 Hough 变换检测直线
- 或使用深度学习模型（如 CascadeTabNet、TableNet）
- 提取表格区域后进行结构化解析

### 手写/印刷分类
- 特征：笔画连续性、字体规整度、墨迹特征
- 可使用传统 CV 特征（Stroke Width Transform）
- 或使用 CNN 分类器（ResNet、EfficientNet）

### OCR 识别
- 印刷体：使用 Tesseract 或 PaddleOCR
- 手写体：需要更强的模型（PaddleOCR、TrOCR）
- 历史文档可能需要针对性训练

## 测试数据

`Pictures/` 目录包含三张示例图像：
- **1.jpg**：包含表格、手写和印刷文字的历史档案
- **2.jpg**：测试样本
- **3.jpg**：测试样本

这些是真实的历史文档扫描件，具有以下特点：
- 纸张老化、有污渍和破损
- 包含印刷表格、手写批注、印章
- 需要处理图像质量问题

## 开发注意事项

1. **图像质量处理**：示例图像显示文档有老化、污渍，需要鲁棒的预处理
2. **多语言支持**：文档包含中文，确保 OCR 模型支持中文识别
3. **表格结构复杂**：需要处理嵌套表格、斜线表头等情况
4. **手写识别难度**：历史文档的手写字体可能难以识别，考虑提供原图选项
5. **性能优化**：处理大图时需要考虑内存和速度优化

## 推荐开发步骤

1. 搭建基础 Web 框架（Flask + 上传界面）
2. 实现图像预处理模块
3. 实现表格检测（先用简单方法，再优化）
4. 实现文字区域分类（手写/印刷）
5. 集成 OCR 引擎
6. 优化和测试

## 相关资源

- PaddleOCR: https://github.com/PaddlePaddle/PaddleOCR
- Table Detection: https://github.com/poloclub/unitable
- OpenCV 表格检测教程
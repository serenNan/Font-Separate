# Font-Separate 部署与使用指南

## 🚀 快速启动

### 1. 环境准备

```bash
# 激活 conda 环境
conda activate font-separate

# 如果环境不存在，创建新环境
conda create -n font-separate python=3.10
conda activate font-separate

# 安装依赖
pip install -r requirements.txt
```

### 2. 启动服务

```bash
python app.py
```

服务将在 `http://localhost:5000` 启动。

---

## 📋 功能说明

### 核心功能

1. **表格检测与分离**
   - 基于 Hough 变换的线条检测
   - 智能表格区域识别
   - 内容密度过滤（排除空白表格和标题框）

2. **手写体/印刷体分类**
   - 使用 **EasyOCR** 进行文字检测
   - 基于**位置分类策略**（左侧=印刷表格，右侧=手写批注）
   - 自动过滤异常宽高比的区域（表格线残留）

### 技术栈

- **后端**: Flask 3.1.2
- **图像处理**: OpenCV 4.6.0.66（含 contrib 模块）
- **OCR 引擎**: EasyOCR 1.7.2（中文+英文）
- **科学计算**: NumPy 1.26.4, SciPy 1.16.2

---

## 🎯 使用流程

### Web 界面操作

1. 打开浏览器访问 `http://localhost:5000`
2. 上传图像文件（支持 JPG/PNG/BMP/TIFF，最大 16MB）
3. 等待处理完成（首次运行会下载 EasyOCR 模型，约 100MB）
4. 查看结果：
   - **线条检测**: 红色垂直线 + 蓝色水平线
   - **表格标注**: 绿色框标记表格区域
   - **表格内容**: 分离后的表格图像
   - **非表格内容**: 去除表格后的图像
   - **文字分类标注**: 红色=手写体，绿色=印刷体
   - **手写体内容**: 分离的手写体图像
   - **印刷体内容**: 分离的印刷体图像

---

## 📂 项目结构

```
Font-Separate/
├── app.py                        # Flask 主应用
├── utils/
│   ├── table_detector.py         # 表格检测器
│   └── text_classifier.py        # 文字分类器（EasyOCR + 位置分类）
├── static/
│   ├── css/style.css             # 前端样式
│   └── js/main.js                # 前端逻辑
├── templates/
│   └── index.html                # 单页应用模板
├── uploads/                      # 上传文件临时存储
├── results/                      # 处理结果输出
├── Pictures/                     # 测试样本（历史文档扫描件）
├── requirements.txt              # Python 依赖
└── DEPLOYMENT.md                 # 本文档
```

---

## 🔧 核心算法

### 表格检测算法（utils/table_detector.py）

1. **预处理**: Otsu 自动阈值二值化
2. **线条检测**: 形态学操作 + Hough 变换
3. **线条合并**: 合并相近平行线（阈值 15px）
4. **区域识别**: 线条交叉点聚类
5. **内容过滤**:
   - 空白表格: density < 0.01
   - 标题框: density > 0.15 或横线 ≤ 2

### 文字分类算法（utils/text_classifier.py）

1. **文字检测**: EasyOCR 检测所有文字区域
2. **位置分类**:
   - `x < width * 0.45` → 印刷体（左侧表格区域）
   - `x >= width * 0.45` → 手写体（右侧批注区域）
3. **异常过滤**: 跳过宽高比 > 5 或 < 0.2 的区域

---

## ⚠️ 注意事项

### 依赖版本

- **必须使用 OpenCV 4.6.0.66**（与 EasyOCR 兼容）
- **不要使用 PaddleOCR**（CPU 指令集兼容性问题 - SIGILL）
- **opencv-contrib-python** 已包含（骨架化等高级功能）

### 首次运行

- EasyOCR 首次运行会下载模型（约 100MB）
- 模型下载地址：`~/.EasyOCR/model/`
- 需要稳定的网络连接

### 性能优化

- 建议图像长边不超过 2000px
- 大图像可能导致 EasyOCR 处理时间较长（2-5 分钟）
- GPU 加速可将 `TextClassifier` 初始化改为 `gpu=True`

---

## 📊 测试结果示例

使用 `Pictures/分离目标.jpg` 测试：

```
检测结果:
- 表格区域: 1 个
- 印刷体区域: 43 个（65.4%）
- 手写体区域: 29 个（34.6%）
```

---

## 🛠️ 故障排查

### 1. ModuleNotFoundError: No module named 'easyocr'

```bash
pip install easyocr==1.7.2
```

### 2. OpenCV 版本冲突

```bash
pip uninstall opencv-python opencv-contrib-python
pip install opencv-python==4.6.0.66 opencv-contrib-python==4.6.0.66
```

### 3. EasyOCR 模型下载失败

- 检查网络连接
- 手动下载模型到 `~/.EasyOCR/model/`
- 或使用国内镜像源

### 4. Flask 无法启动

```bash
# 检查端口是否被占用
lsof -i:5000

# 更换端口
python app.py --port 8080
```

---

## 📝 开发记录

### 技术选型历程

1. **PaddleOCR** → 放弃（SIGILL 错误，CPU 不支持 AVX2/AVX512）
2. **YOLOv8 手写体检测模型** → 放弃（准确率低，误检严重）
3. **形态学特征分类** → 放弃（参数难以调优，泛化性差）
4. **EasyOCR + 位置分类** → ✅ 最终方案（稳定、兼容性好）

### 最终方案优势

- ✅ 无 CPU 指令集依赖（兼容旧机器）
- ✅ 基于实际文档布局特点（左表格右批注）
- ✅ 简单有效，易于调整参数
- ✅ 不需要训练模型或标注数据

---

## 📄 许可证

本项目仅供学习交流使用。

# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 项目概述

**Font-Separate** 是一个文档图像颜色分类系统,基于网格聚类技术区分不同颜色的文字:
- **当前功能**: 网格颜色分类 (基于 K-Means 聚类)
- **已移除功能**: 表格分离、手写/印刷体分类 (2025-10-08 用户简化需求)
- **适用场景**: 彩色现代文档 (黑/蓝/红笔区分,印章批注标记)
- **不适用**: 历史文档 (墨迹褪色成灰色,饱和度<25)

**技术栈**: Python 3.x + Flask 3.1 + OpenCV 4.6 + scikit-learn 1.7

## 核心命令

### 环境设置
```bash
# 激活 conda 环境
conda activate font-separate  # 或 base

# 安装依赖
pip install -r requirements.txt
```

### 运行应用
```bash
# 启动 Flask 服务器 (监听 0.0.0.0:5000)
python app.py

# 访问 Web 界面
# http://localhost:5000
```

### 独立测试脚本
```bash
# 网格颜色分类演示
python test_scripts/color_classify_demo.py Pictures/原始.jpg

# 色相分类 (替代方案)
python test_scripts/color_hue_classify.py Pictures/原始.jpg

# 目标颜色提取
python test_scripts/color_target_classify.py Pictures/原始.jpg

# 原始颜色检查
python test_scripts/inspect_raw_colors.py Pictures/原始.jpg
```

## 架构设计

### 核心算法: 网格颜色聚类 (utils/color_classifier.py)

```python
ColorClassifier (430行)
├── 1. 网格划分 (85-108行)
│   └── 将图像分成 grid_size×grid_size 像素网格 (默认20×20,约1mm×1mm)
├── 2. 网格颜色提取 (92-108行)
│   ├── 过滤白色像素 (RGB≥white_threshold,默认200)
│   ├── 计算每个网格的中位数颜色 (至少10个非白色像素)
│   └── 输出: 网格颜色数组 (N_grids, 3)
├── 3. HSV转换 + 饱和度过滤 (117-147行)
│   ├── RGB → HSV,提取色相H (0-180度)
│   ├── 过滤低饱和度网格 (S<min_saturation,默认30) → 灰色类
│   └── 输出: 彩色网格 + 灰色网格
├── 4. K-Means聚类 (彩色) (164-197行)
│   ├── 自动确定最佳K值 (silhouette score, K=2-6)
│   ├── 仅对色相H进行聚类
│   └── 输出: 彩色类别标签 (K个)
├── 5. 灰色二次聚类 (203-246行)
│   ├── 按明度V分为2类 (深灰/浅灰)
│   └── 输出: 总类别数 = K_colored + 2_gray
├── 6. 结果生成 (270-368行)
│   ├── 网格映射回像素级 (color_class_map)
│   ├── 生成各类别独立图像 (白色背景)
│   ├── 标注图 (网格边界,彩虹色)
│   └── 颜色色板 (可视化)
```

**核心创新**:
- **网格聚类**: 解决像素级聚类计算量大的问题 (20×20网格 → 100倍加速)
- **分层聚类**: 彩色(色相H) + 灰色(明度V) 分别处理,提高准确度
- **自动K值**: silhouette score 自动确定最佳类别数,无需手动调参

### Web应用流程 (app.py)

```python
Flask App (175行)
├── / (GET) → 渲染上传页面 (templates/index.html)
├── /upload (POST) → 处理文件上传和分类 (50-151行)
│   ├── 1. 文件验证 (58-69行)
│   ├── 2. 保存文件 (71-81行, 添加时间戳防冲突)
│   ├── 3. 延迟初始化分类器 (83-84行, 首次使用才加载)
│   ├── 4. 执行颜色分类 (88-94行)
│   ├── 5. NumPy类型转换 (98-112行, 解决JSON序列化)
│   └── 6. 返回JSON响应 (114-140行)
├── /uploads/<filename> (GET) → 提供上传文件 (154-157行)
└── /results/<filename> (GET) → 提供结果文件 (160-163行)

错误处理:
- BrokenPipeError / ConnectionResetError → 499 (客户端取消)
- 其他异常 → 500 + traceback
```

### 前端界面 (templates/index.html + static/)

```
单页应用设计:
├── 上传区 (uploadBox) - 点击或拖拽上传
├── 加载动画 (loading) - 旋转动画 + 提示文字
├── 结果展示 (results)
│   ├── 统计信息 (stats) - 类别数、颜色信息
│   ├── 图像网格 (images-grid)
│   │   ├── 原始图像
│   │   ├── 标注图像 (网格边界)
│   │   └── 颜色色板
│   └── 颜色类别展示 (colorClusters)
│       └── 各类别独立图像
└── 重新上传按钮 (newImageBtn)

交互特性:
- 文件拖拽上传 (drag & drop)
- 实时进度显示
- 响应式布局
- 错误提示
```

### 输出结果 (results/ 目录)

对于输入文件 `example.jpg`,输出以下文件:

1. **`example_color_annotated.jpg`** - 标注图 (网格边界,彩虹色标记)
2. **`example_cluster_0.jpg`** - 类别0独立图像 (白色背景)
3. **`example_cluster_1.jpg`** - 类别1独立图像
4. **`example_cluster_N.jpg`** - 类别N独立图像 (N=自动检测的类别数)
5. **`example_color_palette.jpg`** - 颜色色板 (可视化各类别颜色)

## 关键参数调整

### 网格大小调整 (utils/color_classifier.py:20)
```python
grid_size=20  # 网格大小(像素)
# 降低 → 更精细,但计算量大
# 增大 → 更快速,但精度降低
# 推荐: 10-30像素 (分辨率300dpi → 0.8-2.5mm)
```

### 白色阈值 (utils/color_classifier.py:19)
```python
white_threshold=200  # RGB≥此值为白色
# 降低 → 保留更多浅色文字 (如淡彩色)
# 增大 → 仅保留深色文字
# 推荐: 180-220
```

### 饱和度阈值 (utils/color_classifier.py:21)
```python
min_saturation=30  # 彩色/灰色分界(HSV的S通道)
# 降低 → 更多低饱和度文字被识别为彩色 (历史文档适用)
# 增大 → 更严格的彩色定义
# 推荐: 10-50
# 历史文档: 10 (墨迹饱和度<25)
```

### 聚类数调整 (utils/color_classifier.py:164-173)
当前为自动模式 (silhouette score, K=2-6)。若需手动指定:
```python
# 修改 classify_by_color() 方法
best_k = 3  # 强制3个彩色类别
# 跳过 for k in range(2, min(7, ...)) 循环
```

## 项目结构

```
Font-Separate/feature/网页完善/
├── app.py (175行)                    # Flask主应用
├── utils/
│   ├── color_classifier.py (430行)   # 网格颜色分类 (核心)
│   ├── table_detector.py (471行)     # [已废弃] 表格检测
│   ├── text_classifier.py (143行)    # [已废弃] 手写体分类
│   └── text_extractor.py (294行)     # [辅助模块]
├── test_scripts/                     # 独立测试脚本
│   ├── color_classify_demo.py        # 网格颜色分类演示
│   ├── color_hue_classify.py         # 色相分类
│   ├── color_target_classify.py      # 目标颜色提取
│   ├── color_hue_range_classify.py   # 色相范围分类
│   ├── color_cluster.py              # 聚类分析
│   └── inspect_raw_colors.py         # 原始颜色检查
├── static/
│   ├── css/style.css                 # 前端样式
│   └── js/main.js                    # 文件上传 + 结果展示
├── templates/
│   └── index.html                    # 单页应用
├── Pictures/                         # 测试样本
├── uploads/                          # 用户上传临时目录
├── results/                          # 处理结果输出
└── docs/                            # 项目文档
    ├── 开发历程文档.md               # 完整开发历程
    └── 要求.md                      # 原始需求
```

## 常见问题

### 1. 类别数检测不准确
```python
# 问题: 自动检测K值不理想
# 解决: 手动指定K值

# 修改 utils/color_classifier.py:164行
best_k = 3  # 强制3个彩色类别
# 注释掉 for k in range(2, min(7, ...)) 循环
```

### 2. 历史文档颜色分类失败
```python
# 问题: 墨迹褪色,饱和度低
# 解决: 降低饱和度阈值

# 修改 utils/color_classifier.py:21行
min_saturation=10  # 默认30 → 10

# 或在初始化时传入
classifier = ColorClassifier(min_saturation=10)
```

### 3. 内存占用过高
```python
# 问题: 大图片内存溢出
# 解决: 在 classify_by_color() 开头添加缩放

# 修改 utils/color_classifier.py:62行后添加
max_size = 2000
h, w = img.shape[:2]
if max(h, w) > max_size:
    scale = max_size / max(h, w)
    img = cv2.resize(img, None, fx=scale, fy=scale)
    h, w = img.shape[:2]
```

### 4. 网格太粗糙,细节丢失
```python
# 问题: 默认20×20网格太大
# 解决: 降低网格大小

# 初始化时传入
classifier = ColorClassifier(grid_size=10)

# 注意: grid_size↓ → 计算时间↑ (指数增长)
```

### 5. JSON序列化错误
```python
# 问题: NumPy类型无法序列化为JSON
# 解决: 已在 app.py:98-112行实现 convert_to_native()

# 若遇到新的NumPy类型错误,检查返回的字典是否经过转换
response['stats'] = convert_to_native(response['stats'])
```

### 6. 客户端断开连接错误 (499)
```python
# 问题: 用户关闭浏览器导致 BrokenPipeError
# 解决: 已在 app.py:142-147行实现异常捕获

# 不会影响服务器运行,正常现象
```

## 网页完善要点 (当前feature分支重点)

### 前端优化建议
1. **响应式设计**: 当前已支持基本响应式,可进一步优化移动端体验
2. **进度条**: 长时间处理时显示实时进度 (可通过WebSocket实现)
3. **预览功能**: 上传后立即显示缩略图,确认后再处理
4. **批量上传**: 支持一次上传多个文件
5. **下载打包**: 将所有结果打包成ZIP下载
6. **参数调整**: Web界面暴露关键参数 (grid_size, min_saturation等)

### 后端优化建议
1. **异步处理**: 使用Celery处理长时间任务,避免超时
2. **缓存机制**: 相同文件不重复处理
3. **错误恢复**: 处理失败时保留中间结果
4. **日志记录**: 详细的操作日志和性能指标
5. **API版本**: 提供RESTful API供外部调用

### 性能优化
```python
# 大图自动缩放 (添加到 app.py:83行后)
MAX_IMAGE_SIZE = 2000

def resize_if_needed(img):
    h, w = img.shape[:2]
    if max(h, w) > MAX_IMAGE_SIZE:
        scale = MAX_IMAGE_SIZE / max(h, w)
        return cv2.resize(img, None, fx=scale, fy=scale)
    return img
```

### 用户体验优化
```javascript
// 添加到 static/js/main.js
// 1. 文件大小限制提示
// 2. 拖拽区域高亮
// 3. 实时预览
// 4. 处理进度估算
// 5. 结果图片缩放查看
```

## 开发建议

### 当前功能评估
- ✅ **网格颜色分类**: 适用于彩色现代文档 (黑/蓝/红笔区分)
- ❌ **历史文档**: 墨迹褪色,饱和度<25,聚类结果无意义

### 改进方向 (网页完善方向)

**短期 (1周)**:
1. ✅ 基础Web界面 (已完成)
2. ⏳ 添加参数调整UI (grid_size, white_threshold, min_saturation)
3. ⏳ 实现文件预览功能
4. ⏳ 添加结果下载打包

**中期 (1个月)**:
1. ⏳ 批量处理功能
2. ⏳ 处理进度实时显示 (WebSocket)
3. ⏳ 结果历史记录 (数据库)
4. ⏳ 用户认证和权限管理

**长期 (3个月)**:
1. ⏳ RESTful API服务
2. ⏳ Docker容器化部署
3. ⏳ 云存储集成 (OSS)
4. ⏳ 分布式处理 (多GPU/多机)

### 已移除功能
- **表格分离** (utils/table_detector.py): 2025-10-08移除,代码仍保留但标记为DEPRECATED
- **手写/印刷体分类** (utils/text_classifier.py): 同上

若需恢复,参考 git 历史记录:
```bash
git log --all --oneline | grep "表格"
git checkout 31ad7c5  # 表格分离完善版本
```

## 测试样本说明

建议使用**彩色现代文档**测试:
- ✅ 多色笔记 (黑/蓝/红笔)
- ✅ 印章批注 (红色印章 vs 黑色文字)
- ✅ 标记重点 (黄色/绿色荧光笔)

不适用场景:
- ❌ 历史档案扫描件 (墨迹褪色)
- ❌ 黑白文档 (无颜色差异)
- ❌ 低质量扫描 (分辨率<150dpi)

测试文件位置: `Pictures/原始.jpg`

## 技术债务

### 高优先级
1. **历史文档支持**: 当前颜色聚类对褪色文档完全失效,需研发墨迹深度分析算法
2. **大图优化**: >2000px图片内存占用高,需自动缩放
3. **Web参数暴露**: 关键参数应在界面可调,无需修改代码

### 中优先级
4. 批量处理功能缺失
5. 异步任务处理 (避免超时)
6. 结果缓存机制

### 低优先级
7. 前端UI美化 (进度条,预览功能)
8. 导出格式扩展 (JSON/CSV/Excel)
9. API文档 (Swagger/OpenAPI)

## 代码规范

遵循全局 `~/.claude/CLAUDE.md` 规范:
- ❌ **严禁 rm 命令**
- ❌ 不要私自新建文件 (在原有基础上修改)
- ✅ 使用 fish 终端
- ✅ Python 虚拟环境使用 conda
- ✅ 注释使用标记: `// TODO`, `// FIXME`, `// !`, `// *`, `// ?`
- ✅ 代码完成后调用 `code-reviewer` 代理

## 相关文档

- **开发历程**: `docs/开发历程文档.md` - 完整的15次提交历程,失败经验总结
- **原始需求**: `docs/要求.md` - 最初的项目需求
- **主分支文档**: `/home/serennan/work/Font-Separate/CLAUDE.md` - 主分支同步的文档

## Git工作流

```bash
# 当前分支: feature/网页完善
git branch
# * 网页完善

# 查看变更
git status

# 提交变更
git add .
git commit -m "完善Web界面: 添加XXX功能"

# 合并到主分支前先pull
git checkout main  # 或默认分支
git pull
git merge 网页完善

# 解决冲突后推送
git push
```

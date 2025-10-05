"""
Font-Separate Web应用
文档图像分离系统 - 表格分离 + 手写体/印刷体分离
"""
from flask import Flask, render_template, request, jsonify, send_from_directory
import os
import cv2
from werkzeug.utils import secure_filename
from utils.table_detector import TableDetector
from utils.text_classifier import TextClassifier
from utils.color_classifier import ColorClassifier
import sys
sys.path.append(os.path.dirname(__file__))
import traceback
from datetime import datetime

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 最大16MB
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['RESULT_FOLDER'] = 'results'
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg', 'bmp', 'tiff'}

# 确保目录存在
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['RESULT_FOLDER'], exist_ok=True)

# 初始化检测器
table_detector = None
text_classifier = None
color_classifier = None


def allowed_file(filename):
    """检查文件扩展名是否允许"""
    return '.' in filename and \
        filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']


def init_detectors():
    """延迟初始化检测器"""
    global table_detector, text_classifier, color_classifier
    if table_detector is None:
        print("正在初始化表格检测器...")
        table_detector = TableDetector(debug=False)
        print("表格检测器初始化完成")
    if text_classifier is None:
        print("正在初始化文字分类器...")
        text_classifier = TextClassifier(debug=False)
        print("文字分类器初始化完成")
    if color_classifier is None:
        print("正在初始化颜色分类器...")
        color_classifier = ColorClassifier(debug=False, color_space='lab')
        print("颜色分类器初始化完成")


@app.route('/')
def index():
    """主页"""
    return render_template('index.html')


@app.route('/upload', methods=['POST'])
def upload_file():
    """处理文件上传、表格分离和手写体/印刷体分离"""
    try:
        # 检测客户端是否断开连接
        if request.environ.get('werkzeug.server.shutdown'):
            print("⚠ 客户端已断开连接")
            return jsonify({'error': '请求已取消'}), 499
        # 检查是否有文件
        if 'file' not in request.files:
            return jsonify({'error': '没有文件'}), 400

        file = request.files['file']

        # 检查文件名
        if file.filename == '':
            return jsonify({'error': '未选择文件'}), 400

        if not allowed_file(file.filename):
            return jsonify({'error': '不支持的文件格式'}), 400

        # 获取用户选择的处理方法
        methods = request.form.getlist('methods[]')  # ['table', 'text', 'color']
        if not methods:
            methods = ['table', 'text', 'color']  # 默认全选

        # 保存上传的文件（添加时间戳确保文件名唯一）
        original_filename = secure_filename(file.filename)
        # 分离文件名和扩展名
        name, ext = os.path.splitext(original_filename)
        # 添加时间戳
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')[:20]  # 精确到毫秒前3位
        filename = f"{name}_{timestamp}{ext}"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)

        print(f"保存文件: {original_filename} -> {filename}")
        file.save(filepath)

        # 初始化检测器（首次使用时）
        init_detectors()

        base_name = os.path.splitext(filename)[0]

        # 统计任务数
        total_tasks = len(methods)
        current_task = 0

        # 初始化结果
        table_result = None
        text_result = None
        color_result = None
        table_mask = None

        # 1. 执行表格分离（如果选中）
        if 'table' in methods:
            current_task += 1
            print(f"[{current_task}/{total_tasks}] 正在分离表格: {filename}")
            table_result = table_detector.separate_and_save(
                filepath,
                app.config['RESULT_FOLDER']
            )
            print(f"  检测到 {table_result['table_count']} 个表格区域")

            # 读取表格掩码(供文字分类使用)
            table_mask_path = os.path.join(app.config['RESULT_FOLDER'], f"{base_name}_table.jpg")
            table_mask = cv2.imread(table_mask_path, cv2.IMREAD_GRAYSCALE)
            if table_mask is not None:
                # 创建表格区域掩码(非零区域为表格)
                _, table_mask = cv2.threshold(table_mask, 10, 255, cv2.THRESH_BINARY)

        # 2. 执行手写体/印刷体分离（如果选中）
        if 'text' in methods:
            current_task += 1
            print(f"[{current_task}/{total_tasks}] 正在分类手写体和印刷体...")
            text_result = text_classifier.classify_and_separate(
                filepath,
                app.config['RESULT_FOLDER'],
                table_mask=table_mask
            )
            print(f"  手写体={text_result['handwritten_count']}, "
                  f"印刷体={text_result['printed_count']}")

        # 3. 执行颜色分类（如果选中）
        if 'color' in methods:
            current_task += 1
            print(f"[{current_task}/{total_tasks}] 正在进行颜色分类...")
            color_result = color_classifier.classify_by_color(
                filepath,
                app.config['RESULT_FOLDER']
            )
            print(f"  检测到 {color_result['n_clusters']} 个颜色类别")

        # 返回结果路径（相对路径）
        # 将 NumPy 类型转换为 Python 原生类型(解决 JSON 序列化问题)
        def convert_to_native(obj):
            """递归转换 NumPy 类型为 Python 原生类型"""
            import numpy as np
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: convert_to_native(v) for k, v in obj.items()}
            elif isinstance(obj, (list, tuple)):
                return [convert_to_native(item) for item in obj]
            else:
                return obj

        # 构建响应
        response = {
            'success': True,
            'original': f'/uploads/{filename}',
            'methods': methods,  # 返回用户选择的方法
            'stats': {}
        }

        # 添加表格分离结果
        if table_result:
            response.update({
                'table': f'/results/{base_name}_table.jpg',
                'non_table': f'/results/{base_name}_non_table.jpg',
                'table_annotated': f'/results/{base_name}_table_annotated.jpg',
                'lines': f'/results/{base_name}_lines.jpg',
            })
            response['stats'].update({
                'table_count': table_result['table_count'],
                'table_regions': table_result['table_regions']
            })

        # 添加文字分类结果
        if text_result:
            response.update({
                'handwritten': f'/results/{base_name}_handwritten.jpg',
                'printed': f'/results/{base_name}_printed.jpg',
                'text_annotated': f'/results/{base_name}_text_annotated.jpg',
            })
            response['stats'].update({
                'handwritten_count': text_result['handwritten_count'],
                'printed_count': text_result['printed_count']
            })

        # 添加颜色分类结果
        if color_result:
            response.update({
                'color_annotated': f'/results/{base_name}_color_annotated.jpg',
                'color_palette': f'/results/{base_name}_color_palette.jpg',
                'color_clusters': [f'/results/{base_name}_cluster_{i}.jpg'
                                  for i in range(color_result['n_clusters'])],
            })
            response['stats'].update({
                'color_categories': color_result['n_clusters'],
                'color_info': [
                    {
                        'type': 'color',
                        'name': info['description'],
                        'count': info['count'],
                        'rgb': info['color_rgb'],
                        'hsv': info['color_hsv']
                    }
                    for info in color_result['clusters'].values()
                ]
            })

        # 转换 NumPy 类型
        response['stats'] = convert_to_native(response['stats'])

        print(f"✓ 处理完成")
        return jsonify(response)

    except BrokenPipeError:
        print("⚠ 客户端已断开连接（BrokenPipeError）")
        return jsonify({'error': '请求已取消'}), 499
    except ConnectionResetError:
        print("⚠ 客户端已重置连接（ConnectionResetError）")
        return jsonify({'error': '请求已取消'}), 499
    except Exception as e:
        print(f"✗ 处理错误: {str(e)}")
        traceback.print_exc()
        return jsonify({'error': f'处理失败: {str(e)}'}), 500


@app.route('/uploads/<filename>')
def uploaded_file(filename):
    """提供上传的文件"""
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)


@app.route('/results/<filename>')
def result_file(filename):
    """提供结果文件"""
    return send_from_directory(app.config['RESULT_FOLDER'], filename)


if __name__ == '__main__':
    print("=" * 60)
    print("Font-Separate 文档图像智能分离系统")
    print("功能: 表格检测 + 手写体/印刷体分类 + 颜色分类")
    print("=" * 60)
    print("启动服务器...")
    print("访问地址: http://localhost:5000")
    print("=" * 60)

    app.run(debug=True, host='0.0.0.0', port=5000)
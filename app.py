"""
Font-Separate Web应用
文档图像分离系统 - 颜色分类
"""
from flask import Flask, render_template, request, jsonify, send_from_directory
import os
import cv2
from werkzeug.utils import secure_filename
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

# 初始化分类器
color_classifier = None


def allowed_file(filename):
    """检查文件扩展名是否允许"""
    return '.' in filename and \
        filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']


def init_classifier():
    """延迟初始化分类器"""
    global color_classifier
    if color_classifier is None:
        print("正在初始化颜色分类器...")
        color_classifier = ColorClassifier(debug=False)
        print("颜色分类器初始化完成")


@app.route('/')
def index():
    """主页"""
    return render_template('index.html')


@app.route('/upload', methods=['POST'])
def upload_file():
    """处理文件上传和分类"""
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

        # 初始化分类器（首次使用时）
        init_classifier()

        base_name = os.path.splitext(filename)[0]

        # 执行颜色分类
        print("[1/1] 正在进行颜色分类...")
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
            'color_annotated': f'/results/{base_name}_color_annotated.jpg',
            'color_clusters': [f'/results/{base_name}_cluster_{i}.jpg'
                              for i in range(color_result['n_clusters'])],
            'stats': {
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
            }
        }

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
    print("功能: 颜色分类")
    print("=" * 60)
    print("启动服务器...")
    print("访问地址: http://localhost:5000")
    print("=" * 60)

    app.run(debug=True, host='0.0.0.0', port=5000)
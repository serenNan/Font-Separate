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
import socket

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
    """处理文件上传和分类 - 支持批量处理"""
    try:
        # 检测客户端是否断开连接
        if request.environ.get('werkzeug.server.shutdown'):
            print("⚠ 客户端已断开连接")
            return jsonify({'error': '请求已取消'}), 499

        # 检查是否有文件
        if 'files' not in request.files:
            return jsonify({'error': '没有文件'}), 400

        files = request.files.getlist('files')

        # 检查是否有文件
        if not files or len(files) == 0:
            return jsonify({'error': '未选择文件'}), 400

        # 初始化分类器（首次使用时）
        init_classifier()

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

        # 处理所有文件
        results = []

        for idx, file in enumerate(files):
            # 检查文件名
            if file.filename == '':
                continue

            if not allowed_file(file.filename):
                results.append({
                    'success': False,
                    'filename': file.filename,
                    'error': '不支持的文件格式'
                })
                continue

            # 检查文件大小
            file.seek(0, 2)  # 移动到文件末尾
            file_size = file.tell()
            file.seek(0)  # 重置到开头

            if file_size > 16 * 1024 * 1024:
                results.append({
                    'success': False,
                    'filename': file.filename,
                    'error': '文件大小超过16MB'
                })
                continue

            # 保存上传的文件（添加时间戳确保文件名唯一）
            original_filename = secure_filename(file.filename)
            name, ext = os.path.splitext(original_filename)
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')[:20]
            filename = f"{name}_{timestamp}_{idx}{ext}"
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)

            print(f"[{idx+1}/{len(files)}] 保存文件: {original_filename} -> {filename}")
            file.save(filepath)

            base_name = os.path.splitext(filename)[0]

            try:
                # 执行颜色分类
                print(f"[{idx+1}/{len(files)}] 正在进行颜色分类...")
                color_result = color_classifier.classify_by_color(
                    filepath,
                    app.config['RESULT_FOLDER']
                )
                print(f"  检测到 {color_result['n_clusters']} 个颜色类别")

                # 构建单个文件的响应
                file_response = {
                    'success': True,
                    'filename': original_filename,
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
                file_response['stats'] = convert_to_native(file_response['stats'])

                results.append(file_response)
                print(f"[{idx+1}/{len(files)}] ✓ 处理完成")

            except Exception as e:
                print(f"[{idx+1}/{len(files)}] ✗ 处理失败: {str(e)}")
                results.append({
                    'success': False,
                    'filename': original_filename,
                    'error': f'处理失败: {str(e)}'
                })

        print(f"✓ 批量处理完成，成功 {sum(1 for r in results if r.get('success'))} 个，失败 {sum(1 for r in results if not r.get('success'))} 个")

        return jsonify({
            'success': True,
            'results': results,
            'total': len(files),
            'processed': len(results)
        })

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


def is_port_available(port, host='0.0.0.0'):
    """
    检测端口是否可用

    Args:
        port: 端口号
        host: 主机地址

    Returns:
        bool: 端口可用返回 True，否则返回 False
    """
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            s.bind((host, port))
            return True
    except OSError:
        return False


if __name__ == '__main__':
    # 检测是否是 reloader 子进程
    # WERKZEUG_RUN_MAIN 环境变量在 reloader 子进程中会被设置为 'true'
    is_reloader = os.environ.get('WERKZEUG_RUN_MAIN') == 'true'

    if not is_reloader:
        # 主进程：查找可用端口
        print("=" * 60)
        print("Font-Separate 文档图像智能分离系统")
        print("功能: 颜色分类")
        print("=" * 60)
        print("正在查找可用端口...")

        # 从5000开始查找可用端口
        port = 5000
        max_port = 5100
        found_port = None

        for current_port in range(port, max_port):
            if is_port_available(current_port):
                found_port = current_port
                print(f"✓ 找到可用端口: {found_port}")
                break
            else:
                print(f"✗ 端口 {current_port} 已被占用")

        if found_port is None:
            print(f"✗ 无法找到可用端口 (尝试范围: {port}-{max_port-1})")
            sys.exit(1)

        # 将端口号保存到环境变量，供 reloader 子进程使用
        os.environ['FLASK_SERVER_PORT'] = str(found_port)

        print("=" * 60)
        print(f"访问地址: http://localhost:{found_port}")
        print(f"局域网访问: http://<你的IP>:{found_port}")
        print("=" * 60)
        print("按 Ctrl+C 停止服务器")
        print("=" * 60)
    else:
        # reloader 子进程：从环境变量读取端口号
        found_port = int(os.environ.get('FLASK_SERVER_PORT', 5000))

    try:
        app.run(debug=True, host='0.0.0.0', port=found_port)
    except KeyboardInterrupt:
        print("\n服务器已停止")
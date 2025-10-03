"""
Font-Separate Web应用
文档图像分离系统 - 表格分离 + 手写体/印刷体分离
"""
from flask import Flask, render_template, request, jsonify, send_from_directory
import os
import cv2
from werkzeug.utils import secure_filename
from utils.table_detector import TableDetector
import sys
sys.path.append(os.path.dirname(__file__))
import traceback

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


def allowed_file(filename):
    """检查文件扩展名是否允许"""
    return '.' in filename and \
        filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']


def init_detectors():
    """延迟初始化检测器"""
    global table_detector, text_classifier
    if table_detector is None:
        print("正在初始化表格检测器...")
        table_detector = TableDetector(debug=False)
        print("表格检测器初始化完成")
    if text_classifier is None:
        print("正在初始化文字分类器...")
        text_classifier = TextClassifier(debug=False)
        print("文字分类器初始化完成")


@app.route('/')
def index():
    """主页"""
    return render_template('index.html')


@app.route('/upload', methods=['POST'])
def upload_file():
    """处理文件上传、表格分离和手写体/印刷体分离"""
    try:
        # 检查是否有文件
        if 'file' not in request.files:
            return jsonify({'error': '没有文件'}), 400

        file = request.files['file']

        # 检查文件名
        if file.filename == '':
            return jsonify({'error': '未选择文件'}), 400

        if not allowed_file(file.filename):
            return jsonify({'error': '不支持的文件格式'}), 400

        # 保存上传的文件
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        # 初始化检测器（首次使用时）
        init_detectors()

        base_name = os.path.splitext(filename)[0]

        # 1. 执行表格分离
        print(f"[1/2] 正在分离表格: {filename}")
        table_result = table_detector.separate_and_save(
            filepath,
            app.config['RESULT_FOLDER']
        )
        print(f"  检测到 {table_result['table_count']} 个表格区域")

        # 2. 执行手写体/印刷体分离
        print(f"[2/2] 正在分类手写体和印刷体...")

        # 读取表格掩码(避免重复处理表格区域)
        table_mask_path = os.path.join(app.config['RESULT_FOLDER'], f"{base_name}_table.jpg")
        table_mask = cv2.imread(table_mask_path, cv2.IMREAD_GRAYSCALE)
        if table_mask is not None:
            # 创建表格区域掩码(非零区域为表格)
            _, table_mask = cv2.threshold(table_mask, 10, 255, cv2.THRESH_BINARY)

        text_result = text_classifier.classify_and_separate(
            filepath,
            app.config['RESULT_FOLDER'],
            table_mask=table_mask
        )
        print(f"  手写体={text_result['handwritten_count']}, "
              f"印刷体={text_result['printed_count']}")

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

        response = {
            'success': True,
            'original': f'/uploads/{filename}',
            # 表格分离结果
            'table': f'/results/{base_name}_table.jpg',
            'non_table': f'/results/{base_name}_non_table.jpg',
            'table_annotated': f'/results/{base_name}_table_annotated.jpg',
            'lines': f'/results/{base_name}_lines.jpg',
            # 文字分类结果
            'handwritten': f'/results/{base_name}_handwritten.jpg',
            'printed': f'/results/{base_name}_printed.jpg',
            'text_annotated': f'/results/{base_name}_text_annotated.jpg',
            # 统计信息(转换所有 NumPy 类型)
            'stats': convert_to_native({
                'table_count': table_result['table_count'],
                'table_regions': table_result['table_regions'],
                'handwritten_count': text_result['handwritten_count'],
                'printed_count': text_result['printed_count']
            })
        }

        print(f"✓ 处理完成")
        return jsonify(response)

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
    print("功能: 表格检测 + 手写体/印刷体分类")
    print("=" * 60)
    print("启动服务器...")
    print("访问地址: http://localhost:5000")
    print("=" * 60)

    app.run(debug=True, host='0.0.0.0', port=5000)
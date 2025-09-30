"""
Font-Separate Web应用
文档图像分离系统 - 表格分离
"""
from flask import Flask, render_template, request, jsonify, send_from_directory
import os
from werkzeug.utils import secure_filename
from utils.table_detector import TableDetector
import traceback

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 最大16MB
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['RESULT_FOLDER'] = 'results'
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg', 'bmp', 'tiff'}

# 确保目录存在
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['RESULT_FOLDER'], exist_ok=True)

# 初始化表格检测器
table_detector = None


def allowed_file(filename):
    """检查文件扩展名是否允许"""
    return '.' in filename and \
        filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']


def init_detector():
    """延迟初始化检测器"""
    global table_detector
    if table_detector is None:
        print("正在初始化表格检测器...")
        table_detector = TableDetector(debug=False)
        print("检测器初始化完成")


@app.route('/')
def index():
    """主页"""
    return render_template('index.html')


@app.route('/upload', methods=['POST'])
def upload_file():
    """处理文件上传和表格分离"""
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
        init_detector()

        # 执行表格分离
        print(f"开始处理图像: {filename}")
        result = table_detector.separate_and_save(
            filepath,
            app.config['RESULT_FOLDER']
        )

        # 返回结果路径（相对路径）
        base_name = os.path.splitext(filename)[0]
        response = {
            'success': True,
            'original': f'/uploads/{filename}',
            'table': f'/results/{base_name}_table.jpg',
            'non_table': f'/results/{base_name}_non_table.jpg',
            'annotated': f'/results/{base_name}_table_annotated.jpg',
            'lines': f'/results/{base_name}_lines.jpg',
            'stats': {
                'table_count': result['table_count'],
                'table_regions': result['table_regions']
            }
        }

        print(f"处理完成: 检测到 {result['table_count']} 个表格区域")
        return jsonify(response)

    except Exception as e:
        print(f"处理错误: {str(e)}")
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
    print("=" * 50)
    print("Font-Separate 文档表格分离系统")
    print("=" * 50)
    print("启动服务器...")
    print("访问地址: http://localhost:5000")
    print("=" * 50)

    app.run(debug=True, host='0.0.0.0', port=5000)
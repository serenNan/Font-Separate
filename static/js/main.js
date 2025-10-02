// 获取DOM元素
const uploadBox = document.getElementById('uploadBox');
const fileInput = document.getElementById('fileInput');
const loading = document.getElementById('loading');
const results = document.getElementById('results');
const error = document.getElementById('error');

// 点击上传区域
uploadBox.addEventListener('click', () => {
    fileInput.click();
});

// 文件选择
fileInput.addEventListener('change', (e) => {
    if (e.target.files.length > 0) {
        handleFile(e.target.files[0]);
    }
});

// 拖拽上传
uploadBox.addEventListener('dragover', (e) => {
    e.preventDefault();
    uploadBox.classList.add('dragover');
});

uploadBox.addEventListener('dragleave', () => {
    uploadBox.classList.remove('dragover');
});

uploadBox.addEventListener('drop', (e) => {
    e.preventDefault();
    uploadBox.classList.remove('dragover');

    if (e.dataTransfer.files.length > 0) {
        handleFile(e.dataTransfer.files[0]);
    }
});

// 处理文件
async function handleFile(file) {
    // 检查文件类型
    if (!file.type.startsWith('image/')) {
        showError('请上传图像文件');
        return;
    }

    // 检查文件大小（16MB）
    if (file.size > 16 * 1024 * 1024) {
        showError('文件大小不能超过16MB');
        return;
    }

    // 隐藏其他元素，显示加载中
    uploadBox.parentElement.style.display = 'none';
    results.style.display = 'none';
    error.style.display = 'none';
    loading.style.display = 'block';

    // 创建FormData
    const formData = new FormData();
    formData.append('file', file);

    try {
        // 发送请求
        const response = await fetch('/upload', {
            method: 'POST',
            body: formData
        });

        const data = await response.json();

        if (!response.ok) {
            throw new Error(data.error || '处理失败');
        }

        // 显示结果
        displayResults(data);

    } catch (err) {
        showError(err.message);
        uploadBox.parentElement.style.display = 'block';
    } finally {
        loading.style.display = 'none';
    }
}

// 显示结果
function displayResults(data) {
    // 设置统计信息
    let statsHtml = `
        <div class="stat-item">
            <div class="stat-number">${data.stats.table_count}</div>
            <div class="stat-label">表格区域</div>
        </div>
        <div class="stat-item">
            <div class="stat-number">${data.stats.handwritten_count}</div>
            <div class="stat-label">手写体区域</div>
        </div>
        <div class="stat-item">
            <div class="stat-number">${data.stats.printed_count}</div>
            <div class="stat-label">印刷体区域</div>
        </div>
    `;

    // 添加每个表格的详细信息
    if (data.stats.table_regions && data.stats.table_regions.length > 0) {
        data.stats.table_regions.forEach((table, index) => {
            statsHtml += `
                <div class="stat-item">
                    <div class="stat-number">${table.h_lines} × ${table.v_lines}</div>
                    <div class="stat-label">表格 ${index + 1} (行×列)</div>
                </div>
            `;
        });
    }

    document.getElementById('stats').innerHTML = statsHtml;

    // 设置图像 - 表格分离结果
    document.getElementById('originalImg').src = data.original;
    document.getElementById('linesImg').src = data.lines;
    document.getElementById('annotatedImg').src = data.table_annotated;
    document.getElementById('tableImg').src = data.table;
    document.getElementById('nonTableImg').src = data.non_table;

    // 设置图像 - 文字分类结果
    document.getElementById('textAnnotatedImg').src = data.text_annotated;
    document.getElementById('handwrittenImg').src = data.handwritten;
    document.getElementById('printedImg').src = data.printed;

    // 显示结果区域
    results.style.display = 'block';
}

// 显示错误
function showError(message) {
    error.textContent = '❌ ' + message;
    error.style.display = 'block';

    // 3秒后自动隐藏
    setTimeout(() => {
        error.style.display = 'none';
    }, 3000);
}
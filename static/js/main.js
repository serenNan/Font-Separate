// 获取DOM元素
const uploadBox = document.getElementById('uploadBox');
const fileInput = document.getElementById('fileInput');
const loading = document.getElementById('loading');
const results = document.getElementById('results');
const error = document.getElementById('error');
const newImageBtn = document.getElementById('newImageBtn');

// 缓存当前上传的文件
let currentFile = null;

// AbortController 用于取消请求
let currentAbortController = null;

// 点击上传区域
uploadBox.addEventListener('click', () => {
    fileInput.click();
});

// 文件选择
fileInput.addEventListener('change', (e) => {
    if (e.target.files.length > 0) {
        const newFile = e.target.files[0];
        console.log('选择了新文件:', newFile.name);
        currentFile = newFile;
        handleFile(currentFile);
    }
});

// 监听单选框变化，自动重新处理
document.querySelectorAll('input[name="method"]').forEach(radio => {
    radio.addEventListener('change', () => {
        if (currentFile) {
            handleFile(currentFile);
        }
    });
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
        currentFile = e.dataTransfer.files[0];
        handleFile(currentFile);
    }
});

// 处理文件
async function handleFile(file) {
    console.log('handleFile 被调用，文件名:', file.name, '文件大小:', file.size);

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

    // 获取用户选择的处理方法（单选）
    const selectedRadio = document.querySelector('input[name="method"]:checked');

    if (!selectedRadio) {
        showError('请选择一种处理方法');
        return;
    }

    const method = selectedRadio.value;
    console.log('使用处理方法:', method);

    // 如果有正在进行的请求，先取消
    if (currentAbortController) {
        currentAbortController.abort();
        console.log('已取消上一次请求');
    }

    // 创建新的 AbortController
    currentAbortController = new AbortController();

    // 隐藏上传区，显示加载中（保留选项）
    document.querySelector('.method-selection').style.display = 'block';
    uploadBox.style.display = 'none';
    results.style.display = 'none';
    error.style.display = 'none';
    loading.style.display = 'block';

    // 在加载时将按钮改为"取消处理"
    newImageBtn.textContent = '取消处理';
    newImageBtn.classList.add('btn-cancel');

    // 创建FormData
    const formData = new FormData();
    formData.append('file', file);
    formData.append('methods[]', method);  // 单选方法

    try {
        // 发送请求（带取消信号）
        const response = await fetch('/upload', {
            method: 'POST',
            body: formData,
            signal: currentAbortController.signal
        });

        const data = await response.json();

        if (!response.ok) {
            throw new Error(data.error || '处理失败');
        }

        // 显示结果
        displayResults(data);

    } catch (err) {
        // 如果是主动取消，不显示错误
        if (err.name === 'AbortError') {
            console.log('请求已被取消');
            // 恢复按钮文本和样式
            newImageBtn.textContent = '处理新图片';
            newImageBtn.classList.remove('btn-cancel');
            return;
        }
        showError(err.message);
        uploadBox.style.display = 'block';
    } finally {
        loading.style.display = 'none';
        currentAbortController = null;
    }
}

// 显示结果
function displayResults(data) {
    // 设置统计信息
    let statsHtml = '';

    // 表格统计
    if (data.stats.table_count !== undefined) {
        statsHtml += `
            <div class="stat-item">
                <div class="stat-number">${data.stats.table_count}</div>
                <div class="stat-label">表格区域</div>
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
    }

    // 文字分类统计
    if (data.stats.handwritten_count !== undefined) {
        statsHtml += `
            <div class="stat-item">
                <div class="stat-number">${data.stats.handwritten_count}</div>
                <div class="stat-label">手写体区域</div>
            </div>
            <div class="stat-item">
                <div class="stat-number">${data.stats.printed_count}</div>
                <div class="stat-label">印刷体区域</div>
            </div>
        `;
    }

    // 颜色分类统计
    if (data.stats.color_categories !== undefined) {
        statsHtml += `
            <div class="stat-item">
                <div class="stat-number">${data.stats.color_categories}</div>
                <div class="stat-label">颜色类别</div>
            </div>
        `;
    }

    document.getElementById('stats').innerHTML = statsHtml;

    // 清空图像网格
    const imagesGrid = document.querySelector('.images-grid');
    imagesGrid.innerHTML = `
        <div class="image-card">
            <h3>原始图像</h3>
            <img src="${data.original}" alt="原始图像">
        </div>
    `;

    // 根据选择的方法显示对应图像
    if (data.table) {
        imagesGrid.innerHTML += `
            <div class="image-card">
                <h3>表格内容</h3>
                <img src="${data.table}" alt="表格内容">
            </div>
        `;
    }

    if (data.handwritten) {
        imagesGrid.innerHTML += `
            <div class="image-card">
                <h3>手写体内容</h3>
                <img src="${data.handwritten}" alt="手写体">
            </div>
            <div class="image-card">
                <h3>印刷体内容</h3>
                <img src="${data.printed}" alt="印刷体">
            </div>
        `;
    }

    if (data.color_annotated) {
        imagesGrid.innerHTML += `
            <div class="image-card">
                <h3>颜色分类标注</h3>
                <img src="${data.color_annotated}" alt="颜色分类标注">
            </div>
            <div class="image-card">
                <h3>颜色色板</h3>
                <img src="${data.color_palette}" alt="颜色色板">
            </div>
        `;
    }

    // 动态添加颜色聚类图像
    const colorClustersDiv = document.getElementById('colorClusters');
    colorClustersDiv.innerHTML = '';
    if (data.color_clusters && data.color_clusters.length > 0) {
        data.color_clusters.forEach((clusterUrl, index) => {
            const card = document.createElement('div');
            card.className = 'image-card';

            // 从统计信息获取颜色类别名称
            let categoryLabel = `颜色类别 ${index}`;
            if (data.stats.color_info && data.stats.color_info[index]) {
                const info = data.stats.color_info[index];
                categoryLabel = `${info.type === 'color' ? '彩色' : '灰度'} - ${info.name} (${info.count}个)`;
            }

            card.innerHTML = `
                <h3>${categoryLabel}</h3>
                <img src="${clusterUrl}" alt="颜色类别 ${index}">
            `;
            colorClustersDiv.appendChild(card);
        });
    }

    // 显示结果区域，恢复按钮为"处理新图片"
    results.style.display = 'block';
    newImageBtn.textContent = '处理新图片';
    newImageBtn.classList.remove('btn-cancel');
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

// 重置上传区
function resetUpload() {
    console.log('resetUpload 被调用');
    console.log('当前 currentFile:', currentFile ? currentFile.name : 'null');

    // 检查当前是否正在处理（按钮显示"取消处理"）
    const isProcessing = newImageBtn.textContent === '取消处理';
    console.log('是否正在处理:', isProcessing);

    if (isProcessing) {
        // 如果正在处理，取消请求但保持当前图片
        if (currentAbortController) {
            currentAbortController.abort();
            currentAbortController = null;
            console.log('已取消当前处理');
        }

        // 恢复界面状态（保留当前图片）
        loading.style.display = 'none';
        uploadBox.style.display = 'none';

        // 如果有之前的结果，显示结果；否则显示上传区
        if (results.querySelector('.images-grid').children.length > 0) {
            results.style.display = 'block';
        } else {
            uploadBox.style.display = 'block';
        }

        newImageBtn.textContent = '处理新图片';
        newImageBtn.classList.remove('btn-cancel');
    } else {
        // 如果不是正在处理，打开文件选择对话框
        // 取消可能存在的请求
        if (currentAbortController) {
            currentAbortController.abort();
            currentAbortController = null;
            console.log('已强制停止图像处理');
        }

        // 先清空 fileInput.value，确保即使选择同一文件也能触发 change 事件
        fileInput.value = '';

        // 重置状态（先清空 fileInput，再清空 currentFile）
        currentFile = null;
        results.style.display = 'none';
        loading.style.display = 'none';
        uploadBox.style.display = 'block';
        error.style.display = 'none';
        newImageBtn.textContent = '处理新图片';
        newImageBtn.classList.remove('btn-cancel');

        // 重置为默认选项（表格分离）
        document.querySelector('input[name="method"][value="table"]').checked = true;

        // 触发文件选择
        fileInput.click();
    }
}
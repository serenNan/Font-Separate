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

    // 获取用户选择的处理方法（单选）
    const selectedRadio = document.querySelector('input[name="method"]:checked');

    if (!selectedRadio) {
        showError('请选择一种处理方法');
        return;
    }

    const method = selectedRadio.value;

    // 隐藏上传区，显示加载中（保留选项）
    document.querySelector('.method-selection').style.display = 'block';
    uploadBox.style.display = 'none';
    results.style.display = 'none';
    error.style.display = 'none';
    loading.style.display = 'block';

    // 创建FormData
    const formData = new FormData();
    formData.append('file', file);
    formData.append('methods[]', method);  // 单选方法

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
        uploadBox.style.display = 'block';
    } finally {
        loading.style.display = 'none';
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

// 重置上传区
function resetUpload() {
    results.style.display = 'none';
    uploadBox.style.display = 'block';
    fileInput.value = '';
}
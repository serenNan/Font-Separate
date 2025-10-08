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

    // 只有颜色分类功能
    console.log('使用颜色分类');

    // 如果有正在进行的请求，先取消
    if (currentAbortController) {
        currentAbortController.abort();
        console.log('已取消上一次请求');
    }

    // 创建新的 AbortController
    currentAbortController = new AbortController();

    // 隐藏上传区和结果，显示加载中
    uploadBox.style.display = 'none';
    results.style.display = 'none';
    error.style.display = 'none';
    loading.style.display = 'block';

    // 创建FormData
    const formData = new FormData();
    formData.append('file', file);

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
    let statsHtml = `
        <div class="stat-item">
            <div class="stat-number">${data.stats.color_categories}</div>
            <div class="stat-label">颜色类别</div>
        </div>
    `;

    document.getElementById('stats').innerHTML = statsHtml;

    // 清空图像网格
    const imagesGrid = document.querySelector('.images-grid');
    imagesGrid.innerHTML = `
        <div class="image-card">
            <h3>原始图像</h3>
            <img src="${data.original}" alt="原始图像">
        </div>
        <div class="image-card">
            <h3>颜色分类标注</h3>
            <img src="${data.color_annotated}" alt="颜色分类标注">
        </div>
    `;

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

    // 平滑滚动到结果区域
    setTimeout(() => {
        results.scrollIntoView({ behavior: 'smooth', block: 'start' });
    }, 100);
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

    // 取消可能存在的请求
    if (currentAbortController) {
        currentAbortController.abort();
        currentAbortController = null;
        console.log('已取消当前处理');
    }

    // 先清空 fileInput.value，确保即使选择同一文件也能触发 change 事件
    fileInput.value = '';

    // 重置状态
    currentFile = null;
    results.style.display = 'none';
    loading.style.display = 'none';
    uploadBox.style.display = 'block';
    error.style.display = 'none';

    // 平滑滚动到上传区
    setTimeout(() => {
        uploadBox.scrollIntoView({ behavior: 'smooth', block: 'center' });
    }, 100);

    // 触发文件选择
    fileInput.click();
}
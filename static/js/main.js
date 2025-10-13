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

    // 收集所有图片
    const images = [];

    // 1. 原始图像
    images.push({
        url: data.original,
        title: '原始图像',
        type: 'original'
    });

    // 2. 颜色分类标注
    images.push({
        url: data.color_annotated,
        title: '颜色分类标注',
        type: 'annotated'
    });

    // 3. 颜色聚类图像
    if (data.color_clusters && data.color_clusters.length > 0) {
        data.color_clusters.forEach((clusterUrl, index) => {
            let categoryLabel = `颜色类别 ${index}`;
            if (data.stats.color_info && data.stats.color_info[index]) {
                const info = data.stats.color_info[index];
                categoryLabel = `${info.type === 'color' ? '彩色' : '灰度'} - ${info.name}`;
            }

            images.push({
                url: clusterUrl,
                title: categoryLabel,
                type: 'cluster',
                index: index
            });
        });
    }

    // 生成缩略图列表
    const thumbnailList = document.getElementById('thumbnailList');
    thumbnailList.innerHTML = '';

    images.forEach((img, index) => {
        const thumbnailItem = document.createElement('div');
        thumbnailItem.className = 'thumbnail-item';
        thumbnailItem.dataset.index = index;

        thumbnailItem.innerHTML = `
            <img src="${img.url}" alt="${img.title}">
            <div class="thumbnail-label">${img.title}</div>
        `;

        // 点击缩略图切换大图
        thumbnailItem.addEventListener('click', () => {
            showMainImage(img, index);
        });

        thumbnailList.appendChild(thumbnailItem);
    });

    // 默认显示第一张图片(原始图像)
    if (images.length > 0) {
        showMainImage(images[0], 0);
    }

    // 显示结果区域
    results.style.display = 'block';

    // 平滑滚动到结果区域
    setTimeout(() => {
        results.scrollIntoView({ behavior: 'smooth', block: 'start' });
    }, 100);
}

// 显示主图像
function showMainImage(imageData, index) {
    const mainImage = document.getElementById('mainImage');
    const currentImageTitle = document.getElementById('currentImageTitle');
    const placeholder = document.querySelector('.placeholder');

    // 更新标题
    currentImageTitle.textContent = imageData.title;

    // 隐藏占位符,显示图片
    placeholder.style.display = 'none';
    mainImage.style.display = 'block';
    mainImage.src = imageData.url;

    // 更新活动状态
    document.querySelectorAll('.thumbnail-item').forEach((item, i) => {
        if (i === index) {
            item.classList.add('active');
        } else {
            item.classList.remove('active');
        }
    });
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
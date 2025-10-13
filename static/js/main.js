const uploadBox = document.getElementById('uploadBox');
const fileInput = document.getElementById('fileInput');
const loading = document.getElementById('loading');
const loadingText = document.getElementById('loadingText');
const results = document.getElementById('results');
const error = document.getElementById('error');
const newImageBtn = document.getElementById('newImageBtn');
const sampleTabs = document.getElementById('sampleTabs');

let allSamplesData = [];
let currentSampleIndex = 0;
let currentAbortController = null;
uploadBox.addEventListener('click', () => {
    fileInput.click();
});

fileInput.addEventListener('change', (e) => {
    if (e.target.files.length > 0) {
        const files = Array.from(e.target.files);
        console.log(`选择了 ${files.length} 个文件:`, files.map(f => f.name));
        handleFiles(files);
    }
});

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
        const files = Array.from(e.dataTransfer.files);
        handleFiles(files);
    }
});

async function handleFiles(files) {
    console.log(`handleFiles 被调用，共 ${files.length} 个文件`);

    const validFiles = files.filter(file => {
        if (!file.type.startsWith('image/')) {
            console.log(`跳过非图像文件: ${file.name}`);
            return false;
        }
        if (file.size > 16 * 1024 * 1024) {
            console.log(`跳过大文件 (>16MB): ${file.name}`);
            showError(`文件 ${file.name} 大小超过16MB，已跳过`);
            return false;
        }
        return true;
    });

    if (validFiles.length === 0) {
        showError('没有有效的图像文件');
        return;
    }

    console.log(`有效文件数: ${validFiles.length}`);

    if (currentAbortController) {
        currentAbortController.abort();
        console.log('已取消上一次请求');
    }

    currentAbortController = new AbortController();

    uploadBox.style.display = 'none';
    results.style.display = 'none';
    error.style.display = 'none';
    loading.style.display = 'block';

    if (validFiles.length > 1) {
        loadingText.textContent = `正在处理 ${validFiles.length} 个图像，请稍候...`;
    } else {
        loadingText.textContent = '正在分析图像，请稍候...';
    }

    const formData = new FormData();
    validFiles.forEach(file => {
        formData.append('files', file);
    });

    try {
        const response = await fetch('/upload', {
            method: 'POST',
            body: formData,
            signal: currentAbortController.signal
        });

        const data = await response.json();

        if (!response.ok) {
            throw new Error(data.error || '处理失败');
        }

        allSamplesData = data.results.filter(r => r.success);

        if (allSamplesData.length === 0) {
            throw new Error('所有文件处理失败');
        }

        console.log(`成功处理 ${allSamplesData.length} 个文件`);

        currentSampleIndex = 0;
        displayAllResults();

    } catch (err) {
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

function displayAllResults() {
    if (allSamplesData.length > 1) {
        sampleTabs.style.display = 'flex';
        sampleTabs.innerHTML = '';

        allSamplesData.forEach((sampleData, index) => {
            const tab = document.createElement('div');
            tab.className = 'sample-tab';
            if (index === currentSampleIndex) {
                tab.classList.add('active');
            }

            tab.innerHTML = `
                <img src="${sampleData.original}" alt="${sampleData.filename}" class="sample-tab-thumbnail" title="${sampleData.filename}">
            `;

            tab.addEventListener('click', () => {
                switchSample(index);
            });

            sampleTabs.appendChild(tab);
        });
    } else {
        sampleTabs.style.display = 'none';
    }

    displayCurrentSample();
}

function switchSample(index) {
    if (index === currentSampleIndex) return;

    currentSampleIndex = index;

    document.querySelectorAll('.sample-tab').forEach((tab, i) => {
        if (i === index) {
            tab.classList.add('active');
        } else {
            tab.classList.remove('active');
        }
    });

    displayCurrentSample();
}

function displayCurrentSample() {
    const data = allSamplesData[currentSampleIndex];

    let statsHtml = `
        <div class="stat-item">
            <div class="stat-number">${data.stats.color_categories}</div>
            <div class="stat-label">颜色类别</div>
        </div>
    `;

    document.getElementById('stats').innerHTML = statsHtml;

    const images = [];

    images.push({
        url: data.original,
        title: '原始图像',
        type: 'original'
    });

    images.push({
        url: data.color_annotated,
        title: '颜色分类标注',
        type: 'annotated'
    });

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

        thumbnailItem.addEventListener('click', () => {
            showMainImage(img, index);
        });

        thumbnailList.appendChild(thumbnailItem);
    });

    if (images.length > 0) {
        showMainImage(images[0], 0);
    }

    results.style.display = 'block';

    setTimeout(() => {
        results.scrollIntoView({ behavior: 'smooth', block: 'start' });
    }, 100);
}

function showMainImage(imageData, index) {
    const mainImage = document.getElementById('mainImage');
    const currentImageTitle = document.getElementById('currentImageTitle');
    const placeholder = document.querySelector('.placeholder');

    currentImageTitle.textContent = imageData.title;

    placeholder.style.display = 'none';
    mainImage.style.display = 'block';
    mainImage.src = imageData.url;

    document.querySelectorAll('.thumbnail-item').forEach((item, i) => {
        if (i === index) {
            item.classList.add('active');
        } else {
            item.classList.remove('active');
        }
    });
}

function showError(message) {
    error.textContent = '❌ ' + message;
    error.style.display = 'block';

    setTimeout(() => {
        error.style.display = 'none';
    }, 3000);
}

function resetUpload() {
    console.log('resetUpload 被调用');

    if (currentAbortController) {
        currentAbortController.abort();
        currentAbortController = null;
        console.log('已取消当前处理');
    }

    fileInput.value = '';
    allSamplesData = [];
    currentSampleIndex = 0;
    sampleTabs.style.display = 'none';
    results.style.display = 'none';
    loading.style.display = 'none';
    uploadBox.style.display = 'block';
    error.style.display = 'none';

    setTimeout(() => {
        uploadBox.scrollIntoView({ behavior: 'smooth', block: 'center' });
    }, 100);

    fileInput.click();
}
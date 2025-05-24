// 全局变量
let currentFile = null;
let currentAnalysis = null;

// DOM 元素
const uploadForm = document.getElementById('uploadForm');
const pdfFileInput = document.getElementById('pdfFile');
const analysisResult = document.getElementById('analysisResult');
const paperTitle = document.getElementById('paperTitle');
const totalPages = document.getElementById('totalPages');
const tocList = document.getElementById('tocList');
const summarySection = document.getElementById('summarySection');
const chapterSummary = document.getElementById('chapterSummary');

// 事件监听器
uploadForm.addEventListener('submit', handleUpload);
pdfFileInput.addEventListener('change', handleFileSelect);

// 处理文件选择
function handleFileSelect(event) {
    const file = event.target.files[0];
    if (file && file.type === 'application/pdf') {
        currentFile = file;
        // 显示文件名
        const fileNameDisplay = document.createElement('p');
        fileNameDisplay.className = 'text-sm text-gray-600 mt-2';
        fileNameDisplay.textContent = `已选择: ${file.name}`;
        
        // 移除之前的文件名显示（如果存在）
        const oldDisplay = uploadForm.querySelector('.text-sm.text-gray-600');
        if (oldDisplay) {
            oldDisplay.remove();
        }
        
        uploadForm.insertBefore(fileNameDisplay, uploadForm.querySelector('button'));
    } else {
        alert('请选择PDF文件');
        event.target.value = '';
    }
}

// 更新按钮状态
function updateButtonState(button, isLoading, text) {
    button.disabled = isLoading;
    button.innerHTML = isLoading ? `<span class="loading"></span> ${text}` : text;
}

// 处理文件上传
async function handleUpload(event) {
    event.preventDefault();
    
    if (!currentFile) {
        alert('请先选择PDF文件');
        return;
    }
    
    const formData = new FormData();
    formData.append('file', currentFile);
    
    const submitButton = uploadForm.querySelector('button[type="submit"]');
    const originalText = submitButton.textContent;
    
    try {
        // 显示上传状态
        updateButtonState(submitButton, true, '上传中...');
        
        // 发送上传请求
        const response = await fetch('/upload', {
            method: 'POST',
            body: formData
        });
        
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        
        const result = await response.json();
        
        if (result.status === 'success') {
            // 更新状态为处理中
            updateButtonState(submitButton, true, '分析中...');
            
            currentAnalysis = result.analysis;
            displayAnalysis(result.analysis);
            
            // 完成后恢复按钮状态
            updateButtonState(submitButton, false, originalText);
        } else {
            throw new Error(result.message || '上传失败');
        }
    } catch (error) {
        alert('处理文件时出错: ' + error.message);
        // 发生错误时恢复按钮状态
        updateButtonState(submitButton, false, originalText);
    }
}

// 显示分析结果
function displayAnalysis(analysis) {
    // 显示基本信息
    paperTitle.textContent = `标题: ${analysis.title}`;
    totalPages.textContent = `总页数: ${analysis.total_pages}`;
    
    // 显示目录
    tocList.innerHTML = '';
    analysis.chapters.forEach((chapter, index) => {
        const chapterItem = document.createElement('div');
        chapterItem.className = 'chapter-item';
        chapterItem.innerHTML = `
            <div class="flex justify-between items-center">
                <span>${chapter.title}</span>
                <span class="text-sm text-gray-500">第 ${chapter.page_start} 页</span>
            </div>
        `;
        
        // 添加点击事件
        chapterItem.addEventListener('click', () => {
            // 移除其他章节的active类
            document.querySelectorAll('.chapter-item').forEach(item => {
                item.classList.remove('active');
            });
            
            // 添加active类到当前章节
            chapterItem.classList.add('active');
            
            // 获取章节摘要
            getChapterSummary(chapter);
        });
        
        tocList.appendChild(chapterItem);
    });
    
    // 显示分析结果区域
    analysisResult.classList.remove('hidden');
}

// 获取章节摘要
async function getChapterSummary(chapter) {
    try {
        // 显示加载状态
        chapterSummary.innerHTML = '<div class="text-center"><span class="loading"></span> 生成摘要中...</div>';
        summarySection.classList.remove('hidden');
        
        // 发送请求获取摘要
        const formData = new FormData();
        formData.append('file_path', currentFile.name);
        formData.append('chapter_title', chapter.title);
        formData.append('page_start', chapter.page_start);
        formData.append('page_end', chapter.page_end || chapter.page_start + 10);
        
        const response = await fetch('/summarize', {
            method: 'POST',
            body: formData
        });
        
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        
        const result = await response.json();
        
        if (result.status === 'success') {
            // 显示摘要
            chapterSummary.innerHTML = `
                <h3>${chapter.title}</h3>
                <div class="mt-4">${result.summary}</div>
            `;
        } else {
            throw new Error(result.message || '生成摘要失败');
        }
    } catch (error) {
        chapterSummary.innerHTML = `<div class="text-red-500">生成摘要时出错: ${error.message}</div>`;
    }
} 
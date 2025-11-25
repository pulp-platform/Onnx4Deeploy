// file-uploader.js - 文件上传组件

/**
 * 文件上传组件
 * 负责处理文件拖放和选择功能的用户界面
 */
class FileUploader {
    /**
     * 初始化文件上传组件
     */
    constructor() {
        // DOM元素
        this.dropArea = document.getElementById('drop-area');
        this.fileInputs = [
            document.getElementById('file-input-1'),
            document.getElementById('file-input-2'),
            document.getElementById('file-input-3')
        ];
        this.fileListContainer = document.getElementById('file-list');
        
        // 文件处理模块引用
        this.fileHandlerModule = null;
        
        // 文件颜色映射
        this.fileColors = ['#4CAF50', '#2196F3', '#FF9800'];
        
        // 已上传的文件信息
        this.uploadedFiles = [];
        
        // 初始化
        this.init();
    }
    
    /**
     * 初始化组件
     */
    init() {
        this.setupEventListeners();
        
        // 订阅文件加载和移除事件以更新UI
        EventBus.on('file:loaded', (fileData) => {
            this.updateUploadedFilesList(fileData);
        });
        
        EventBus.on('file:removed', (fileIndex) => {
            this.removeFileFromList(fileIndex);
        });
        
        console.log('FileUploader initialized.');
    }
    
    /**
     * 设置文件处理模块
     * @param {Object} fileHandlerModule - 文件处理模块
     */
    setFileHandler(fileHandlerModule) {
        this.fileHandlerModule = fileHandlerModule;
    }
    
    /**
     * 设置事件监听器
     */
    setupEventListeners() {
        // 防止默认拖放行为
        ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
            this.dropArea.addEventListener(eventName, (e) => {
                e.preventDefault();
                e.stopPropagation();
            }, false);
        });
        
        // 拖放区域高亮
        ['dragenter', 'dragover'].forEach(eventName => {
            this.dropArea.addEventListener(eventName, () => {
                this.dropArea.classList.add('highlight');
            }, false);
        });
        
        // 取消拖放区域高亮
        ['dragleave', 'drop'].forEach(eventName => {
            this.dropArea.addEventListener(eventName, () => {
                this.dropArea.classList.remove('highlight');
            }, false);
        });
        
        // 处理拖放文件
        this.dropArea.addEventListener('drop', (e) => {
            const dt = e.dataTransfer;
            const files = dt.files;
            
            if (files.length) {
                // 限制最多处理3个文件
                for (let i = 0; i < Math.min(files.length, 3); i++) {
                    if (files[i].name.toLowerCase().endsWith('.csv')) {
                        this.handleFileSelection(files[i], i);
                    } else {
                        this.showError(`${files[i].name} is not a CSV file.`);
                    }
                }
            }
        }, false);
        
        // 处理文件输入选择
        this.fileInputs.forEach((input, index) => {
            input.addEventListener('change', (e) => {
                if (e.target.files.length) {
                    const file = e.target.files[0];
                    if (file.name.toLowerCase().endsWith('.csv')) {
                        this.handleFileSelection(file, index);
                    } else {
                        this.showError(`${file.name} is not a CSV file.`);
                        // 重置文件输入，允许重新选择相同文件
                        e.target.value = '';
                    }
                }
            });
        });
    }
    
    /**
     * 处理文件选择
     * @param {File} file - 选择的文件
     * @param {number} fileIndex - 文件索引
     */
    handleFileSelection(file, fileIndex) {
        console.log(`Selected file ${fileIndex + 1}:`, file.name);
        
        // 检查是否有文件处理模块
        if (!this.fileHandlerModule) {
            console.error('File handler module not set.');
            this.showError('Unable to process file. Please try again later.');
            return;
        }
        
        // 调用文件处理模块处理文件
        this.fileHandlerModule.handleFile(file, fileIndex);
    }
    
    /**
     * 更新已上传文件列表
     * @param {Object} fileData - 文件数据对象
     */
    updateUploadedFilesList(fileData) {
        // 更新内部文件列表
        const existingIndex = this.uploadedFiles.findIndex(f => f.fileIndex === fileData.fileIndex);
        
        if (existingIndex >= 0) {
            // 更新现有文件
            this.uploadedFiles[existingIndex] = fileData;
        } else {
            // 添加新文件
            this.uploadedFiles.push(fileData);
        }
        
        // 刷新显示
        this.renderFileList();
    }
    
    /**
     * 从列表中移除文件
     * @param {number} fileIndex - 文件索引
     */
    removeFileFromList(fileIndex) {
        // 更新内部文件列表
        this.uploadedFiles = this.uploadedFiles.filter(f => f.fileIndex !== fileIndex);
        
        // 重置相应的文件输入，允许重新上传
        if (fileIndex >= 0 && fileIndex < this.fileInputs.length) {
            this.fileInputs[fileIndex].value = '';
        }
        
        // 刷新显示
        this.renderFileList();
    }
    
    /**
     * 渲染文件列表
     */
    renderFileList() {
        // 清空文件列表容器
        this.fileListContainer.innerHTML = '';
        
        // 为每个文件创建标记
        this.uploadedFiles.forEach(file => {
            const badge = document.createElement('div');
            badge.className = 'file-badge';
            badge.style.backgroundColor = file.color;
            
            badge.innerHTML = `
                ${file.fileName} (${file.data.length} Layers)
                <span class="remove-btn" data-file-index="${file.fileIndex}">×</span>
            `;
            
            this.fileListContainer.appendChild(badge);
            
            // 添加删除按钮的事件监听
            const removeBtn = badge.querySelector('.remove-btn');
            removeBtn.addEventListener('click', (e) => {
                const fileIndex = parseInt(e.target.getAttribute('data-file-index'));
                this.removeFile(fileIndex);
            });
        });
        
        // 如果有文件，则显示可视化容器
        const vizContainer = document.getElementById('viz-container');
        if (vizContainer) {
            vizContainer.style.display = this.uploadedFiles.length > 0 ? 'block' : 'none';
        }
    }
    
    /**
     * 移除文件
     * @param {number} fileIndex - 文件索引
     */
    removeFile(fileIndex) {
        console.log(`Removing file with index ${fileIndex}`);
        
        // 检查是否有文件处理模块
        if (!this.fileHandlerModule) {
            console.error('File handler module not set.');
            return;
        }
        
        // 调用文件处理模块移除文件
        this.fileHandlerModule.removeFile(fileIndex);
    }
    
    /**
     * 显示错误消息
     * @param {string} message - 错误消息
     */
    showError(message) {
        alert(message);
    }
    
    /**
     * 获取已上传文件数量
     * @returns {number} 文件数量
     */
    getFileCount() {
        return this.uploadedFiles.length;
    }
    
    /**
     * 重置组件
     * 清除所有上传的文件
     */
    reset() {
        // 清空上传的文件列表
        this.uploadedFiles = [];
        
        // 重置所有文件输入
        this.fileInputs.forEach(input => {
            input.value = '';
        });
        
        // 清空文件列表显示
        this.fileListContainer.innerHTML = '';
        
        // 隐藏可视化容器
        const vizContainer = document.getElementById('viz-container');
        if (vizContainer) {
            vizContainer.style.display = 'none';
        }
        
        // 对每个文件索引触发文件移除事件
        for (let i = 0; i < 3; i++) {
            EventBus.emit('file:removed', i);
        }
    }
}

// 当DOM加载完成后初始化
document.addEventListener('DOMContentLoaded', () => {
    // 文件上传组件会被main.js初始化
    // 或者在这里直接初始化：
    // window.fileUploaderComponent = new FileUploader();
});
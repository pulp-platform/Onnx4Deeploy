// file-handler.js - 文件处理模块

/**
 * 文件处理模块
 * 负责文件上传、解析和管理
 */
class FileHandler {
    /**
     * 初始化文件处理器
     */
    constructor() {
        // 存储文件颜色映射
        this.fileColors = ['#4CAF50', '#2196F3', '#FF9800'];
        
        // 初始化
        this.init();
    }
    
    /**
     * 初始化模块
     */
    init() {
        console.log('FileHandler initialized.');
    }
    
    /**
     * 处理文件上传
     * @param {File} file - 上传的文件对象
     * @param {number} fileIndex - 文件索引 (0-2)
     */
    handleFile(file, fileIndex) {
        // 确保是CSV文件
        if (!file.name.toLowerCase().endsWith('.csv')) {
            this.showError(`${file.name} is not a CSV file.`);
            return;
        }
        
        // 显示加载指示器
        this.showLoading(true);
        
        // 解析CSV文件
        this.parseCSV(file, fileIndex);
    }
    
    /**
     * 解析CSV文件
     * @param {File} file - CSV文件
     * @param {number} fileIndex - 文件索引
     */
    parseCSV(file, fileIndex) {
        // 使用PapaParse库解析CSV
        Papa.parse(file, {
            header: true,         // 第一行作为列名
            dynamicTyping: true,  // 自动转换数据类型
            skipEmptyLines: true, // 跳过空行
            complete: (results) => {
                // 解析完成回调
                console.log(`Parsed CSV ${fileIndex + 1}:`, results);
                
                // 检查数据格式是否有效
                if (!this.validateData(results.data)) {
                    this.showError(`Invalid data format in ${file.name}.`);
                    this.showLoading(false);
                    return;
                }
                
                // 处理数据成功，发出事件通知
                EventBus.emit('file:loaded', {
                    fileIndex: fileIndex,
                    fileName: file.name,
                    data: results.data,
                    color: this.fileColors[fileIndex % this.fileColors.length]
                });
                
                // 隐藏加载指示器
                this.showLoading(false);
            },
            error: (error) => {
                // 解析错误处理
                console.error(`Error parsing CSV ${fileIndex + 1}:`, error);
                this.showError(`Error parsing ${file.name}: ${error.message}`);
                this.showLoading(false);
            }
        });
    }
    
    /**
     * 验证数据格式
     * @param {Array} data - 解析后的数据
     * @returns {boolean} 是否有效
     */
    validateData(data) {
        // 检查是否有数据
        if (!data || data.length === 0) {
            return false;
        }
        
        // 检查必要的列是否存在
        const requiredColumns = ['layer_name', 'total_cycles'];
        const firstRow = data[0];
        
        for (const column of requiredColumns) {
            if (!(column in firstRow)) {
                console.error(`Required column '${column}' not found in CSV data.`);
                return false;
            }
        }
        
        return true;
    }
    
    /**
     * 移除文件
     * @param {number} fileIndex - 要移除的文件索引
     */
    removeFile(fileIndex) {
        // 发出移除文件事件
        EventBus.emit('file:removed', fileIndex);
    }
    
    /**
     * 显示/隐藏加载指示器
     * @param {boolean} show - 是否显示
     */
    showLoading(show) {
        const loadingElement = document.getElementById('loading');
        if (loadingElement) {
            loadingElement.style.display = show ? 'block' : 'none';
        }
    }
    
    /**
     * 显示错误消息
     * @param {string} message - 错误消息
     */
    showError(message) {
        alert(message);
    }
}

// 当DOM加载完成后初始化
document.addEventListener('DOMContentLoaded', () => {
    // 文件处理器会被main.js初始化
    // 或者在这里直接初始化：
    // window.fileHandlerComponent = new FileHandler();
});
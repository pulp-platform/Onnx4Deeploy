// main.js - 应用程序主入口模块

/**
 * 主应用程序类
 * 负责初始化和协调所有组件
 */
class BenchmarkVisualizationApp {
    /**
     * 初始化应用
     */
    constructor() {
        // 应用级状态
        this.dataStore = new DataStore();
        
        // 初始化组件
        this.initComponents();
        
        // 设置事件监听
        this.setupEventListeners();
        
        console.log('Benchmark Visualization Application initialized.');
    }
    
    /**
     * 初始化所有组件
     */
    initComponents() {
        // 文件处理
        this.fileHandler = new FileHandler();
        this.dataProcessor = new DataProcessor();
        
        // UI组件
        this.fileUploader = new FileUploader();
        this.tabManager = new TabManager();
        
        // 可视化组件
        this.cyclesChart = new CyclesChart();
        this.pieChart = new PieChart();
        this.dataTable = new DataTable();
        this.chartControls = new ChartControls();
        
        // 组件连接
        this.fileUploader.setFileHandler(this.fileHandler);
    }
    
    /**
     * 设置全局事件监听
     */
    setupEventListeners() {
        // 监听文件加载完成事件
        EventBus.on('file:loaded', (fileData) => {
            // 处理文件数据
            const processedData = this.dataProcessor.processData(fileData.data);
            
            // 更新数据存储
            this.dataStore.addFile({
                fileIndex: fileData.fileIndex,
                fileName: fileData.fileName,
                data: processedData,
                color: fileData.color
            });
            
            // 通知所有组件数据已更新
            this.notifyDataUpdated();
        });
        
        // 监听文件移除事件
        EventBus.on('file:removed', (fileIndex) => {
            // 从数据存储中移除
            this.dataStore.removeFile(fileIndex);
            
            // 通知所有组件数据已更新
            this.notifyDataUpdated();
        });
        
        // 处理所有数据请求
        EventBus.on('request:all-data', () => {
            // 响应数据请求
            const allData = this.dataStore.getAllData();
            EventBus.emit('response:all-data', allData);
        });
        
        // 监听图层选择事件
        EventBus.on('chart:layer-selected', (data) => {
            // 在表格中高亮显示选中的图层
            EventBus.emit('table:highlight-layer', data.layerName);
        });
    }
    
    /**
     * 通知所有组件数据已更新
     */
    notifyDataUpdated() {
        const dataFiles = this.dataStore.getFiles();
        EventBus.emit('data:updated', dataFiles);
    }
}

/**
 * 数据存储类
 * 集中管理所有数据
 */
class DataStore {
    /**
     * 初始化数据存储
     */
    constructor() {
        this.files = [];
    }
    
    /**
     * 添加文件数据
     * @param {Object} fileData - 文件数据对象
     */
    addFile(fileData) {
        // 检查是否已存在相同索引的文件
        const existingIndex = this.files.findIndex(f => f.fileIndex === fileData.fileIndex);
        
        if (existingIndex >= 0) {
            // 更新现有文件
            this.files[existingIndex] = fileData;
        } else {
            // 添加新文件
            this.files.push(fileData);
        }
    }
    
    /**
     * 移除文件
     * @param {number} fileIndex - 文件索引
     */
    removeFile(fileIndex) {
        this.files = this.files.filter(f => f.fileIndex !== fileIndex);
    }
    
    /**
     * 获取所有文件
     * @returns {Array} 文件数组
     */
    getFiles() {
        return [...this.files];
    }
    
    /**
     * 获取所有数据 (扁平化)
     * @returns {Array} 扁平化的数据数组
     */
    getAllData() {
        let allData = [];
        
        this.files.forEach(file => {
            file.data.forEach(row => {
                allData.push({
                    fileIndex: file.fileIndex,
                    fileName: file.fileName,
                    color: file.color,
                    ...row
                });
            });
        });
        
        return allData;
    }
}

// 当DOM加载完成后初始化应用
document.addEventListener('DOMContentLoaded', () => {
    // 初始化主应用
    window.app = new BenchmarkVisualizationApp();
});
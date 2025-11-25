// pie-chart.js - 饼图组件

/**
 * 饼图组件
 * 负责渲染和管理层周期分布饼图
 */
class PieChart {
    /**
     * 初始化饼图
     */
    constructor() {
        // DOM元素
        this.pieChartsContainer = document.getElementById('pie-charts-container');
        
        // 状态变量
        this.currentData = [];
        this.chartSettings = {
            maxLayers: 10, // 饼图默认显示前10个层
            sortBy: 'total_cycles',
            sortDirection: 'desc'
        };
        
        // 层类型颜色映射
        this.typeColors = {
            'Matrix Operations': '#4285F4',     // 蓝色
            'Convolution': '#EA4335',           // 红色
            'Pooling': '#FBBC05',               // 黄色
            'Activation': '#34A853',            // 绿色
            'Element-wise': '#8E44AD',          // 紫色
            'Data Movement': '#F39C12',         // 橙色
            'Normalization': '#1ABC9C',         // 青绿色
            'Regularization': '#D35400',        // 深橙色
            'Attention': '#2ECC71',             // 翠绿色
            'Embedding': '#E74C3C',             // 亮红色
            'Fully Connected': '#3498DB',       // 天蓝色
            'Optimization': '#16A085',          // 翠绿色
            'Other': '#95A5A6'                  // 灰色（未分类）
        };
        
        // 层类型颜色映射
        this.typeColors = {
            'Matrix Operations': '#4285F4',     // 蓝色
            'Convolution': '#EA4335',           // 红色
            'Pooling': '#FBBC05',               // 黄色
            'Activation': '#34A853',            // 绿色
            'Element-wise': '#8E44AD',          // 紫色
            'Data Movement': '#F39C12',         // 橙色
            'Normalization': '#1ABC9C',         // 青绿色
            'Regularization': '#D35400',        // 深橙色
            'Attention': '#2ECC71',             // 翠绿色
            'Embedding': '#E74C3C',             // 亮红色
            'Fully Connected': '#3498DB',       // 天蓝色
            'Other': '#95A5A6'                  // 灰色（未分类）
        };
        
        // 层类型颜色映射
        this.typeColors = {
            'Matrix Operations': '#4285F4',     // 蓝色
            'Convolution': '#EA4335',           // 红色
            'Pooling': '#FBBC05',               // 黄色
            'Activation': '#34A853',            // 绿色
            'Element-wise': '#8E44AD',          // 紫色
            'Data Movement': '#F39C12',         // 橙色
            'Normalization': '#1ABC9C',         // 青绿色
            'Regularization': '#D35400',        // 深橙色
            'Attention': '#2ECC71',             // 翠绿色
            'Embedding': '#E74C3C',             // 亮红色
            'Fully Connected': '#3498DB',       // 天蓝色
            'Other': '#95A5A6'                  // 灰色（未分类）
        };
        
        // 初始化
        this.init();
    }
    
    /**
     * 初始化图表
     */
    init() {
        // 订阅数据更新事件
        EventBus.on('data:updated', (dataFiles) => {
            this.currentData = dataFiles;
            this.renderCharts();
        });
        
        // 订阅设置更新，如果饼图需要响应控制面板设置
        EventBus.on('chart:settings-changed', (settings) => {
            // 只更新饼图关心的设置，如排序和最大显示数量
            if (settings.sortBy !== undefined) {
                this.chartSettings.sortBy = settings.sortBy;
            }
            if (settings.sortDirection !== undefined) {
                this.chartSettings.sortDirection = settings.sortDirection;
            }
            // 饼图的显示数量也可以单独设置
            if (settings.maxLayers !== undefined) {
                this.chartSettings.maxLayers = settings.maxLayers;
            }
            
            this.renderCharts();
        });
        
        // 添加Tab切换监听（可选）
        const tabs = document.querySelectorAll('.tab-btn');
        tabs.forEach(tab => {
            tab.addEventListener('click', () => {
                if (tab.getAttribute('data-tab') === 'pie-tab') {
                    // 切换到饼图Tab时重新渲染
                    setTimeout(() => {
                        this.renderCharts();
                    }, 100);
                }
            });
        });
        
        console.log('PieChart initialized.');
    }
    
    /**
     * 推断层类型
     * @param {string} layerName - 层名称
     * @returns {string} 层类型
     */
    inferLayerType(layerName) {
        if (!layerName) return 'Other';
        
        try {
            // 分割层名称
            const parts = layerName.split('_');
            
            // 从后向前遍历查找有意义的操作名
            // 越靠后的关键字优先级越高
            for (let i = parts.length - 1; i >= 0; i--) {
                let part = parts[i].toLowerCase();  // 转换为小写进行比较
                
                // 跳过纯数字或空部分
                if (/^\d+$/.test(part) || !part) {
                    continue;
                }
                
                // 检查是否是已知类型（精确匹配）
                if (this.layerTypeMap[part]) {
                    return this.layerTypeMap[part];
                }
                
                // 检查部分包含关键字的情况
                for (const [key, type] of Object.entries(this.layerTypeMap)) {
                    if (part.includes(key)) {
                        return type;
                    }
                }
            }
            
            // 如果从部分中找不到，则检查完整的层名称
            const lowerLayerName = layerName.toLowerCase();
            
            // 检查完整名称是否包含关键字
            for (const [key, type] of Object.entries(this.layerTypeMap)) {
                if (lowerLayerName.includes(key)) {
                    return type;
                }
            }
            
            return 'Other';
        } catch (error) {
            console.error("Error in inferLayerType:", error);
            return 'Other';
        }
    }
    
    /**
     * 渲染所有饼图
     */
    renderCharts() {
        // 清空容器
        this.pieChartsContainer.innerHTML = '';
        
        if (!this.currentData || this.currentData.length === 0) {
            this.showNoDataMessage();
            return;
        }
        
        // 为每个文件创建两个饼图：按层和按类型
        this.currentData.forEach(file => {
            // // 创建层级饼图
            // this.createLayerPieChart(file);
            
            // 创建类型饼图
            this.createTypePieChart(file);
        });
    }
    
    // /**
    //  * 为单个文件创建层级饼图（原始饼图）
    //  * @param {Object} file - 文件数据对象
    //  */
    // createLayerPieChart(file) {
    //     // 创建图表容器
    //     const chartDiv = document.createElement('div');
    //     chartDiv.className = 'chart-container';
    //     chartDiv.id = `pie-chart-layer-${file.fileIndex}`;
    //     this.pieChartsContainer.appendChild(chartDiv);
        
    //     // 根据设置排序数据
    //     const sortedData = [...file.data].sort((a, b) => {
    //         const valueA = a[this.chartSettings.sortBy] || 0;
    //         const valueB = b[this.chartSettings.sortBy] || 0;
            
    //         return this.chartSettings.sortDirection === 'asc' 
    //             ? valueA - valueB 
    //             : valueB - valueA;
    //     });
        
    //     // 获取前N个层
    //     const topLayers = sortedData.slice(0, this.chartSettings.maxLayers);
        
    //     // 准备饼图数据
    //     const values = topLayers.map(row => row[this.chartSettings.sortBy] || 0);
    //     const labels = topLayers.map(row => row.layer_name || 'Unknown');
        
    //     // 创建颜色阴影
    //     const baseColor = file.color;
    //     const colors = this.generateShades(baseColor, topLayers.length);
        
    //     // 饼图数据
    //     const data = [{
    //         values: values,
    //         labels: labels,
    //         type: 'pie',
    //         textinfo: 'percent',
    //         hoverinfo: 'label+percent+value',
    //         title: {
    //             text: `<b>${file.fileName}</b><br>Top ${this.chartSettings.maxLayers} Layers`,
    //             font: { size: 14 }
    //         },
    //         marker: { colors: colors },
    //         hole: 0.4,  // 创建环形图，更现代的外观
    //         textposition: 'outside',
    //         textfont: {
    //             size: 11
    //         },
    //         automargin: true
    //     }];
        
    //     // 布局设置
    //     const layout = {
    //         height: 400,
    //         margin: { t: 60, b: 20, l: 20, r: 100 },  // 增加右侧margin为legend留出空间
    //         showlegend: true,
    //         legend: { 
    //             font: { size: 10 },
    //             orientation: 'v',
    //             x: 1.1,    // 将legend向右移动，与图表拉开距离
    //             y: 1,
    //             xanchor: 'left',  // 修改为left，确保legend不会重叠
    //             yanchor: 'top',
    //             tracegroupgap: 5  // 减少legend项目之间的间距
    //         }
    //     };
        
    //     // 渲染图表
    //     Plotly.newPlot(`pie-chart-layer-${file.fileIndex}`, data, layout);
        
    //     // 添加点击事件
    //     document.getElementById(`pie-chart-layer-${file.fileIndex}`).on('plotly_click', (data) => {
    //         if (data.points && data.points.length > 0) {
    //             const point = data.points[0];
    //             const layerName = point.label;
                
    //             // 发出事件通知其他组件
    //             EventBus.emit('chart:layer-selected', {
    //                 layerName: layerName
    //             });
    //         }
    //     });
    // }
    
    /**
     * 为单个文件创建类型饼图（按层类型分组）
     * @param {Object} file - 文件数据对象
     */
    createTypePieChart(file) {
        // 创建图表容器
        const chartDiv = document.createElement('div');
        chartDiv.className = 'chart-container';
        chartDiv.id = `pie-chart-type-${file.fileIndex}`;
        this.pieChartsContainer.appendChild(chartDiv);
        
        // 按类型分组数据
        const typeGroups = {};
        const layersByType = {};
        
        // 遍历所有层，按类型分组
        file.data.forEach(row => {
            // 使用表格中的type列作为类型标识，如果没有则使用'Other'
            const layerName = row.layer_name || 'Unknown';
            const layerType = row.type ? row.type.toLowerCase() : 'other';
            const value = row[this.chartSettings.sortBy] || 0;
            
            // 确定最终分类
            let finalType = this.getCategoryForType(layerType);
            
            // 累加类型值
            if (!typeGroups[finalType]) {
                typeGroups[finalType] = 0;
                layersByType[finalType] = [];
            }
            
            typeGroups[finalType] += value;
            layersByType[finalType].push({
                name: layerName,
                value: value
            });
        });
        
        // 转换为饼图数据格式
        const types = Object.keys(typeGroups);
        const values = types.map(type => typeGroups[type]);
        const colors = types.map(type => this.typeColors[type] || this.typeColors['Other']);
        
        // 为每种类型准备悬停信息
        const customData = types.map(type => {
            // 按值排序
            const sortedLayers = layersByType[type].sort((a, b) => b.value - a.value);
            // 获取前3个最大的层
            const topLayers = sortedLayers.slice(0, 3);
            return topLayers.map(layer => `${layer.name}: ${layer.value.toLocaleString()}`).join('<br>');
        });
        
        // 饼图数据
        const data = [{
            values: values,
            labels: types,
            type: 'pie',
            textinfo: 'percent',
            hoverinfo: 'label+percent+value+text',
            hovertext: customData,
            title: {
                text: `<b>${file.fileName}</b><br>By Layer Type`,
                font: { size: 14 }
            },
            marker: { colors: colors },
            hole: 0.4,
            textposition: 'outside',
            textfont: {
                size: 11
            },
            automargin: true
        }];
        
        // 布局设置
        const layout = {
            height: 400,
            margin: { t: 60, b: 20, l: 20, r: 100 },  // 增加右侧margin为legend留出空间
            showlegend: true,
            legend: { 
                font: { size: 10 },
                orientation: 'v',
                x: 1.1,    // 将legend向右移动，与图表拉开距离
                y: 1,
                xanchor: 'left',  // 修改为left，确保legend不会重叠
                yanchor: 'top',
                tracegroupgap: 5  // 减少legend项目之间的间距
            }
        };
        
        // 渲染图表
        Plotly.newPlot(`pie-chart-type-${file.fileIndex}`, data, layout);
        
        // 添加点击事件
        document.getElementById(`pie-chart-type-${file.fileIndex}`).on('plotly_click', (data) => {
            if (data.points && data.points.length > 0) {
                const point = data.points[0];
                const layerType = point.label;
                
                // 发出事件通知其他组件
                EventBus.emit('chart:type-selected', {
                    layerType: layerType,
                    layersByType: layersByType[layerType]
                });
            }
        });
    }
    
    /**
     * 获取类型的分类
     * 将原始type归类到主要分类中
     * @param {string} type - 原始类型
     * @returns {string} 归类后的类型
     */
    getCategoryForType(type) {
        // 映射表格中的type列到我们的分类
        const typeMapping = {
            'conv': 'Convolution',
            'gemm': 'Matrix Operations',
            'matmul': 'Matrix Operations',
            'maxpool': 'Pooling',
            'avgpool': 'Pooling',
            'pool': 'Pooling',
            'relu': 'Activation',
            'sigmoid': 'Activation',
            'softmax': 'Activation',
            'tanh': 'Activation',
            'gelu': 'Activation',
            'add': 'Element-wise',
            'mul': 'Element-wise',
            'div': 'Element-wise',
            'sub': 'Element-wise',
            'transpose': 'Data Movement',
            'reshape': 'Data Movement',
            'concat': 'Data Movement',
            'split': 'Data Movement',
            'layernorm': 'Normalization',
            'batchnorm': 'Normalization',
            'norm': 'Normalization',
            'sgd': 'sgd',
            'other': 'Other'
        };
        
        return typeMapping[type] || 'Other';
    }
    
    /**
     * 生成颜色阴影
     * @param {string} baseColor - 基础颜色 (HEX格式)
     * @param {number} count - 需要的颜色数量
     * @returns {Array} 颜色数组
     */
    generateShades(baseColor, count) {
        const shades = [];
        
        // 解析RGB值
        let r = parseInt(baseColor.slice(1, 3), 16);
        let g = parseInt(baseColor.slice(3, 5), 16);
        let b = parseInt(baseColor.slice(5, 7), 16);
        
        for (let i = 0; i < count; i++) {
            // 创建由深到浅的颜色变化
            const factor = 0.5 + (i * 0.5 / count);
            const newR = Math.min(255, Math.floor(r * factor));
            const newG = Math.min(255, Math.floor(g * factor));
            const newB = Math.min(255, Math.floor(b * factor));
            
            // 转换为HEX格式
            const hexR = newR.toString(16).padStart(2, '0');
            const hexG = newG.toString(16).padStart(2, '0');
            const hexB = newB.toString(16).padStart(2, '0');
            
            shades.push(`#${hexR}${hexG}${hexB}`);
        }
        
        return shades;
    }
    
    /**
     * 显示无数据消息
     */
    showNoDataMessage() {
        this.pieChartsContainer.innerHTML = `
            <div class="no-data-message">
                <div>No data available to display pie charts.</div>
                <div>Please upload at least one CSV file.</div>
            </div>
        `;
    }
}

// 当DOM加载完成后初始化
document.addEventListener('DOMContentLoaded', () => {
    // 饼图组件会被main.js初始化
    // 或者在这里直接初始化：
    // window.pieChartComponent = new PieChart();
});
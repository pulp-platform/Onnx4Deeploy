// chart-controls.js - 仅保留柱状图选项的完整版本

/**
 * 图表控制组件
 * 负责管理各种图表视图的控制选项 - 仅保留柱状图功能
 */
class ChartControls {
    /**
     * 初始化图表控制
     */
    constructor() {
        // DOM元素
        this.sortBySelector = document.getElementById('sort-by');
        this.sortDirectionBtn = document.getElementById('sort-direction');
        this.maxLayersSlider = document.getElementById('max-layers');
        this.maxLayersValue = document.getElementById('max-layers-value');
        this.yScaleTypeSelector = document.getElementById('y-scale-type');
        this.exportChartBtn = document.getElementById('export-chart');
        this.exportDataBtn = document.getElementById('export-data');
        
        // 状态变量
        this.chartSettings = {
            chartType: 'bar',  // 固定为柱状图
            sortBy: 'total_cycles',
            sortDirection: 'desc',
            maxLayers: 25,
            yScaleType: 'linear'
        };
        
        // 初始化
        this.init();
    }
    
    /**
     * 初始化控件
     */
    init() {
        this.setupEventListeners();
        
        // 监听表格排序变更事件
        EventBus.on('table:sort-changed', (data) => {
            this.chartSettings.sortBy = data.column;
            this.chartSettings.sortDirection = data.direction;
            this.updateControlsState();
        });
        
        console.log('ChartControls initialized with bar chart only.');
    }
    
    /**
     * 设置事件监听器
     */
    setupEventListeners() {
        // 排序字段选择
        this.sortBySelector.addEventListener('change', () => {
            this.chartSettings.sortBy = this.sortBySelector.value;
            this.notifySettingsChanged();
            
            // 通知排序控件变更
            EventBus.emit('controls:sort-changed', {
                column: this.chartSettings.sortBy,
                direction: this.chartSettings.sortDirection
            });
        });
        
        // 排序方向切换
        this.sortDirectionBtn.addEventListener('click', () => {
            this.chartSettings.sortDirection = 
                this.chartSettings.sortDirection === 'asc' ? 'desc' : 'asc';
            
            // 更新箭头图标
            this.sortDirectionBtn.querySelector('span').textContent = 
                this.chartSettings.sortDirection === 'asc' ? '↑' : '↓';
            
            this.notifySettingsChanged();
            
            // 通知排序控件变更
            EventBus.emit('controls:sort-changed', {
                column: this.chartSettings.sortBy,
                direction: this.chartSettings.sortDirection
            });
        });
        
        // 最大层数滑块
        this.maxLayersSlider.addEventListener('input', () => {
            this.chartSettings.maxLayers = parseInt(this.maxLayersSlider.value);
            this.maxLayersValue.textContent = this.chartSettings.maxLayers;
            this.notifySettingsChanged();
        });
        
        // Y轴比例类型
        this.yScaleTypeSelector.addEventListener('change', () => {
            this.chartSettings.yScaleType = this.yScaleTypeSelector.value;
            this.notifySettingsChanged();
        });
        
        // 导出图表
        this.exportChartBtn.addEventListener('click', () => {
            this.exportChart();
        });
        
        // 导出数据
        this.exportDataBtn.addEventListener('click', () => {
            this.exportData();
        });
    }
    
    /**
     * 更新控件状态
     */
    updateControlsState() {
        this.sortBySelector.value = this.chartSettings.sortBy;
        this.sortDirectionBtn.querySelector('span').textContent = 
            this.chartSettings.sortDirection === 'asc' ? '↑' : '↓';
        this.maxLayersSlider.value = this.chartSettings.maxLayers;
        this.maxLayersValue.textContent = this.chartSettings.maxLayers;
        this.yScaleTypeSelector.value = this.chartSettings.yScaleType;
    }
    
    /**
     * 通知设置变更
     */
    notifySettingsChanged() {
        // 始终将chartType设为'bar'
        this.chartSettings.chartType = 'bar';
        
        // 发出设置变更事件
        EventBus.emit('chart:settings-changed', this.chartSettings);
    }
    
    /**
     * 导出当前图表为图片
     */
    exportChart() {
        // 获取当前激活的tab
        const activeTab = document.querySelector('.tabcontent.visible');
        const chartElement = activeTab.querySelector('[id$="-chart"]');
        
        if (!chartElement) {
            alert('No chart available to export');
            return;
        }
        
        // 使用Plotly的toImage功能
        if (window.Plotly) {
            Plotly.toImage(chartElement, {
                format: 'png',
                width: 1200,
                height: 800,
                scale: 2  // 高分辨率
            })
            .then(function(dataUrl) {
                // 创建下载链接
                const link = document.createElement('a');
                link.href = dataUrl;
                link.download = 'benchmark-chart.png';
                link.click();
            })
            .catch(function(err) {
                console.error('Error exporting chart:', err);
                alert('Failed to export chart: ' + err.message);
            });
        } else {
            alert('Plotly library not available for export');
        }
    }
    
    /**
     * 导出数据为CSV
     */
    exportData() {
        // 获取所有数据
        EventBus.emit('request:all-data', {});
        
        // 监听一次性响应
        EventBus.once('response:all-data', (allData) => {
            if (!allData || !allData.length) {
                alert('No data available to export');
                return;
            }
            
            try {
                // 获取CSV头部（列名）
                const headers = [
                    'File Name',
                    'Layer Name',
                    'L2 Input DMA',
                    'L2 Computation',
                    'L2 Output DMA',
                    'L3 Input DMA',
                    'L3 Output DMA',
                    'Total Cycles',
                    'Ops',
                    'Ops/Cycle'  // 修改为Ops/Cycle
                ];
                
                // 构建CSV行
                let csvContent = headers.join(',') + '\n';
                
                allData.forEach(row => {
                    const efficiency = row.total_cycles > 0 
                        ? (row.ops / row.total_cycles).toFixed(4)  // 修改为Ops/Cycle计算方式
                        : 'N/A';
                    
                    const csvRow = [
                        `"${row.fileName || ''}"`,
                        `"${row.layer_name || 'Unknown'}"`,
                        row.L2_input_dma || 0,
                        row.L2_computation || 0,
                        row.L2_output_dma || 0,
                        row.L3_input_dma || 0,
                        row.L3_output_dma || 0,
                        row.total_cycles || 0,
                        row.ops || 0,
                        efficiency
                    ];
                    
                    csvContent += csvRow.join(',') + '\n';
                });
                
                // 创建Blob并下载
                const blob = new Blob([csvContent], { type: 'text/csv;charset=utf-8;' });
                const url = URL.createObjectURL(blob);
                const link = document.createElement('a');
                link.href = url;
                link.setAttribute('download', 'benchmark-data.csv');
                link.click();
                URL.revokeObjectURL(url);
            } catch (err) {
                console.error('Error exporting data:', err);
                alert('Failed to export data: ' + err.message);
            }
        });
    }
    
    /**
     * 获取当前图表设置
     * @returns {Object} 图表设置
     */
    getChartSettings() {
        // 确保返回的设置中chartType始终为'bar'
        return { ...this.chartSettings, chartType: 'bar' };
    }
}

// 当DOM加载完成后初始化
document.addEventListener('DOMContentLoaded', () => {
    // 图表控制组件会被main.js初始化
    // 或者在这里直接初始化：
    // window.chartControlsComponent = new ChartControls();
});
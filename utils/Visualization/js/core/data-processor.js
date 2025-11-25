// data-processor.js - 数据处理模块 (修改版)

/**
 * 数据处理模块
 * 负责处理和转换CSV数据
 */
class DataProcessor {
    /**
     * 初始化数据处理器
     */
    constructor() {
        // 可配置的处理选项
        this.options = {
            calculateEfficiency: true,         // 是否计算效率
            normalizeLayerNames: true,         // 是否标准化层名称
            inferMissingValues: true,          // 是否推断缺失值
            derivedCalculations: true,         // 是否计算派生字段
            maxDecimalPoints: 4                // 小数点精度 (改为4位，适合Ops/Cycle)
        };
        
        // 初始化
        this.init();
    }
    
    /**
     * 初始化模块
     */
    init() {
        console.log('DataProcessor initialized.');
    }
    
    /**
     * 处理数据
     * @param {Array} rawData - 原始CSV数据
     * @returns {Array} 处理后的数据
     */
    processData(rawData) {
        if (!rawData || !Array.isArray(rawData)) {
            console.error('Invalid data passed to processData');
            return [];
        }
        
        // 创建新数组进行处理，不修改原始数据
        const processedData = rawData.map(row => this.processRow(row));
        
        return processedData;
    }
    
    /**
     * 处理单行数据
     * @param {Object} row - 原始数据行
     * @returns {Object} 处理后的数据行
     */
    processRow(row) {
        // 创建新对象，不修改原始对象
        const processedRow = { ...row };
        
        // 标准化层名称
        if (this.options.normalizeLayerNames && processedRow.layer_name) {
            processedRow.layer_name = this.normalizeLayerName(processedRow.layer_name);
        }
        
        // 确保必要字段存在
        this.ensureRequiredFields(processedRow);
        
        // 推断缺失值
        if (this.options.inferMissingValues) {
            this.inferMissingFieldValues(processedRow);
        }
        
        // 计算效率 (每周期操作数) - 修改为 Ops/Cycle
        if (this.options.calculateEfficiency && 
            processedRow.total_cycles !== undefined && 
            processedRow.ops !== undefined && 
            processedRow.total_cycles > 0) {
            
            processedRow.efficiency = this.calculateEfficiency(
                processedRow.ops, 
                processedRow.total_cycles
            );
        } else {
            processedRow.efficiency = 0;
        }
        
        // 计算派生字段
        if (this.options.derivedCalculations) {
            this.calculateDerivedFields(processedRow);
        }
        
        return processedRow;
    }
    
    /**
     * 标准化层名称
     * @param {string} layerName - 原始层名称
     * @returns {string} 标准化后的层名称
     */
    normalizeLayerName(layerName) {
        if (!layerName) return 'Unknown';
        
        // 去除前后空格
        let normalized = layerName.trim();
        
        // 移除多余的空格
        normalized = normalized.replace(/\s+/g, ' ');
        
        // 如果名称过长，截断并添加省略号
        const maxLength = 50;
        if (normalized.length > maxLength) {
            normalized = normalized.substring(0, maxLength) + '...';
        }
        
        return normalized;
    }
    
    /**
     * 确保必要字段存在
     * @param {Object} row - 数据行
     */
    ensureRequiredFields(row) {
        // 定义必要字段及其默认值
        const requiredFields = {
            layer_name: 'Unknown',
            total_cycles: 0,
            ops: 0,
            L2_input_dma: 0,
            L2_computation: 0,
            L2_output_dma: 0,
            L3_input_dma: 0,
            L3_output_dma: 0
        };
        
        // 为缺失字段设置默认值
        for (const [field, defaultValue] of Object.entries(requiredFields)) {
            if (row[field] === undefined || row[field] === null || row[field] === '') {
                row[field] = defaultValue;
            }
        }
    }
    
    /**
     * 推断缺失字段值
     * @param {Object} row - 数据行
     */
    inferMissingFieldValues(row) {
        // 尝试从其他字段推断总周期
        if (row.total_cycles === 0) {
            // 如果有计算周期和DMA周期，可以推断总周期
            const L2Cycles = (row.L2_input_dma || 0) + 
                             (row.L2_computation || 0) + 
                             (row.L2_output_dma || 0);
                             
            const L3Cycles = (row.L3_input_dma || 0) + 
                             (row.L3_output_dma || 0);
                             
            // 总周期是L2和L3周期的最大值
            row.total_cycles = Math.max(L2Cycles, L3Cycles);
        }
        
        // 修改: 尝试从效率和总周期推断操作数
        if (row.ops === 0 && row.efficiency && row.efficiency > 0 && row.total_cycles > 0) {
            // 修改计算方式，适应Ops/Cycle
            row.ops = Math.round(row.efficiency * row.total_cycles);
        }
    }
    
    /**
     * 计算每周期操作数 (效率) - 修改为 Ops/Cycle
     * @param {number} ops - 操作数
     * @param {number} totalCycles - 总周期数
     * @returns {number} 每周期操作数
     */
    calculateEfficiency(ops, totalCycles) {
        if (!totalCycles || totalCycles === 0) return 0;
        
        const efficiency = ops / totalCycles;
        
        // 格式化小数点位数
        return parseFloat(efficiency.toFixed(this.options.maxDecimalPoints));
    }
    
    /**
     * 计算派生字段
     * @param {Object} row - 数据行
     */
    calculateDerivedFields(row) {
        // 计算各部分周期占总周期的百分比
        if (row.total_cycles > 0) {
            // L2周期百分比
            row.L2_input_percentage = this.calculatePercentage(row.L2_input_dma, row.total_cycles);
            row.L2_compute_percentage = this.calculatePercentage(row.L2_computation, row.total_cycles);
            row.L2_output_percentage = this.calculatePercentage(row.L2_output_dma, row.total_cycles);
            
            // L3周期百分比
            row.L3_input_percentage = this.calculatePercentage(row.L3_input_dma, row.total_cycles);
            row.L3_output_percentage = this.calculatePercentage(row.L3_output_dma, row.total_cycles);
            
            // 计算计算周期与DMA周期的比率
            const dmaCycles = (row.L2_input_dma || 0) + 
                              (row.L2_output_dma || 0) + 
                              (row.L3_input_dma || 0) + 
                              (row.L3_output_dma || 0);
                              
            row.compute_to_dma_ratio = dmaCycles > 0 
                ? this.formatDecimal(row.L2_computation / dmaCycles)
                : 0;
                
            // 计算L2和L3的比率
            const l2Cycles = (row.L2_input_dma || 0) + 
                             (row.L2_computation || 0) + 
                             (row.L2_output_dma || 0);
                             
            const l3Cycles = (row.L3_input_dma || 0) + 
                             (row.L3_output_dma || 0);
                             
            row.l2_to_l3_ratio = l3Cycles > 0 
                ? this.formatDecimal(l2Cycles / l3Cycles)
                : 0;
        }
    }
    
    /**
     * 计算百分比
     * @param {number} value - 值
     * @param {number} total - 总值
     * @returns {number} 百分比值
     */
    calculatePercentage(value, total) {
        if (!total || total === 0) return 0;
        
        const percentage = (value / total) * 100;
        return this.formatDecimal(percentage);
    }
    
    /**
     * 格式化小数
     * @param {number} value - 数值
     * @returns {number} 格式化后的数值
     */
    formatDecimal(value) {
        return parseFloat(value.toFixed(this.options.maxDecimalPoints));
    }
    
    /**
     * 设置处理选项
     * @param {Object} options - 新选项
     */
    setOptions(options) {
        this.options = { ...this.options, ...options };
    }
}

// 当DOM加载完成后初始化
document.addEventListener('DOMContentLoaded', () => {
    // 数据处理器会被main.js初始化
    // 或者在这里直接初始化：
    // window.dataProcessorComponent = new DataProcessor();
});
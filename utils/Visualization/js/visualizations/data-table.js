// data-table.js - 数据表格组件

/**
 * 数据表格组件
 * 负责管理和渲染表格数据、排序和过滤功能
 */
class DataTable {
    /**
     * 初始化数据表格
     */
    constructor() {
        // DOM元素
        this.tableBody = document.getElementById('table-body');
        this.fileSelector = document.getElementById('table-file-selector');
        this.searchInput = document.getElementById('table-search');
        this.tableHeaders = document.querySelectorAll('#data-table th');
        
        // 状态变量
        this.displayData = [];
        this.allData = [];
        this.sortColumn = 'total_cycles';
        this.sortDirection = 'desc';
        this.currentPage = 1;
        this.rowsPerPage = 20;
        this.filteredData = [];
        
        // 初始化
        this.init();
    }
    
    /**
     * 初始化表格组件
     */
    init() {
        // 设置事件监听器
        this.setupEventListeners();
        
        // 订阅数据更新事件
        EventBus.on('data:updated', (dataFiles) => {
            this.updateDataSource(dataFiles);
        });
        
        console.log('DataTable initialized.');
    }
    
    /**
     * 设置事件监听器
     */
    setupEventListeners() {
        // 文件选择器变更事件
        this.fileSelector.addEventListener('change', () => {
            this.filterDataByFile();
            this.renderTable();
        });
        
        // 搜索输入事件
        this.searchInput.addEventListener('input', () => {
            this.filterData();
            this.renderTable();
        });
        
        // 表头点击排序事件
        this.tableHeaders.forEach(header => {
            const column = header.getAttribute('data-sort');
            if (column) {
                header.addEventListener('click', () => {
                    this.sortTable(column);
                });
            }
        });
        
        // 监听排序控件事件
        EventBus.on('controls:sort-changed', (data) => {
            this.sortTable(data.column, data.direction);
        });
    }
    
    /**
     * 更新数据源
     * @param {Array} dataFiles - 含有各文件数据的数组
     */
    updateDataSource(dataFiles) {
        // 清空之前的数据
        this.allData = [];
        
        // 处理所有文件数据
        dataFiles.forEach(file => {
            file.data.forEach(row => {
                this.allData.push({
                    fileIndex: file.fileIndex,
                    fileName: file.fileName,
                    color: file.color,
                    ...row
                });
            });
        });
        
        // 更新文件选择器
        this.updateFileSelector(dataFiles);
        
        // 应用过滤和排序
        this.filterData();
        this.sortData();
        
        // 渲染表格
        this.renderTable();
    }
    
    /**
     * 更新文件选择器选项
     * @param {Array} dataFiles - 文件数据数组
     */
    updateFileSelector(dataFiles) {
        this.fileSelector.innerHTML = '<option value="all">All Files</option>';
        
        dataFiles.forEach(file => {
            const option = document.createElement('option');
            option.value = file.fileIndex;
            option.textContent = file.fileName;
            this.fileSelector.appendChild(option);
        });
    }
    
    /**
     * 按文件过滤数据
     */
    filterDataByFile() {
        const selectedFileIndex = this.fileSelector.value;
        
        if (selectedFileIndex === 'all') {
            // 显示所有文件的数据
            this.displayData = [...this.allData];
        } else {
            // 显示选中文件的数据
            this.displayData = this.allData.filter(row => 
                row.fileIndex.toString() === selectedFileIndex
            );
        }
        
        this.filterData();
    }
    
    /**
     * 根据搜索词过滤数据
     */
    filterData() {
        const searchTerm = this.searchInput.value.toLowerCase();
        
        // 先按文件过滤
        this.filterDataByFile();
        
        // 如果有搜索词，再按搜索词过滤
        if (searchTerm) {
            this.filteredData = this.displayData.filter(row => {
                // 在layer_name中搜索
                return row.layer_name && 
                       row.layer_name.toLowerCase().includes(searchTerm);
            });
        } else {
            this.filteredData = [...this.displayData];
        }
        
        // 重置到第一页
        this.currentPage = 1;
    }
    
    /**
     * 排序表格
     * @param {string} column - 排序列
     * @param {string} direction - 排序方向 (可选)
     */
    sortTable(column, direction) {
        // 如果是同一列，切换排序方向
        if (column === this.sortColumn && !direction) {
            this.sortDirection = this.sortDirection === 'asc' ? 'desc' : 'asc';
        } else if (direction) {
            // 使用传入的方向
            this.sortDirection = direction;
        } else {
            // 新列，默认降序
            this.sortDirection = 'desc';
        }
        
        this.sortColumn = column;
        
        // 排序数据并重绘表格
        this.sortData();
        this.renderTable();
        
        // 更新排序图标
        this.updateSortIndicators();
        
        // 通知其他组件排序变更
        EventBus.emit('table:sort-changed', {
            column: this.sortColumn,
            direction: this.sortDirection
        });
    }
    
    /**
     * 排序数据
     */
    sortData() {
        const column = this.sortColumn;
        const direction = this.sortDirection;
        
        this.filteredData.sort((a, b) => {
            let valueA = a[column] !== undefined ? a[column] : 0;
            let valueB = b[column] !== undefined ? b[column] : 0;
            
            // 对于文本列进行特殊处理
            if (column === 'layer_name') {
                valueA = (valueA || '').toString();
                valueB = (valueB || '').toString();
                return direction === 'asc' 
                    ? valueA.localeCompare(valueB) 
                    : valueB.localeCompare(valueA);
            }
            
            // 数值比较
            return direction === 'asc' 
                ? valueA - valueB 
                : valueB - valueA;
        });
    }
    
    /**
     * 更新排序指示器
     */
    updateSortIndicators() {
        // 移除所有排序类
        this.tableHeaders.forEach(header => {
            header.classList.remove('sort-asc', 'sort-desc');
            
            // 移除现有的排序图标
            const existingIcon = header.querySelector('.sort-icon');
            if (existingIcon) {
                header.removeChild(existingIcon);
            }
        });
        
        // 为当前排序列添加类和图标
        const currentHeader = document.querySelector(`th[data-sort="${this.sortColumn}"]`);
        if (currentHeader) {
            currentHeader.classList.add(`sort-${this.sortDirection}`);
            
            const sortIcon = document.createElement('span');
            sortIcon.className = 'sort-icon';
            currentHeader.appendChild(sortIcon);
        }
    }
    
    /**
     * 获取当前页的数据
     * @returns {Array} 当前页的数据
     */
    getCurrentPageData() {
        const startIndex = (this.currentPage - 1) * this.rowsPerPage;
        const endIndex = startIndex + this.rowsPerPage;
        return this.filteredData.slice(startIndex, endIndex);
    }

    /**
     * 计算统计数据
     * @returns {Object} 统计数据对象
     */
    calculateStats() {
        // 初始化统计数据
        const stats = {
            totalL2Transport: 0, // L2 输入输出搬运总时间
            totalL3Transport: 0, // L3 输入输出搬运总时间
            totalL2Computation: 0, // L2 计算总时间
            totalCycles: 0 // 总时间
        };
        
        // 使用当前过滤后的数据计算
        this.filteredData.forEach(row => {
            // L2 搬运时间 = 输入DMA + 输出DMA
            const l2Transport = (row.L2_input_dma || 0) + (row.L2_output_dma || 0);
            stats.totalL2Transport += l2Transport;
            
            // L3 搬运时间 = 输入DMA + 输出DMA
            const l3Transport = (row.L3_input_dma || 0) + (row.L3_output_dma || 0);
            stats.totalL3Transport += l3Transport;
            
            // L2 计算时间
            stats.totalL2Computation += (row.L2_computation || 0);
            
            // 总时间
            stats.totalCycles += (row.total_cycles || 0);
        });
        
        return stats;
    }
    
    /**
     * 渲染表格
     */
    renderTable() {
        this.tableBody.innerHTML = '';
        
        // 获取当前页数据
        const pageData = this.getCurrentPageData();
        
        if (pageData.length === 0) {
            // 没有数据时显示提示
            const tr = document.createElement('tr');
            const td = document.createElement('td');
            td.colSpan = this.tableHeaders.length;
            td.textContent = 'No data available';
            td.style.textAlign = 'center';
            tr.appendChild(td);
            this.tableBody.appendChild(tr);
            return;
        }
        
        // 渲染数据行
        pageData.forEach(row => {
            const tr = document.createElement('tr');
            
            // 添加文件颜色标识和层名称
            const nameTd = document.createElement('td');
            const colorIndicator = document.createElement('span');
            colorIndicator.className = 'file-indicator';
            colorIndicator.style.backgroundColor = row.color;
            nameTd.appendChild(colorIndicator);
            nameTd.appendChild(document.createTextNode(row.layer_name || 'Unknown'));
            tr.appendChild(nameTd);
            
            // L2 Input DMA
            this.appendCell(tr, row.L2_input_dma);
            
            // L2 Computation
            this.appendCell(tr, row.L2_computation);
            
            // L2 Output DMA
            this.appendCell(tr, row.L2_output_dma);
            
            // L3 Input DMA
            this.appendCell(tr, row.L3_input_dma);
            
            // L3 Output DMA
            this.appendCell(tr, row.L3_output_dma);
            
            // Total Cycles
            this.appendCell(tr, row.total_cycles, true);
            
            // Operations
            this.appendCell(tr, row.ops);
            
            // Cycles per Operation
            const efficiencyTd = document.createElement('td');
            if (row.ops > 0) {
                const efficiency = row.total_cycles / row.ops;
                efficiencyTd.textContent = efficiency.toFixed(2);
                
                // 如果效率较低，添加高亮
                if (efficiency > 100) {
                    efficiencyTd.classList.add('over-threshold');
                }
            } else {
                efficiencyTd.textContent = 'N/A';
            }
            tr.appendChild(efficiencyTd);
            
            this.tableBody.appendChild(tr);
        });
        
        // 添加统计行
        this.renderStatsRow();
        
        // 创建或更新分页控件
        this.renderPagination();
    }
    
    /**
     * 渲染统计行
     */
    renderStatsRow() {
        // 计算统计数据
        const stats = this.calculateStats();
        
        // 创建统计行
        const statsRow = document.createElement('tr');
        statsRow.className = 'stats-row';
        
        // 添加标题单元格
        const titleCell = document.createElement('td');
        titleCell.textContent = '统计信息';
        titleCell.style.fontWeight = 'bold';
        statsRow.appendChild(titleCell);
        
        // 添加L2 Input DMA单元格（留空）
        statsRow.appendChild(document.createElement('td'));
        
        // 添加L2 Computation单元格
        const l2CompCell = document.createElement('td');
        l2CompCell.textContent = stats.totalL2Computation.toLocaleString();
        l2CompCell.classList.add('highlight-value');
        statsRow.appendChild(l2CompCell);
        
        // 添加L2 Output DMA单元格（留空）
        statsRow.appendChild(document.createElement('td'));
        
        // 添加L3 Input DMA单元格（留空）
        statsRow.appendChild(document.createElement('td'));
        
        // 添加L3 Output DMA单元格（留空）
        statsRow.appendChild(document.createElement('td'));
        
        // 添加Total Cycles单元格
        const totalCyclesCell = document.createElement('td');
        totalCyclesCell.textContent = stats.totalCycles.toLocaleString();
        totalCyclesCell.classList.add('highlight-value');
        statsRow.appendChild(totalCyclesCell);
        
        // 添加Operations单元格（留空）
        statsRow.appendChild(document.createElement('td'));
        
        // 添加Cycles per Operation单元格（留空）
        statsRow.appendChild(document.createElement('td'));
        
        // 将统计行添加到表格底部
        this.tableBody.appendChild(statsRow);
        
        // 添加L2总搬运时间行
        this.appendStatsRow('L2总搬运时间', stats.totalL2Transport);
        
        // 添加L3总搬运时间行
        this.appendStatsRow('L3总搬运时间', stats.totalL3Transport);
    }
    
    /**
     * 添加统计数据行
     * @param {string} label - 标签
     * @param {number} value - 值
     */
    appendStatsRow(label, value) {
        const row = document.createElement('tr');
        row.className = 'stats-row';
        
        // 添加标签单元格
        const labelCell = document.createElement('td');
        labelCell.textContent = label;
        labelCell.style.fontWeight = 'bold';
        row.appendChild(labelCell);
        
        // 添加空白单元格
        for (let i = 0; i < 5; i++) {
            row.appendChild(document.createElement('td'));
        }
        
        // 添加值单元格
        const valueCell = document.createElement('td');
        valueCell.textContent = value.toLocaleString();
        valueCell.classList.add('highlight-value');
        row.appendChild(valueCell);
        
        // 添加剩余空白单元格
        for (let i = 0; i < 2; i++) {
            row.appendChild(document.createElement('td'));
        }
        
        // 将行添加到表格
        this.tableBody.appendChild(row);
    }
    
    /**
     * 添加单元格
     * @param {HTMLElement} tr - 表格行
     * @param {number} value - 单元格值
     * @param {boolean} highlight - 是否高亮 (可选)
     */
    appendCell(tr, value, highlight = false) {
        const td = document.createElement('td');
        td.textContent = value ? value.toLocaleString() : '0';
        
        if (highlight && value) {
            td.classList.add('highlight-value');
        }
        
        tr.appendChild(td);
    }
    
    /**
     * 渲染分页控件
     */
    renderPagination() {
        // 移除现有分页
        const existingPagination = document.querySelector('.pagination');
        if (existingPagination) {
            existingPagination.remove();
        }
        
        // 计算总页数
        const totalPages = Math.ceil(this.filteredData.length / this.rowsPerPage);
        
        // 如果只有一页，不显示分页
        if (totalPages <= 1) return;
        
        // 创建分页容器
        const paginationDiv = document.createElement('div');
        paginationDiv.className = 'pagination';
        
        // 上一页按钮
        const prevBtn = document.createElement('button');
        prevBtn.innerHTML = '&laquo; Prev';
        prevBtn.disabled = this.currentPage === 1;
        prevBtn.addEventListener('click', () => {
            if (this.currentPage > 1) {
                this.currentPage--;
                this.renderTable();
            }
        });
        paginationDiv.appendChild(prevBtn);
        
        // 页码按钮
        const maxVisiblePages = 5;
        const halfVisible = Math.floor(maxVisiblePages / 2);
        let startPage = Math.max(1, this.currentPage - halfVisible);
        let endPage = Math.min(totalPages, startPage + maxVisiblePages - 1);
        
        if (endPage - startPage + 1 < maxVisiblePages) {
            startPage = Math.max(1, endPage - maxVisiblePages + 1);
        }
        
        for (let i = startPage; i <= endPage; i++) {
            const pageBtn = document.createElement('button');
            pageBtn.textContent = i;
            pageBtn.classList.toggle('active', i === this.currentPage);
            pageBtn.addEventListener('click', () => {
                this.currentPage = i;
                this.renderTable();
            });
            paginationDiv.appendChild(pageBtn);
        }
        
        // 下一页按钮
        const nextBtn = document.createElement('button');
        nextBtn.innerHTML = 'Next &raquo;';
        nextBtn.disabled = this.currentPage === totalPages;
        nextBtn.addEventListener('click', () => {
            if (this.currentPage < totalPages) {
                this.currentPage++;
                this.renderTable();
            }
        });
        paginationDiv.appendChild(nextBtn);
        
        // 添加分页到表格下方
        document.getElementById('table-tab').appendChild(paginationDiv);
    }
}

// 当DOM加载完成后初始化
document.addEventListener('DOMContentLoaded', () => {
    // 表格组件会被main.js初始化
    // 或者在这里直接初始化：
    // window.dataTableComponent = new DataTable();
});
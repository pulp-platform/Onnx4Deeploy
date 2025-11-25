// tab-manager.js - 标签页管理模块

/**
 * 标签页管理器
 * 负责处理标签页切换和相关行为
 */
class TabManager {
    /**
     * 初始化标签页管理器
     */
    constructor() {
        // DOM元素
        this.tabButtons = document.querySelectorAll('.tablinks');
        this.tabContents = document.querySelectorAll('.tabcontent');
        
        // 当前激活的标签页
        this.activeTabId = 'cycles-tab';
        
        // 初始化
        this.init();
    }
    
    /**
     * 初始化标签页管理器
     */
    init() {
        this.setupEventListeners();
        
        // 默认显示第一个标签页
        this.showTab('cycles-tab');
        
        console.log('TabManager initialized.');
    }
    
    /**
     * 设置事件监听器
     */
    setupEventListeners() {
        // 为每个标签按钮添加点击事件
        this.tabButtons.forEach(button => {
            button.addEventListener('click', (event) => {
                const tabId = event.currentTarget.getAttribute('data-tab');
                this.showTab(tabId);
            });
        });
        
        // 监听URL哈希变化以支持标签页导航
        window.addEventListener('hashchange', () => {
            this.handleHashChange();
        });
        
        // 检查初始哈希
        this.handleHashChange();
    }
    
    /**
     * 处理URL哈希变化
     */
    handleHashChange() {
        const hash = window.location.hash.substring(1);
        if (hash && this.isValidTab(hash)) {
            this.showTab(hash);
        }
    }
    
    /**
     * 验证标签页ID是否有效
     * @param {string} tabId - 标签页ID
     * @returns {boolean} 是否是有效的标签页ID
     */
    isValidTab(tabId) {
        return ['cycles-tab', 'pie-tab', 'table-tab'].includes(tabId);
    }
    
    /**
     * 显示指定的标签页
     * @param {string} tabId - 标签页ID
     */
    showTab(tabId) {
        if (!this.isValidTab(tabId)) {
            console.error(`Invalid tab ID: ${tabId}`);
            return;
        }
        
        // 隐藏所有标签内容
        this.tabContents.forEach(content => {
            content.classList.remove('visible');
        });
        
        // 取消激活所有标签按钮
        this.tabButtons.forEach(button => {
            button.classList.remove('active');
        });
        
        // 显示选中的标签内容
        const selectedTab = document.getElementById(tabId);
        if (selectedTab) {
            selectedTab.classList.add('visible');
        }
        
        // 激活相应的标签按钮
        const selectedButton = document.querySelector(`.tablinks[data-tab="${tabId}"]`);
        if (selectedButton) {
            selectedButton.classList.add('active');
        }
        
        // 更新当前激活的标签ID
        this.activeTabId = tabId;
        
        // 更新URL哈希，但不触发额外的hashchange事件
        this.updateUrlHash(tabId);
        
        // 发出标签切换事件
        this.notifyTabChange(tabId);
    }
    
    /**
     * 更新URL哈希
     * @param {string} tabId - 标签页ID
     */
    updateUrlHash(tabId) {
        // 临时移除hashchange监听器
        window.removeEventListener('hashchange', this.handleHashChange);
        
        // 更新URL哈希
        window.location.hash = tabId;
        
        // 恢复hashchange监听器
        setTimeout(() => {
            window.addEventListener('hashchange', this.handleHashChange);
        }, 0);
    }
    
    /**
     * 通知标签页变更
     * @param {string} tabId - 标签页ID
     */
    notifyTabChange(tabId) {
        // 根据标签页ID转换为标签类型
        let tabType = 'cycles';
        
        switch (tabId) {
            case 'cycles-tab':
                tabType = 'cycles';
                break;
            case 'pie-tab':
                tabType = 'pie';
                break;
            case 'table-tab':
                tabType = 'table';
                break;
        }
        
        // 发出标签切换事件
        EventBus.emit('tab:changed', {
            tabId: tabId,
            tabType: tabType
        });
        
        // 根据标签类型，可能需要重新调整特定图表
        this.handleTabSpecificActions(tabType);
    }
    
    /**
     * 处理标签特定的操作
     * @param {string} tabType - 标签类型
     */
    handleTabSpecificActions(tabType) {
        // 在标签切换时可能需要调整图表大小或刷新
        switch (tabType) {
            case 'cycles':
                // 周期图表可能需要调整尺寸
                EventBus.emit('chart:resize', { chartType: 'cycles' });
                break;
                
            case 'pie':
                // 饼图可能需要调整尺寸
                EventBus.emit('chart:resize', { chartType: 'pie' });
                break;
                
            case 'table':
                // 表格可能需要刷新
                EventBus.emit('table:refresh', {});
                break;
        }
    }
    
    /**
     * 获取当前激活的标签页ID
     * @returns {string} 当前标签页ID
     */
    getActiveTabId() {
        return this.activeTabId;
    }
    
    /**
     * 以编程方式激活标签页
     * @param {string} tabId - 要激活的标签页ID
     */
    activateTab(tabId) {
        this.showTab(tabId);
    }
}

// 当DOM加载完成后初始化
document.addEventListener('DOMContentLoaded', () => {
    // 标签页管理器会被main.js初始化
    // 或者在这里直接初始化：
    // window.tabManagerComponent = new TabManager();
});
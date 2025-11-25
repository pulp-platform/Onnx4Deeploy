// event-bus.js - 事件总线模块

/**
 * 事件总线
 * 用于模块间的事件通信
 */
class EventBusClass {
    constructor() {
        this.events = {};
        this.onceEvents = {};
    }
    
    /**
     * 订阅事件
     * @param {string} event - 事件名称
     * @param {Function} callback - 回调函数
     */
    on(event, callback) {
        if (!this.events[event]) {
            this.events[event] = [];
        }
        this.events[event].push(callback);
    }
    
    /**
     * 订阅一次性事件 (触发后自动取消订阅)
     * @param {string} event - 事件名称
     * @param {Function} callback - 回调函数
     */
    once(event, callback) {
        if (!this.onceEvents[event]) {
            this.onceEvents[event] = [];
        }
        this.onceEvents[event].push(callback);
    }
    
    /**
     * 取消订阅事件
     * @param {string} event - 事件名称
     * @param {Function} callback - 回调函数
     */
    off(event, callback) {
        if (this.events[event]) {
            this.events[event] = this.events[event].filter(cb => cb !== callback);
        }
        
        if (this.onceEvents[event]) {
            this.onceEvents[event] = this.onceEvents[event].filter(cb => cb !== callback);
        }
    }
    
    /**
     * 触发事件
     * @param {string} event - 事件名称
     * @param {*} data - 事件数据
     */
    emit(event, data) {
        // 触发普通订阅
        if (this.events[event]) {
            this.events[event].forEach(callback => {
                try {
                    callback(data);
                } catch (err) {
                    console.error(`Error in event handler for ${event}:`, err);
                }
            });
        }
        
        // 触发一次性订阅
        if (this.onceEvents[event]) {
            const callbacks = [...this.onceEvents[event]];
            this.onceEvents[event] = [];
            
            callbacks.forEach(callback => {
                try {
                    callback(data);
                } catch (err) {
                    console.error(`Error in once event handler for ${event}:`, err);
                }
            });
        }
    }
    
    /**
     * 清除所有事件监听器
     */
    clear() {
        this.events = {};
        this.onceEvents = {};
    }
    
    /**
     * 获取特定事件的监听器数量
     * @param {string} event - 事件名称
     * @returns {number} 监听器数量
     */
    listenerCount(event) {
        const regularCount = this.events[event] ? this.events[event].length : 0;
        const onceCount = this.onceEvents[event] ? this.onceEvents[event].length : 0;
        return regularCount + onceCount;
    }
}

// 创建全局事件总线实例
const EventBus = new EventBusClass();

// 确保在页面卸载时清除所有事件
window.addEventListener('unload', () => {
    EventBus.clear();
});

console.log('EventBus initialized and available globally.');
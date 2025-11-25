class CyclesChart {
    /**
     * Initialize the cycles chart
     */
    constructor() {
        // DOM element
        this.chartContainer = document.getElementById('cycles-chart');
        
        // State variables
        this.currentData = [];
        this.chartSettings = {
            chartType: 'bar',  // Use bar chart only
            sortBy: 'total_cycles',
            sortDirection: 'desc',
            maxLayers: 25,
            yScaleType: 'linear'
        };
        
        // Statistics
        this.stats = {
            totalCycles: 0,
            fileStats: []
        };
        
        // Initialization
        this.init();
    }
    
    /**
     * Initialize chart events and listeners
     */
    init() {
        // Subscribe to data update event
        EventBus.on('data:updated', (dataFiles) => {
            this.currentData = dataFiles;
            this.calculateStats();
            this.renderChart();
        });
        
        // Subscribe to chart settings update event
        EventBus.on('chart:settings-changed', (settings) => {
            // Force chartType to 'bar', allow other settings to update
            this.chartSettings = {
                ...settings,
                chartType: 'bar'
            };
            this.renderChart();
        });
        
        console.log('CyclesChart initialized with enhanced hover info and no text labels.');
    }
    
    /**
     * Compute statistics from input data
     */
    calculateStats() {
        // Reset stats
        this.stats = {
            totalCycles: 0,
            fileStats: []
        };
        
        if (!this.currentData || this.currentData.length === 0) return;
        
        // Calculate total cycles per file
        this.currentData.forEach(file => {
            let fileTotalCycles = 0;
            file.data.forEach(layer => {
                const cycles = layer.total_cycles || 0;
                fileTotalCycles += cycles;
                this.stats.totalCycles += cycles;
            });
            this.stats.fileStats.push({
                fileName: file.fileName,
                color: file.color,
                totalCycles: fileTotalCycles,
                layerCount: file.data.length
            });
        });

        console.log('Statistics calculated:', this.stats);
    }
    
    /**
     * Render the chart
     */
    renderChart() {
        if (!this.currentData || this.currentData.length === 0) {
            this.showNoDataMessage();
            return;
        }

        console.log('Rendering chart. Number of files:', this.currentData.length);

        try {
            const { traces, xAxisCategories } = this.prepareNonOverlappingTraces();
            const layout = this.prepareLayout(xAxisCategories);
            this.chartContainer.innerHTML = '';
            this.createStatsPanel();

            const config = {
                responsive: true,
                displayModeBar: true,
                displaylogo: false,
                modeBarButtonsToAdd: ['resetScale2d'],
                toImageButtonOptions: {
                    format: 'png',
                    filename: 'benchmark_cycles_chart',
                    height: 800,
                    width: 1200,
                    scale: 2
                }
            };

            console.log('Calling Plotly.newPlot to render chart');
            Plotly.newPlot(this.chartContainer, traces, layout, config);

            window.addEventListener('resize', this.handleResize.bind(this));
            this.chartContainer.on('plotly_click', (data) => {
                this.handleChartClick(data);
            });

        } catch (error) {
            console.error('Error rendering chart:', error);
            this.showErrorMessage(error.message);
        }
    }

    /**
     * Create a statistics info panel
     */
    createStatsPanel() {
        const statsPanel = document.createElement('div');
        statsPanel.className = 'stats-panel';
        statsPanel.style = `
            position: absolute;
            top: 10px;
            right: 10px;
            background-color: rgba(255, 255, 255, 0.9);
            padding: 10px;
            border: 1px solid #ddd;
            border-radius: 5px;
            font-size: 14px;
            z-index: 1000;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        `;

        const title = document.createElement('div');
        title.textContent = 'Total Cycle Statistics';
        title.style.fontWeight = 'bold';
        title.style.marginBottom = '5px';
        title.style.borderBottom = '1px solid #ddd';
        title.style.paddingBottom = '3px';
        statsPanel.appendChild(title);

        const totalDiv = document.createElement('div');
        totalDiv.innerHTML = `<strong>Total cycles across all layers:</strong> <span style="color: #007bff">${this.stats.totalCycles.toLocaleString()}</span>`;
        totalDiv.style.marginBottom = '8px';
        statsPanel.appendChild(totalDiv);

        if (this.stats.fileStats.length > 1) {
            this.stats.fileStats.forEach(fileStat => {
                const fileDiv = document.createElement('div');
                fileDiv.style.display = 'flex';
                fileDiv.style.alignItems = 'center';
                fileDiv.style.marginBottom = '3px';

                const colorIndicator = document.createElement('span');
                colorIndicator.style = `
                    display: inline-block;
                    width: 12px;
                    height: 12px;
                    border-radius: 50%;
                    background-color: ${fileStat.color};
                    margin-right: 5px;
                `;
                fileDiv.appendChild(colorIndicator);

                const fileInfo = document.createElement('span');
                fileInfo.innerHTML = `${fileStat.fileName}: <span style="color: #007bff">${fileStat.totalCycles.toLocaleString()}</span> cycles (${fileStat.layerCount} layers)`;
                fileDiv.appendChild(fileInfo);

                statsPanel.appendChild(fileDiv);
            });
        }

        this.chartContainer.appendChild(statsPanel);
    }

    /**
     * Handle window resize
     */
    handleResize() {
        Plotly.relayout(this.chartContainer, {
            'xaxis.autorange': true,
            'yaxis.autorange': true
        });
    }

    /**
     * Display error message
     * @param {string} message - Error message
     */
    showErrorMessage(message) {
        this.chartContainer.innerHTML = `
            <div class="error-message" style="text-align: center; padding: 20px; color: #d32f2f;">
                <div>Chart rendering error</div>
                <div>${message}</div>
            </div>
        `;
    }

    /**
     * Prepare non-overlapping chart traces with enhanced hover info
     * @returns {Object} traces and xAxisCategories
     */
    prepareNonOverlappingTraces() {
        if (!this.currentData || this.currentData.length === 0) {
            return { traces: [], xAxisCategories: [] };
        }

        const firstFile = this.currentData[0];
        if (!firstFile || !firstFile.data) {
            return { traces: [], xAxisCategories: [] };
        }

        console.log('Preparing chart data for file:', firstFile.fileName);

        const firstFileDataCopy = JSON.parse(JSON.stringify(firstFile.data));

        const sortedFirstData = firstFileDataCopy.sort((a, b) => {
            const valueA = a[this.chartSettings.sortBy] ?? 0;
            const valueB = b[this.chartSettings.sortBy] ?? 0;
            return this.chartSettings.sortDirection === 'asc' ? valueA - valueB : valueB - valueA;
        });

        const topFirstLayers = sortedFirstData.slice(0, this.chartSettings.maxLayers);
        const layerOriginalNames = topFirstLayers.map(layer => layer.layer_name || 'Unknown');
        const referenceLayerOriginalNames = [...layerOriginalNames];
        const xAxisCategories = layerOriginalNames;
        const fileCount = this.currentData.length;
        const traces = [];

        for (let fileIndex = 0; fileIndex < fileCount; fileIndex++) {
            const file = this.currentData[fileIndex];
            if (!file || !file.data) continue;

            const xData = [];
            const yData = [];
            const hoverData = [];

            for (let layerIndex = 0; layerIndex < referenceLayerOriginalNames.length; layerIndex++) {
                const refLayerName = referenceLayerOriginalNames[layerIndex];
                const xPosition = layerIndex;

                const matchingLayer = file.data.find(layer => (layer.layer_name || 'Unknown') === refLayerName);
                let value = 0;
                let ops = 0;
                let efficiency = 0;
                let opType = 'unknown';

                if (matchingLayer) {
                    const cycles = matchingLayer[this.chartSettings.sortBy];
                    value = isNaN(cycles) ? 0 : cycles;
                    ops = matchingLayer.ops || 0;
                    efficiency = value > 0 ?  ops / value : 0;
                    opType = matchingLayer.op_type || 'unknown';
                }

                xData.push(xPosition);
                yData.push(value);

                const cyclesFormatted = value.toLocaleString();
                const opsFormatted = ops.toLocaleString();
                let hoverText = `<b>${refLayerName}</b><br>` +
                                `Cycles: ${cyclesFormatted}<br>` +
                                `Ops: ${opsFormatted}`;
                if (ops > 0) {
                    hoverText += `<br>Op per Total Cycle: ${efficiency.toFixed(4)}`;
                }
                if (opType && opType !== 'unknown') {
                    hoverText += `<br>Op Type: ${opType}`;
                }

                hoverData.push(hoverText);
            }

            traces.push({
                x: xAxisCategories,
                y: yData,
                name: file.fileName,
                type: 'bar',
                text: hoverData,
                hoverinfo: 'text+name',
                hovertemplate: '%{text}<extra>%{fullData.name}</extra>',
                marker: {
                    color: file.color,
                    line: {
                        width: 1,
                        color: this.darkenColor(file.color, 20)
                    }
                },
                textposition: 'none'
            });
        }

        return { traces, xAxisCategories };
    }

    /**
     * Prepare chart layout
     * @param {Array} xAxisCategories
     * @returns {Object} layout
     */
    prepareLayout(xAxisCategories) {
        return {
            barmode: 'group',
            bargap: 0.15,
            bargroupgap: 0.1,
            xaxis: {
                tickangle: -45,
                tickfont: { size: 10 },
                automargin: true,
                title: {
                    text: 'Layer Name',
                    font: { size: 12, color: '#7f7f7f' }
                },
                type: 'category',
                categoryorder: 'array',
                categoryarray: xAxisCategories,
                tickmode: 'array',
                tickvals: Array.from(Array(xAxisCategories.length).keys()),
                ticktext: xAxisCategories
            },
            yaxis: {
                title: {
                    text: this.chartSettings.sortBy === 'total_cycles'
                        ? 'Cycles'
                        : this.chartSettings.sortBy,
                    font: { size: 12, color: '#7f7f7f' }
                },
                type: this.chartSettings.yScaleType
            },
            hovermode: 'closest',
            showlegend: true,
            legend: {
                x: 0,
                y: 1.1,
                orientation: 'h'
            },
            margin: {
                b: 200,
                l: 80,
                r: 40,
                t: 80
            }
        };
    }

    /**
     * Darken a HEX color
     * @param {string} color - HEX color
     * @param {number} percent - Darken percentage
     * @returns {string} Darkened color
     */
    darkenColor(color, percent) {
        const num = parseInt(color.replace('#', ''), 16);
        const amt = Math.round(2.55 * percent);
        const R = (num >> 16) - amt;
        const G = (num >> 8 & 0x00FF) - amt;
        const B = (num & 0x0000FF) - amt;

        return '#' + (
            0x1000000 +
            (R < 0 ? 0 : R) * 0x10000 +
            (G < 0 ? 0 : G) * 0x100 +
            (B < 0 ? 0 : B)
        ).toString(16).slice(1);
    }

    /**
     * Show message when no data is available
     */
    showNoDataMessage() {
        this.chartContainer.innerHTML = `
            <div class="no-data-message">
                <div>No data available to display.</div>
                <div>Please upload at least one CSV file.</div>
            </div>
        `;
    }

    /**
     * Handle chart click event
     * @param {Object} data - Click event data
     */
    handleChartClick(data) {
        const point = data.points[0];
        const layerName = point.x;

        console.log(`Clicked on layer: ${layerName}`);

        EventBus.emit('chart:layer-selected', {
            layerName: layerName
        });
    }
}

// Initialize when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    // You can initialize CyclesChart here if needed
});

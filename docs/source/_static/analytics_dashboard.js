/**
 * Documentation Analytics Dashboard
 * Provides real-time analytics and monitoring for documentation usage
 */

class AnalyticsDashboard {
    constructor() {
        this.apiEndpoint = '/api/analytics';
        this.refreshInterval = 30000; // 30 seconds
        this.init();
    }

    init() {
        this.createDashboard();
        this.startDataCollection();
        this.setupEventListeners();
    }

    createDashboard() {
        // Only create dashboard if user has admin privileges
        if (!this.hasAdminAccess()) {
            return;
        }

        const dashboard = document.createElement('div');
        dashboard.id = 'analytics-dashboard';
        dashboard.innerHTML = `
            <div class="dashboard-toggle">
                <button id="dashboard-btn" title="Analytics Dashboard">📊</button>
            </div>
            <div class="dashboard-panel hidden">
                <div class="dashboard-header">
                    <h3>Documentation Analytics</h3>
                    <button class="dashboard-close">&times;</button>
                </div>
                <div class="dashboard-content">
                    <div class="metrics-grid">
                        <div class="metric-card">
                            <h4>Page Views</h4>
                            <div class="metric-value" id="page-views">-</div>
                            <div class="metric-change" id="page-views-change">-</div>
                        </div>
                        <div class="metric-card">
                            <h4>Unique Visitors</h4>
                            <div class="metric-value" id="unique-visitors">-</div>
                            <div class="metric-change" id="unique-visitors-change">-</div>
                        </div>
                        <div class="metric-card">
                            <h4>Search Queries</h4>
                            <div class="metric-value" id="search-queries">-</div>
                            <div class="metric-change" id="search-queries-change">-</div>
                        </div>
                        <div class="metric-card">
                            <h4>Feedback Score</h4>
                            <div class="metric-value" id="feedback-score">-</div>
                            <div class="metric-change" id="feedback-score-change">-</div>
                        </div>
                    </div>
                    
                    <div class="charts-section">
                        <div class="chart-container">
                            <h4>Popular Pages</h4>
                            <div id="popular-pages-chart"></div>
                        </div>
                        <div class="chart-container">
                            <h4>User Journey</h4>
                            <div id="user-journey-chart"></div>
                        </div>
                    </div>
                    
                    <div class="alerts-section">
                        <h4>System Alerts</h4>
                        <div id="system-alerts"></div>
                    </div>
                    
                    <div class="actions-section">
                        <button id="export-data">Export Data</button>
                        <button id="refresh-data">Refresh</button>
                        <button id="clear-cache">Clear Cache</button>
                    </div>
                </div>
            </div>
        `;

        dashboard.style.cssText = `
            position: fixed;
            top: 20px;
            left: 20px;
            z-index: 3000;
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
        `;

        document.body.appendChild(dashboard);
        this.addDashboardStyles();
    }

    addDashboardStyles() {
        const style = document.createElement('style');
        style.textContent = `
            .dashboard-toggle button {
                background: #2c3e50;
                color: white;
                border: none;
                border-radius: 50%;
                width: 50px;
                height: 50px;
                cursor: pointer;
                font-size: 20px;
                box-shadow: 0 2px 10px rgba(0,0,0,0.2);
                transition: all 0.3s ease;
            }
            
            .dashboard-toggle button:hover {
                background: #34495e;
                transform: scale(1.1);
            }
            
            .dashboard-panel {
                position: absolute;
                top: 60px;
                left: 0;
                width: 800px;
                max-height: 600px;
                background: white;
                border-radius: 8px;
                box-shadow: 0 4px 20px rgba(0,0,0,0.3);
                overflow-y: auto;
            }
            
            .dashboard-panel.hidden {
                display: none;
            }
            
            .dashboard-header {
                padding: 20px;
                border-bottom: 1px solid #eee;
                display: flex;
                justify-content: space-between;
                align-items: center;
                background: #f8f9fa;
                border-radius: 8px 8px 0 0;
            }
            
            .dashboard-header h3 {
                margin: 0;
                color: #2c3e50;
            }
            
            .dashboard-close {
                background: none;
                border: none;
                font-size: 24px;
                cursor: pointer;
                color: #999;
            }
            
            .dashboard-content {
                padding: 20px;
            }
            
            .metrics-grid {
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
                gap: 15px;
                margin-bottom: 30px;
            }
            
            .metric-card {
                background: #f8f9fa;
                padding: 20px;
                border-radius: 6px;
                text-align: center;
                border: 1px solid #dee2e6;
            }
            
            .metric-card h4 {
                margin: 0 0 10px 0;
                color: #6c757d;
                font-size: 14px;
                font-weight: 500;
            }
            
            .metric-value {
                font-size: 28px;
                font-weight: bold;
                color: #2c3e50;
                margin-bottom: 5px;
            }
            
            .metric-change {
                font-size: 12px;
                color: #28a745;
            }
            
            .metric-change.negative {
                color: #dc3545;
            }
            
            .charts-section {
                display: grid;
                grid-template-columns: 1fr 1fr;
                gap: 20px;
                margin-bottom: 30px;
            }
            
            .chart-container {
                background: #f8f9fa;
                padding: 20px;
                border-radius: 6px;
                border: 1px solid #dee2e6;
            }
            
            .chart-container h4 {
                margin: 0 0 15px 0;
                color: #2c3e50;
                font-size: 16px;
            }
            
            .alerts-section {
                margin-bottom: 20px;
            }
            
            .alerts-section h4 {
                margin: 0 0 15px 0;
                color: #2c3e50;
            }
            
            .alert {
                padding: 10px 15px;
                border-radius: 4px;
                margin-bottom: 10px;
                border-left: 4px solid;
            }
            
            .alert.warning {
                background: #fff3cd;
                border-color: #ffc107;
                color: #856404;
            }
            
            .alert.error {
                background: #f8d7da;
                border-color: #dc3545;
                color: #721c24;
            }
            
            .alert.info {
                background: #d1ecf1;
                border-color: #17a2b8;
                color: #0c5460;
            }
            
            .actions-section {
                display: flex;
                gap: 10px;
                justify-content: flex-end;
            }
            
            .actions-section button {
                padding: 8px 16px;
                border: 1px solid #dee2e6;
                border-radius: 4px;
                background: white;
                cursor: pointer;
                font-size: 14px;
                transition: all 0.2s;
            }
            
            .actions-section button:hover {
                background: #f8f9fa;
            }
            
            .actions-section button:active {
                background: #e9ecef;
            }
            
            @media (max-width: 768px) {
                .dashboard-panel {
                    width: calc(100vw - 40px);
                    max-width: none;
                }
                
                .charts-section {
                    grid-template-columns: 1fr;
                }
                
                .actions-section {
                    flex-direction: column;
                }
            }
        `;
        document.head.appendChild(style);
    }

    setupEventListeners() {
        if (!this.hasAdminAccess()) return;

        // Dashboard toggle
        document.getElementById('dashboard-btn').addEventListener('click', () => {
            this.toggleDashboard();
        });

        // Dashboard close
        document.querySelector('.dashboard-close').addEventListener('click', () => {
            this.hideDashboard();
        });

        // Action buttons
        document.getElementById('export-data').addEventListener('click', () => {
            this.exportData();
        });

        document.getElementById('refresh-data').addEventListener('click', () => {
            this.refreshData();
        });

        document.getElementById('clear-cache').addEventListener('click', () => {
            this.clearCache();
        });

        // Close dashboard when clicking outside
        document.addEventListener('click', (e) => {
            const dashboard = document.getElementById('analytics-dashboard');
            if (!dashboard.contains(e.target)) {
                this.hideDashboard();
            }
        });
    }

    hasAdminAccess() {
        // Check if user has admin access (implement your own logic)
        // For demo purposes, check for admin parameter or local development
        return window.location.hostname === 'localhost' || 
               window.location.search.includes('admin=true') ||
               localStorage.getItem('cais_admin') === 'true';
    }

    toggleDashboard() {
        const panel = document.querySelector('.dashboard-panel');
        panel.classList.toggle('hidden');
        
        if (!panel.classList.contains('hidden')) {
            this.refreshData();
        }
    }

    hideDashboard() {
        document.querySelector('.dashboard-panel').classList.add('hidden');
    }

    async refreshData() {
        try {
            // Simulate API call - in production, this would fetch real data
            const data = await this.fetchAnalyticsData();
            this.updateMetrics(data);
            this.updateCharts(data);
            this.updateAlerts(data);
        } catch (error) {
            console.error('Error refreshing analytics data:', error);
            this.showError('Failed to refresh analytics data');
        }
    }

    async fetchAnalyticsData() {
        // Simulate analytics data - in production, this would be real API calls
        return new Promise(resolve => {
            setTimeout(() => {
                resolve({
                    pageViews: {
                        current: Math.floor(Math.random() * 10000) + 5000,
                        change: Math.floor(Math.random() * 20) - 10
                    },
                    uniqueVisitors: {
                        current: Math.floor(Math.random() * 2000) + 1000,
                        change: Math.floor(Math.random() * 15) - 5
                    },
                    searchQueries: {
                        current: Math.floor(Math.random() * 500) + 200,
                        change: Math.floor(Math.random() * 25) - 10
                    },
                    feedbackScore: {
                        current: (Math.random() * 2 + 3).toFixed(1),
                        change: (Math.random() * 0.4 - 0.2).toFixed(1)
                    },
                    popularPages: [
                        { page: '/getting_started/', views: 1250 },
                        { page: '/methods/decision_tree/', views: 980 },
                        { page: '/api/', views: 750 },
                        { page: '/tutorials/', views: 650 },
                        { page: '/user_guide/basic_usage/', views: 520 }
                    ],
                    userJourney: [
                        { step: 'Landing', users: 1000 },
                        { step: 'Getting Started', users: 750 },
                        { step: 'First Tutorial', users: 500 },
                        { step: 'API Docs', users: 300 },
                        { step: 'Advanced Usage', users: 150 }
                    ],
                    alerts: [
                        { type: 'info', message: 'Documentation build completed successfully' },
                        { type: 'warning', message: '3 broken links detected in tutorials section' },
                        { type: 'info', message: 'New feedback received: 4.2/5 average rating' }
                    ]
                });
            }, 500);
        });
    }

    updateMetrics(data) {
        // Update page views
        document.getElementById('page-views').textContent = data.pageViews.current.toLocaleString();
        const pageViewsChange = document.getElementById('page-views-change');
        pageViewsChange.textContent = `${data.pageViews.change > 0 ? '+' : ''}${data.pageViews.change}%`;
        pageViewsChange.className = `metric-change ${data.pageViews.change < 0 ? 'negative' : ''}`;

        // Update unique visitors
        document.getElementById('unique-visitors').textContent = data.uniqueVisitors.current.toLocaleString();
        const visitorsChange = document.getElementById('unique-visitors-change');
        visitorsChange.textContent = `${data.uniqueVisitors.change > 0 ? '+' : ''}${data.uniqueVisitors.change}%`;
        visitorsChange.className = `metric-change ${data.uniqueVisitors.change < 0 ? 'negative' : ''}`;

        // Update search queries
        document.getElementById('search-queries').textContent = data.searchQueries.current.toLocaleString();
        const searchChange = document.getElementById('search-queries-change');
        searchChange.textContent = `${data.searchQueries.change > 0 ? '+' : ''}${data.searchQueries.change}%`;
        searchChange.className = `metric-change ${data.searchQueries.change < 0 ? 'negative' : ''}`;

        // Update feedback score
        document.getElementById('feedback-score').textContent = data.feedbackScore.current;
        const feedbackChange = document.getElementById('feedback-score-change');
        feedbackChange.textContent = `${data.feedbackScore.change > 0 ? '+' : ''}${data.feedbackScore.change}`;
        feedbackChange.className = `metric-change ${data.feedbackScore.change < 0 ? 'negative' : ''}`;
    }

    updateCharts(data) {
        // Update popular pages chart
        const popularPagesChart = document.getElementById('popular-pages-chart');
        popularPagesChart.innerHTML = data.popularPages.map(page => `
            <div style="display: flex; justify-content: space-between; margin-bottom: 8px; padding: 5px; background: white; border-radius: 3px;">
                <span style="font-size: 12px; color: #6c757d;">${page.page}</span>
                <span style="font-size: 12px; font-weight: bold; color: #2c3e50;">${page.views}</span>
            </div>
        `).join('');

        // Update user journey chart
        const userJourneyChart = document.getElementById('user-journey-chart');
        const maxUsers = Math.max(...data.userJourney.map(step => step.users));
        userJourneyChart.innerHTML = data.userJourney.map(step => {
            const percentage = (step.users / maxUsers) * 100;
            return `
                <div style="margin-bottom: 10px;">
                    <div style="display: flex; justify-content: space-between; margin-bottom: 3px;">
                        <span style="font-size: 12px; color: #6c757d;">${step.step}</span>
                        <span style="font-size: 12px; font-weight: bold; color: #2c3e50;">${step.users}</span>
                    </div>
                    <div style="background: #e9ecef; height: 8px; border-radius: 4px;">
                        <div style="background: #007bff; height: 100%; width: ${percentage}%; border-radius: 4px;"></div>
                    </div>
                </div>
            `;
        }).join('');
    }

    updateAlerts(data) {
        const alertsContainer = document.getElementById('system-alerts');
        if (data.alerts.length === 0) {
            alertsContainer.innerHTML = '<p style="color: #6c757d; font-style: italic;">No alerts</p>';
            return;
        }

        alertsContainer.innerHTML = data.alerts.map(alert => `
            <div class="alert ${alert.type}">
                ${alert.message}
            </div>
        `).join('');
    }

    startDataCollection() {
        // Collect basic usage data
        this.collectPageData();
        this.collectUserInteractions();
        
        // Set up periodic data refresh
        setInterval(() => {
            if (!document.querySelector('.dashboard-panel').classList.contains('hidden')) {
                this.refreshData();
            }
        }, this.refreshInterval);
    }

    collectPageData() {
        const pageData = {
            url: window.location.href,
            title: document.title,
            timestamp: new Date().toISOString(),
            userAgent: navigator.userAgent,
            referrer: document.referrer,
            viewport: {
                width: window.innerWidth,
                height: window.innerHeight
            }
        };

        // Store locally for now
        this.storeAnalyticsData('page_view', pageData);
    }

    collectUserInteractions() {
        // Track clicks on important elements
        document.addEventListener('click', (e) => {
            if (e.target.matches('a[href^="#"]') || 
                e.target.matches('.toctree-l1 a') || 
                e.target.matches('.reference')) {
                
                this.storeAnalyticsData('navigation_click', {
                    element: e.target.tagName,
                    text: e.target.textContent.trim(),
                    href: e.target.href,
                    timestamp: new Date().toISOString()
                });
            }
        });

        // Track scroll depth
        let maxScroll = 0;
        window.addEventListener('scroll', () => {
            const scrollPercent = Math.round((window.scrollY / (document.body.scrollHeight - window.innerHeight)) * 100);
            if (scrollPercent > maxScroll) {
                maxScroll = scrollPercent;
            }
        });

        // Send scroll data on page unload
        window.addEventListener('beforeunload', () => {
            this.storeAnalyticsData('scroll_depth', {
                maxScroll: maxScroll,
                timestamp: new Date().toISOString()
            });
        });
    }

    storeAnalyticsData(eventType, data) {
        const analyticsData = JSON.parse(localStorage.getItem('cais_analytics') || '[]');
        analyticsData.push({
            type: eventType,
            data: data,
            sessionId: this.getSessionId()
        });
        
        // Keep only last 1000 events
        if (analyticsData.length > 1000) {
            analyticsData.splice(0, analyticsData.length - 1000);
        }
        
        localStorage.setItem('cais_analytics', JSON.stringify(analyticsData));
    }

    getSessionId() {
        let sessionId = sessionStorage.getItem('cais_session_id');
        if (!sessionId) {
            sessionId = 'session_' + Date.now() + '_' + Math.random().toString(36).substr(2, 9);
            sessionStorage.setItem('cais_session_id', sessionId);
        }
        return sessionId;
    }

    exportData() {
        const analyticsData = localStorage.getItem('cais_analytics');
        const feedbackData = localStorage.getItem('cais_feedback');
        
        const exportData = {
            analytics: JSON.parse(analyticsData || '[]'),
            feedback: JSON.parse(feedbackData || '[]'),
            exportDate: new Date().toISOString()
        };

        const blob = new Blob([JSON.stringify(exportData, null, 2)], { type: 'application/json' });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `cais_documentation_data_${new Date().toISOString().split('T')[0]}.json`;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        URL.revokeObjectURL(url);
    }

    clearCache() {
        if (confirm('Are you sure you want to clear all analytics data?')) {
            localStorage.removeItem('cais_analytics');
            localStorage.removeItem('cais_feedback');
            this.showSuccess('Analytics data cleared successfully');
        }
    }

    showError(message) {
        this.showNotification(message, 'error');
    }

    showSuccess(message) {
        this.showNotification(message, 'success');
    }

    showNotification(message, type) {
        const notification = document.createElement('div');
        notification.style.cssText = `
            position: fixed;
            top: 20px;
            right: 20px;
            background: ${type === 'error' ? '#dc3545' : '#28a745'};
            color: white;
            padding: 15px 20px;
            border-radius: 6px;
            z-index: 4000;
            box-shadow: 0 2px 10px rgba(0,0,0,0.2);
        `;
        notification.textContent = message;
        document.body.appendChild(notification);

        setTimeout(() => {
            notification.remove();
        }, 3000);
    }
}

// Initialize analytics dashboard when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    new AnalyticsDashboard();
});
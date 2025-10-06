#!/usr/bin/env python3
"""
Documentation Analytics Dashboard
Simple web dashboard for viewing documentation analytics and feedback.
"""

import json
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List
import argparse

try:
    from flask import Flask, render_template_string, jsonify, request
    FLASK_AVAILABLE = True
except ImportError:
    FLASK_AVAILABLE = False
    print("Flask not available. Install with: pip install flask")

class AnalyticsDashboard:
    """Simple analytics dashboard for documentation metrics."""
    
    def __init__(self, db_path: str = "docs_analytics.db"):
        self.db_path = db_path
        self.app = Flask(__name__) if FLASK_AVAILABLE else None
        self.setup_routes()
    
    def setup_routes(self):
        """Setup Flask routes for the dashboard."""
        if not self.app:
            return
            
        @self.app.route('/')
        def dashboard():
            return render_template_string(self.get_dashboard_template())
        
        @self.app.route('/api/metrics')
        def api_metrics():
            return jsonify(self.get_metrics())
        
        @self.app.route('/api/feedback')
        def api_feedback():
            return jsonify(self.get_feedback_data())
        
        @self.app.route('/api/link-health')
        def api_link_health():
            return jsonify(self.get_link_health())
    
    def get_metrics(self) -> Dict:
        """Get comprehensive metrics from database."""
        if not Path(self.db_path).exists():
            return {'error': 'Database not found. Run monitoring script first.'}
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Get recent analytics
        metrics = {
            'overview': {},
            'content': {},
            'links': {},
            'trends': {}
        }
        
        try:
            # Overview metrics
            cursor.execute('''
                SELECT COUNT(*) as total_pages,
                       SUM(word_count) as total_words,
                       AVG(word_count) as avg_words,
                       SUM(code_blocks_count) as total_code_blocks
                FROM page_analytics
                WHERE analyzed_at > datetime('now', '-7 days')
            ''')
            
            overview = cursor.fetchone()
            if overview:
                metrics['overview'] = {
                    'total_pages': overview[0] or 0,
                    'total_words': overview[1] or 0,
                    'avg_words_per_page': round(overview[2] or 0, 1),
                    'total_code_blocks': overview[3] or 0
                }
            
            # Link health
            cursor.execute('''
                SELECT COUNT(*) as total_checks,
                       SUM(CASE WHEN error_message IS NOT NULL THEN 1 ELSE 0 END) as failed_checks
                FROM link_checks
                WHERE checked_at > datetime('now', '-7 days')
            ''')
            
            link_health = cursor.fetchone()
            if link_health and link_health[0] > 0:
                metrics['links'] = {
                    'total_checks': link_health[0],
                    'failed_checks': link_health[1] or 0,
                    'success_rate': round(((link_health[0] - (link_health[1] or 0)) / link_health[0]) * 100, 1)
                }
            else:
                metrics['links'] = {
                    'total_checks': 0,
                    'failed_checks': 0,
                    'success_rate': 100
                }
            
            # Content trends (last 30 days)
            cursor.execute('''
                SELECT DATE(analyzed_at) as date,
                       COUNT(*) as pages_analyzed,
                       SUM(word_count) as words_added
                FROM page_analytics
                WHERE analyzed_at > datetime('now', '-30 days')
                GROUP BY DATE(analyzed_at)
                ORDER BY date
            ''')
            
            trends = cursor.fetchall()
            metrics['trends'] = {
                'dates': [row[0] for row in trends],
                'pages': [row[1] for row in trends],
                'words': [row[2] for row in trends]
            }
            
        except Exception as e:
            metrics['error'] = str(e)
        finally:
            conn.close()
        
        return metrics
    
    def get_feedback_data(self) -> Dict:
        """Get feedback data from localStorage (simulated)."""
        # In a real implementation, this would come from a backend service
        return {
            'total_feedback': 0,
            'average_rating': 0,
            'recent_comments': [],
            'feedback_by_section': {}
        }
    
    def get_link_health(self) -> Dict:
        """Get detailed link health information."""
        if not Path(self.db_path).exists():
            return {'error': 'Database not found'}
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            # Get broken links
            cursor.execute('''
                SELECT url, error_message, checked_at
                FROM link_checks
                WHERE error_message IS NOT NULL
                ORDER BY checked_at DESC
                LIMIT 50
            ''')
            
            broken_links = [
                {
                    'url': row[0],
                    'error': row[1],
                    'checked_at': row[2]
                }
                for row in cursor.fetchall()
            ]
            
            # Get link check history
            cursor.execute('''
                SELECT DATE(checked_at) as date,
                       COUNT(*) as total_checks,
                       SUM(CASE WHEN error_message IS NOT NULL THEN 1 ELSE 0 END) as failed_checks
                FROM link_checks
                WHERE checked_at > datetime('now', '-30 days')
                GROUP BY DATE(checked_at)
                ORDER BY date
            ''')
            
            history = cursor.fetchall()
            
            return {
                'broken_links': broken_links,
                'history': {
                    'dates': [row[0] for row in history],
                    'total_checks': [row[1] for row in history],
                    'failed_checks': [row[2] for row in history]
                }
            }
            
        except Exception as e:
            return {'error': str(e)}
        finally:
            conn.close()
    
    def get_dashboard_template(self) -> str:
        """Get HTML template for the dashboard."""
        return '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Documentation Analytics Dashboard</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            margin: 0;
            padding: 20px;
            background-color: #f5f5f5;
        }
        .container {
            max-width: 1200px;
            margin: 0 auto;
        }
        .header {
            background: white;
            padding: 20px;
            border-radius: 8px;
            margin-bottom: 20px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        .metrics-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 20px;
            margin-bottom: 20px;
        }
        .metric-card {
            background: white;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        .metric-value {
            font-size: 2em;
            font-weight: bold;
            color: #2c3e50;
        }
        .metric-label {
            color: #7f8c8d;
            margin-top: 5px;
        }
        .chart-container {
            background: white;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            margin-bottom: 20px;
        }
        .status-good { color: #27ae60; }
        .status-warning { color: #f39c12; }
        .status-error { color: #e74c3c; }
        .broken-links {
            background: white;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        .broken-link {
            padding: 10px;
            border-left: 4px solid #e74c3c;
            margin: 10px 0;
            background: #fdf2f2;
        }
        .loading {
            text-align: center;
            padding: 40px;
            color: #7f8c8d;
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>📊 Documentation Analytics Dashboard</h1>
            <p>Real-time monitoring and analytics for CAIS documentation</p>
            <p><small>Last updated: <span id="last-updated">Loading...</span></small></p>
        </div>
        
        <div id="loading" class="loading">
            <p>Loading analytics data...</p>
        </div>
        
        <div id="dashboard-content" style="display: none;">
            <div class="metrics-grid">
                <div class="metric-card">
                    <div class="metric-value" id="total-pages">-</div>
                    <div class="metric-label">Total Pages</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value" id="total-words">-</div>
                    <div class="metric-label">Total Words</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value" id="avg-words">-</div>
                    <div class="metric-label">Avg Words/Page</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value" id="link-health">-</div>
                    <div class="metric-label">Link Health</div>
                </div>
            </div>
            
            <div class="chart-container">
                <h3>Content Growth Trend</h3>
                <canvas id="content-chart" width="400" height="200"></canvas>
            </div>
            
            <div class="broken-links">
                <h3>🔗 Link Health Status</h3>
                <div id="broken-links-list">
                    <p>No broken links found! 🎉</p>
                </div>
            </div>
        </div>
    </div>

    <script>
        async function loadDashboard() {
            try {
                // Load metrics
                const metricsResponse = await fetch('/api/metrics');
                const metrics = await metricsResponse.json();
                
                if (metrics.error) {
                    document.getElementById('loading').innerHTML = 
                        `<p style="color: #e74c3c;">Error: ${metrics.error}</p>`;
                    return;
                }
                
                // Update overview metrics
                document.getElementById('total-pages').textContent = 
                    metrics.overview.total_pages || 0;
                document.getElementById('total-words').textContent = 
                    (metrics.overview.total_words || 0).toLocaleString();
                document.getElementById('avg-words').textContent = 
                    metrics.overview.avg_words_per_page || 0;
                
                // Update link health
                const linkHealth = metrics.links.success_rate || 100;
                const linkHealthElement = document.getElementById('link-health');
                linkHealthElement.textContent = linkHealth + '%';
                linkHealthElement.className = 'metric-value ' + 
                    (linkHealth >= 95 ? 'status-good' : 
                     linkHealth >= 80 ? 'status-warning' : 'status-error');
                
                // Create content trend chart
                if (metrics.trends && metrics.trends.dates.length > 0) {
                    const ctx = document.getElementById('content-chart').getContext('2d');
                    new Chart(ctx, {
                        type: 'line',
                        data: {
                            labels: metrics.trends.dates,
                            datasets: [{
                                label: 'Words Added',
                                data: metrics.trends.words,
                                borderColor: '#3498db',
                                backgroundColor: 'rgba(52, 152, 219, 0.1)',
                                tension: 0.4
                            }]
                        },
                        options: {
                            responsive: true,
                            scales: {
                                y: {
                                    beginAtZero: true
                                }
                            }
                        }
                    });
                }
                
                // Load broken links
                const linkHealthResponse = await fetch('/api/link-health');
                const linkHealth = await linkHealthResponse.json();
                
                if (linkHealth.broken_links && linkHealth.broken_links.length > 0) {
                    const brokenLinksHtml = linkHealth.broken_links.map(link => `
                        <div class="broken-link">
                            <strong>${link.url}</strong><br>
                            <small>Error: ${link.error}</small><br>
                            <small>Checked: ${new Date(link.checked_at).toLocaleString()}</small>
                        </div>
                    `).join('');
                    
                    document.getElementById('broken-links-list').innerHTML = brokenLinksHtml;
                }
                
                // Show dashboard
                document.getElementById('loading').style.display = 'none';
                document.getElementById('dashboard-content').style.display = 'block';
                document.getElementById('last-updated').textContent = new Date().toLocaleString();
                
            } catch (error) {
                document.getElementById('loading').innerHTML = 
                    `<p style="color: #e74c3c;">Error loading dashboard: ${error.message}</p>`;
            }
        }
        
        // Load dashboard on page load
        loadDashboard();
        
        // Refresh every 5 minutes
        setInterval(loadDashboard, 5 * 60 * 1000);
    </script>
</body>
</html>
        '''
    
    def run(self, host: str = '127.0.0.1', port: int = 5000, debug: bool = False):
        """Run the dashboard server."""
        if not FLASK_AVAILABLE:
            print("Flask is required to run the dashboard. Install with: pip install flask")
            return
        
        print(f"Starting analytics dashboard at http://{host}:{port}")
        print("Press Ctrl+C to stop the server")
        
        self.app.run(host=host, port=port, debug=debug)
    
    def generate_static_report(self, output_file: str = "analytics_report.html"):
        """Generate a static HTML report."""
        metrics = self.get_metrics()
        link_health = self.get_link_health()
        
        # Simple static report
        html_content = f'''
<!DOCTYPE html>
<html>
<head>
    <title>Documentation Analytics Report</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 40px; }}
        .metric {{ background: #f8f9fa; padding: 15px; margin: 10px 0; border-radius: 5px; }}
        .error {{ color: #dc3545; }}
        .success {{ color: #28a745; }}
    </style>
</head>
<body>
    <h1>Documentation Analytics Report</h1>
    <p>Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
    
    <h2>Overview</h2>
    <div class="metric">
        <strong>Total Pages:</strong> {metrics.get('overview', {}).get('total_pages', 0)}
    </div>
    <div class="metric">
        <strong>Total Words:</strong> {metrics.get('overview', {}).get('total_words', 0):,}
    </div>
    <div class="metric">
        <strong>Average Words per Page:</strong> {metrics.get('overview', {}).get('avg_words_per_page', 0)}
    </div>
    
    <h2>Link Health</h2>
    <div class="metric">
        <strong>Success Rate:</strong> 
        <span class="{'success' if metrics.get('links', {}).get('success_rate', 100) >= 95 else 'error'}">
            {metrics.get('links', {}).get('success_rate', 100)}%
        </span>
    </div>
    
    <h2>Broken Links</h2>
    {self._format_broken_links_html(link_health.get('broken_links', []))}
    
    <hr>
    <p><small>This report was generated automatically by the documentation monitoring system.</small></p>
</body>
</html>
        '''
        
        with open(output_file, 'w') as f:
            f.write(html_content)
        
        print(f"Static report generated: {output_file}")
    
    def _format_broken_links_html(self, broken_links: List[Dict]) -> str:
        """Format broken links for HTML display."""
        if not broken_links:
            return '<p class="success">No broken links found! 🎉</p>'
        
        html = '<ul>'
        for link in broken_links[:10]:  # Show first 10
            html += f'''
            <li class="error">
                <strong>{link['url']}</strong><br>
                <small>Error: {link['error']}</small>
            </li>
            '''
        
        if len(broken_links) > 10:
            html += f'<li><em>... and {len(broken_links) - 10} more</em></li>'
        
        html += '</ul>'
        return html

def main():
    """Main function for command-line usage."""
    parser = argparse.ArgumentParser(description='Documentation Analytics Dashboard')
    parser.add_argument('--db-path', default='docs_analytics.db',
                       help='Path to analytics database')
    parser.add_argument('--host', default='127.0.0.1',
                       help='Host to bind the server to')
    parser.add_argument('--port', type=int, default=5000,
                       help='Port to bind the server to')
    parser.add_argument('--static-report', action='store_true',
                       help='Generate static HTML report instead of running server')
    parser.add_argument('--output', default='analytics_report.html',
                       help='Output file for static report')
    
    args = parser.parse_args()
    
    dashboard = AnalyticsDashboard(args.db_path)
    
    if args.static_report:
        dashboard.generate_static_report(args.output)
    else:
        dashboard.run(args.host, args.port)

if __name__ == "__main__":
    main()
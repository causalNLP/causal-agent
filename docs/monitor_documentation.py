#!/usr/bin/env python3
"""
Documentation Monitoring Script
Monitors documentation for broken links, build failures, and performance issues
"""

import os
import sys
import json
import time
import requests
import subprocess
from datetime import datetime, timedelta
from pathlib import Path
from urllib.parse import urljoin, urlparse
from bs4 import BeautifulSoup
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

class DocumentationMonitor:
    def __init__(self, config_file='monitor_config.json'):
        self.config = self.load_config(config_file)
        self.base_url = self.config.get('base_url', 'http://localhost:8000')
        self.build_dir = Path(self.config.get('build_dir', 'build/html'))
        self.source_dir = Path(self.config.get('source_dir', 'source'))
        self.alerts = []
        self.metrics = {
            'total_pages': 0,
            'broken_links': 0,
            'build_time': 0,
            'last_build': None,
            'performance_score': 0
        }

    def load_config(self, config_file):
        """Load monitoring configuration"""
        default_config = {
            'base_url': 'http://localhost:8000',
            'build_dir': 'build/html',
            'source_dir': 'source',
            'email_notifications': False,
            'email_recipients': [],
            'smtp_server': 'smtp.gmail.com',
            'smtp_port': 587,
            'smtp_username': '',
            'smtp_password': '',
            'check_interval': 300,  # 5 minutes
            'max_response_time': 5.0,  # seconds
            'excluded_urls': [
                'mailto:',
                'javascript:',
                'tel:',
                '#'
            ]
        }
        
        try:
            with open(config_file, 'r') as f:
                config = json.load(f)
                default_config.update(config)
        except FileNotFoundError:
            print(f"Config file {config_file} not found, using defaults")
            # Create default config file
            with open(config_file, 'w') as f:
                json.dump(default_config, f, indent=2)
        
        return default_config

    def run_monitoring_cycle(self):
        """Run a complete monitoring cycle"""
        print(f"🔍 Starting documentation monitoring cycle at {datetime.now()}")
        
        # Check build status
        self.check_build_status()
        
        # Check for broken links
        self.check_broken_links()
        
        # Check performance
        self.check_performance()
        
        # Generate report
        self.generate_report()
        
        # Send notifications if needed
        if self.alerts and self.config.get('email_notifications'):
            self.send_notifications()
        
        print(f"✅ Monitoring cycle completed at {datetime.now()}")
        return self.metrics, self.alerts

    def check_build_status(self):
        """Check if documentation builds successfully"""
        print("📖 Checking documentation build status...")
        
        start_time = time.time()
        try:
            # Run sphinx build
            result = subprocess.run([
                'python', '-m', 'sphinx', 
                '-b', 'html',
                '-W',  # Treat warnings as errors
                str(self.source_dir),
                str(self.build_dir)
            ], capture_output=True, text=True, timeout=300)
            
            build_time = time.time() - start_time
            self.metrics['build_time'] = build_time
            self.metrics['last_build'] = datetime.now().isoformat()
            
            if result.returncode != 0:
                self.alerts.append({
                    'type': 'error',
                    'category': 'build',
                    'message': f'Documentation build failed with return code {result.returncode}',
                    'details': result.stderr,
                    'timestamp': datetime.now().isoformat()
                })
                print(f"❌ Build failed: {result.stderr}")
            else:
                print(f"✅ Build successful in {build_time:.2f} seconds")
                
                # Check for warnings
                if result.stderr:
                    warning_count = result.stderr.count('WARNING')
                    if warning_count > 0:
                        self.alerts.append({
                            'type': 'warning',
                            'category': 'build',
                            'message': f'Build completed with {warning_count} warnings',
                            'details': result.stderr,
                            'timestamp': datetime.now().isoformat()
                        })
                        print(f"⚠️  Build completed with {warning_count} warnings")
                        
        except subprocess.TimeoutExpired:
            self.alerts.append({
                'type': 'error',
                'category': 'build',
                'message': 'Documentation build timed out after 5 minutes',
                'timestamp': datetime.now().isoformat()
            })
            print("❌ Build timed out")
        except Exception as e:
            self.alerts.append({
                'type': 'error',
                'category': 'build',
                'message': f'Build error: {str(e)}',
                'timestamp': datetime.now().isoformat()
            })
            print(f"❌ Build error: {e}")

    def check_broken_links(self):
        """Check for broken internal and external links"""
        print("🔗 Checking for broken links...")
        
        if not self.build_dir.exists():
            self.alerts.append({
                'type': 'error',
                'category': 'links',
                'message': 'Build directory does not exist, cannot check links',
                'timestamp': datetime.now().isoformat()
            })
            return
        
        broken_links = []
        total_links = 0
        
        # Find all HTML files
        html_files = list(self.build_dir.rglob('*.html'))
        self.metrics['total_pages'] = len(html_files)
        
        for html_file in html_files:
            try:
                with open(html_file, 'r', encoding='utf-8') as f:
                    soup = BeautifulSoup(f.read(), 'html.parser')
                
                # Check all links
                links = soup.find_all('a', href=True)
                for link in links:
                    href = link['href']
                    total_links += 1
                    
                    # Skip excluded URLs
                    if any(excluded in href for excluded in self.config['excluded_urls']):
                        continue
                    
                    # Check internal links
                    if href.startswith('/') or not href.startswith('http'):
                        if self.is_broken_internal_link(href, html_file):
                            broken_links.append({
                                'url': href,
                                'source_file': str(html_file.relative_to(self.build_dir)),
                                'type': 'internal',
                                'text': link.get_text().strip()[:50]
                            })
                    
                    # Check external links (sample only to avoid overwhelming servers)
                    elif href.startswith('http') and total_links % 10 == 0:  # Check every 10th external link
                        if self.is_broken_external_link(href):
                            broken_links.append({
                                'url': href,
                                'source_file': str(html_file.relative_to(self.build_dir)),
                                'type': 'external',
                                'text': link.get_text().strip()[:50]
                            })
                            
            except Exception as e:
                print(f"Error checking links in {html_file}: {e}")
        
        self.metrics['broken_links'] = len(broken_links)
        
        if broken_links:
            self.alerts.append({
                'type': 'warning' if len(broken_links) < 5 else 'error',
                'category': 'links',
                'message': f'Found {len(broken_links)} broken links',
                'details': broken_links[:10],  # Include first 10 broken links
                'timestamp': datetime.now().isoformat()
            })
            print(f"⚠️  Found {len(broken_links)} broken links")
        else:
            print("✅ No broken links found")

    def is_broken_internal_link(self, href, source_file):
        """Check if an internal link is broken"""
        try:
            if href.startswith('#'):
                return False  # Skip anchor links for now
            
            if href.startswith('/'):
                # Absolute path from root
                target_path = self.build_dir / href.lstrip('/')
            else:
                # Relative path
                target_path = source_file.parent / href
            
            # Remove anchor
            if '#' in str(target_path):
                target_path = Path(str(target_path).split('#')[0])
            
            # Check if file exists
            if target_path.suffix == '':
                # Try adding .html
                target_path = target_path / 'index.html'
                if not target_path.exists():
                    target_path = target_path.parent.with_suffix('.html')
            
            return not target_path.exists()
            
        except Exception:
            return True  # Assume broken if we can't check

    def is_broken_external_link(self, url):
        """Check if an external link is broken"""
        try:
            response = requests.head(url, timeout=10, allow_redirects=True)
            return response.status_code >= 400
        except Exception:
            return True

    def check_performance(self):
        """Check documentation performance metrics"""
        print("⚡ Checking performance metrics...")
        
        try:
            # Check if local server is running
            response = requests.get(self.base_url, timeout=self.config['max_response_time'])
            response_time = response.elapsed.total_seconds()
            
            if response_time > self.config['max_response_time']:
                self.alerts.append({
                    'type': 'warning',
                    'category': 'performance',
                    'message': f'Slow response time: {response_time:.2f}s (threshold: {self.config["max_response_time"]}s)',
                    'timestamp': datetime.now().isoformat()
                })
                print(f"⚠️  Slow response time: {response_time:.2f}s")
            
            # Calculate performance score (0-100)
            performance_score = max(0, 100 - (response_time * 20))
            self.metrics['performance_score'] = performance_score
            
            print(f"✅ Performance score: {performance_score:.1f}/100")
            
        except requests.exceptions.RequestException as e:
            self.alerts.append({
                'type': 'error',
                'category': 'performance',
                'message': f'Cannot connect to documentation server: {str(e)}',
                'timestamp': datetime.now().isoformat()
            })
            print(f"❌ Cannot connect to server: {e}")

    def generate_report(self):
        """Generate monitoring report"""
        report = {
            'timestamp': datetime.now().isoformat(),
            'metrics': self.metrics,
            'alerts': self.alerts,
            'summary': {
                'total_alerts': len(self.alerts),
                'error_count': len([a for a in self.alerts if a['type'] == 'error']),
                'warning_count': len([a for a in self.alerts if a['type'] == 'warning']),
                'status': 'healthy' if not any(a['type'] == 'error' for a in self.alerts) else 'issues_detected'
            }
        }
        
        # Save report to file
        report_file = Path('monitoring_report.json')
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"📊 Report saved to {report_file}")
        return report

    def send_notifications(self):
        """Send email notifications for alerts"""
        if not self.config.get('email_notifications') or not self.config.get('email_recipients'):
            return
        
        try:
            # Create email content
            subject = f"CAIS Documentation Alert - {len(self.alerts)} issues detected"
            
            body = f"""
Documentation Monitoring Report
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

SUMMARY:
- Total Pages: {self.metrics['total_pages']}
- Broken Links: {self.metrics['broken_links']}
- Build Time: {self.metrics['build_time']:.2f}s
- Performance Score: {self.metrics['performance_score']:.1f}/100

ALERTS:
"""
            
            for alert in self.alerts:
                body += f"\n[{alert['type'].upper()}] {alert['category']}: {alert['message']}\n"
                if 'details' in alert:
                    body += f"Details: {str(alert['details'])[:200]}...\n"
            
            body += f"\nFull report available at: {self.base_url}\n"
            
            # Send email
            msg = MIMEMultipart()
            msg['From'] = self.config['smtp_username']
            msg['To'] = ', '.join(self.config['email_recipients'])
            msg['Subject'] = subject
            
            msg.attach(MIMEText(body, 'plain'))
            
            server = smtplib.SMTP(self.config['smtp_server'], self.config['smtp_port'])
            server.starttls()
            server.login(self.config['smtp_username'], self.config['smtp_password'])
            server.send_message(msg)
            server.quit()
            
            print(f"📧 Notifications sent to {len(self.config['email_recipients'])} recipients")
            
        except Exception as e:
            print(f"❌ Failed to send notifications: {e}")

    def run_continuous_monitoring(self):
        """Run continuous monitoring with specified interval"""
        print(f"🔄 Starting continuous monitoring (interval: {self.config['check_interval']}s)")
        
        while True:
            try:
                self.run_monitoring_cycle()
                print(f"😴 Sleeping for {self.config['check_interval']} seconds...")
                time.sleep(self.config['check_interval'])
            except KeyboardInterrupt:
                print("\n🛑 Monitoring stopped by user")
                break
            except Exception as e:
                print(f"❌ Monitoring error: {e}")
                time.sleep(60)  # Wait 1 minute before retrying

def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Monitor CAIS documentation')
    parser.add_argument('--continuous', action='store_true', help='Run continuous monitoring')
    parser.add_argument('--config', default='monitor_config.json', help='Configuration file')
    parser.add_argument('--base-url', help='Base URL for documentation')
    
    args = parser.parse_args()
    
    monitor = DocumentationMonitor(args.config)
    
    if args.base_url:
        monitor.base_url = args.base_url
    
    if args.continuous:
        monitor.run_continuous_monitoring()
    else:
        metrics, alerts = monitor.run_monitoring_cycle()
        
        # Print summary
        print(f"\n📊 MONITORING SUMMARY:")
        print(f"   Total Pages: {metrics['total_pages']}")
        print(f"   Broken Links: {metrics['broken_links']}")
        print(f"   Build Time: {metrics['build_time']:.2f}s")
        print(f"   Performance Score: {metrics['performance_score']:.1f}/100")
        print(f"   Alerts: {len(alerts)}")
        
        if alerts:
            print(f"\n⚠️  ALERTS:")
            for alert in alerts:
                print(f"   [{alert['type'].upper()}] {alert['message']}")
        
        # Exit with error code if there are critical issues
        if any(alert['type'] == 'error' for alert in alerts):
            sys.exit(1)

if __name__ == '__main__':
    main()
#!/usr/bin/env python3
"""
Setup script for documentation monitoring and analytics system.
"""

import os
import sys
import subprocess
from pathlib import Path

def install_dependencies():
    """Install required dependencies for monitoring."""
    print("📦 Installing monitoring dependencies...")
    
    dependencies = [
        'requests',
        'flask',
        'sqlite3'  # Usually built-in with Python
    ]
    
    for dep in dependencies:
        try:
            __import__(dep)
            print(f"✅ {dep} already installed")
        except ImportError:
            print(f"📥 Installing {dep}...")
            subprocess.check_call([sys.executable, '-m', 'pip', 'install', dep])

def setup_database():
    """Initialize the monitoring database."""
    print("🗄️ Setting up monitoring database...")
    
    from monitor_documentation import DocumentationMonitor
    monitor = DocumentationMonitor()
    print("✅ Database initialized")

def run_initial_monitoring():
    """Run initial monitoring to populate database."""
    print("🔍 Running initial monitoring...")
    
    from monitor_documentation import DocumentationMonitor
    monitor = DocumentationMonitor()
    
    # Run full monitoring
    report = monitor.generate_report()
    
    print("✅ Initial monitoring complete")
    print(f"📊 Report saved: docs_monitoring_report.json")
    
    # Print summary
    if 'summary' in report.get('link_check', {}):
        link_summary = report['link_check']['summary']
        print(f"🔗 Links checked: {link_summary.get('total_links', 0)}")
        print(f"❌ Broken links: {link_summary.get('broken_links', 0)}")
    
    if 'summary' in report.get('content_analysis', {}):
        content_summary = report['content_analysis']['summary']
        print(f"📄 Pages analyzed: {content_summary.get('total_pages', 0)}")
        print(f"📝 Total words: {content_summary.get('total_words', 0):,}")

def setup_cron_job():
    """Setup cron job for regular monitoring (Linux/macOS only)."""
    if os.name != 'posix':
        print("⚠️ Cron job setup only available on Linux/macOS")
        return
    
    print("⏰ Setting up cron job for daily monitoring...")
    
    script_path = Path(__file__).parent.absolute() / "monitor_documentation.py"
    cron_command = f"0 6 * * * cd {Path.cwd()} && python {script_path} --full-report"
    
    print(f"Add this line to your crontab (run 'crontab -e'):")
    print(f"{cron_command}")
    print("This will run monitoring daily at 6 AM")

def create_monitoring_config():
    """Create configuration file for monitoring."""
    print("⚙️ Creating monitoring configuration...")
    
    config = {
        "base_url": "https://causal-ai-scientist.readthedocs.io",
        "local_build_path": "build/html",
        "check_external_links": True,
        "monitoring_frequency": "daily",
        "alert_thresholds": {
            "broken_links_max": 10,
            "min_pages": 5,
            "min_success_rate": 95
        },
        "notifications": {
            "email": None,
            "slack_webhook": None,
            "github_issues": True
        }
    }
    
    import json
    with open('monitoring_config.json', 'w') as f:
        json.dump(config, f, indent=2)
    
    print("✅ Configuration saved to monitoring_config.json")
    print("📝 Edit this file to customize monitoring settings")

def main():
    """Main setup function."""
    print("🚀 Setting up Documentation Monitoring System")
    print("=" * 50)
    
    try:
        # Check if we're in the docs directory
        if not Path('source').exists():
            print("❌ Please run this script from the docs/ directory")
            sys.exit(1)
        
        # Install dependencies
        install_dependencies()
        
        # Setup database
        setup_database()
        
        # Create config
        create_monitoring_config()
        
        # Run initial monitoring
        run_initial_monitoring()
        
        # Setup cron (optional)
        setup_cron_job()
        
        print("\n🎉 Monitoring system setup complete!")
        print("\n📋 Next steps:")
        print("1. Edit monitoring_config.json to customize settings")
        print("2. Replace 'G-XXXXXXXXXX' in conf.py with your Google Analytics ID")
        print("3. Run 'python analytics_dashboard.py' to start the dashboard")
        print("4. Run 'python monitor_documentation.py --full-report' for manual monitoring")
        print("5. Set up the cron job for automated daily monitoring")
        
        print("\n🌐 Dashboard URLs:")
        print("- Local dashboard: http://localhost:5000")
        print("- Static report: monitoring_report.html")
        
    except Exception as e:
        print(f"❌ Setup failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
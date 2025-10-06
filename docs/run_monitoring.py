#!/usr/bin/env python3
"""
Simple script to run documentation monitoring
"""

import sys
import os
from pathlib import Path

# Add the docs directory to Python path
docs_dir = Path(__file__).parent
sys.path.insert(0, str(docs_dir))

from monitor_documentation import DocumentationMonitor

def main():
    """Run monitoring with simple interface"""
    print("🚀 CAIS Documentation Monitoring")
    print("=" * 40)
    
    monitor = DocumentationMonitor()
    
    try:
        metrics, alerts = monitor.run_monitoring_cycle()
        
        # Print results
        print("\n📊 RESULTS:")
        print(f"   ✅ Total Pages: {metrics['total_pages']}")
        print(f"   🔗 Broken Links: {metrics['broken_links']}")
        print(f"   ⏱️  Build Time: {metrics['build_time']:.2f}s")
        print(f"   ⚡ Performance: {metrics['performance_score']:.1f}/100")
        
        if alerts:
            print(f"\n⚠️  ALERTS ({len(alerts)}):")
            for i, alert in enumerate(alerts, 1):
                icon = "❌" if alert['type'] == 'error' else "⚠️"
                print(f"   {icon} {i}. [{alert['category']}] {alert['message']}")
        else:
            print("\n✅ No alerts - documentation is healthy!")
        
        print(f"\n📄 Full report saved to: monitoring_report.json")
        
        return len([a for a in alerts if a['type'] == 'error'])
        
    except KeyboardInterrupt:
        print("\n🛑 Monitoring cancelled by user")
        return 1
    except Exception as e:
        print(f"\n❌ Monitoring failed: {e}")
        return 1

if __name__ == '__main__':
    exit_code = main()
    sys.exit(exit_code)
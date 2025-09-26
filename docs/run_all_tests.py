#!/usr/bin/env python3
"""
Comprehensive test runner for CAIS documentation performance and accessibility.
Runs all tests including responsive design, accessibility compliance, and performance validation.
"""

import os
import sys
import json
import time
import logging
import subprocess
from pathlib import Path
from typing import Dict, List, Optional
import argparse

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ComprehensiveTestRunner:
    """Runs all performance and accessibility tests"""
    
    def __init__(self, docs_dir: str = "docs", base_url: str = "http://localhost:8000"):
        self.docs_dir = Path(docs_dir)
        self.build_dir = self.docs_dir / "build"
        self.base_url = base_url
        self.results_dir = Path("test_results")
        self.results_dir.mkdir(exist_ok=True)
        
    def run_all_tests(self, skip_build: bool = False, skip_optimization: bool = False) -> Dict[str, any]:
        """Run all tests in sequence"""
        logger.info("Starting comprehensive testing suite...")
        
        start_time = time.time()
        results = {
            'start_time': start_time,
            'build': None,
            'optimization': None,
            'validation': None,
            'responsive_design': None,
            'performance_audit': None,
            'summary': {}
        }
        
        try:
            # Step 1: Build documentation
            if not skip_build:
                logger.info("Step 1: Building documentation...")
                results['build'] = self.build_documentation()
                if not results['build']['success']:
                    logger.error("Documentation build failed. Stopping tests.")
                    return results
            else:
                logger.info("Step 1: Skipping documentation build")
                results['build'] = {'success': True, 'skipped': True}
            
            # Step 2: Run performance optimizations
            if not skip_optimization:
                logger.info("Step 2: Running performance optimizations...")
                results['optimization'] = self.run_performance_optimization()
            else:
                logger.info("Step 2: Skipping performance optimization")
                results['optimization'] = {'success': True, 'skipped': True}
            
            # Step 3: Run comprehensive validation suite
            logger.info("Step 3: Running comprehensive validation suite...")
            results['validation'] = self.run_validation_suite()
            
            # Step 4: Validate optimizations
            logger.info("Step 4: Validating optimizations...")
            results['optimization_validation'] = self.validate_optimizations()
            
            # Step 5: Run responsive design tests
            logger.info("Step 5: Running responsive design tests...")
            results['responsive_design'] = self.run_responsive_design_tests()
            
            # Step 6: Run performance audit
            logger.info("Step 6: Running performance audit...")
            results['performance_audit'] = self.run_performance_audit()
            
            # Generate summary
            results['summary'] = self.generate_summary(results)
            results['end_time'] = time.time()
            results['total_duration'] = results['end_time'] - start_time
            
            # Generate comprehensive report
            self.generate_comprehensive_report(results)
            
        except Exception as e:
            logger.error(f"Test suite failed with error: {str(e)}")
            results['error'] = str(e)
            results['success'] = False
        
        return results
    
    def build_documentation(self) -> Dict[str, any]:
        """Build the documentation"""
        try:
            # Change to docs directory
            original_cwd = os.getcwd()
            os.chdir(self.docs_dir)
            
            # Run Sphinx build
            result = subprocess.run(
                ['make', 'html'],
                capture_output=True,
                text=True,
                timeout=300  # 5 minute timeout
            )
            
            os.chdir(original_cwd)
            
            return {
                'success': result.returncode == 0,
                'stdout': result.stdout,
                'stderr': result.stderr,
                'return_code': result.returncode
            }
            
        except subprocess.TimeoutExpired:
            return {
                'success': False,
                'error': 'Build timed out after 5 minutes'
            }
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }
    
    def run_performance_optimization(self) -> Dict[str, any]:
        """Run performance optimization script"""
        try:
            result = subprocess.run([
                sys.executable, 'optimize_performance.py',
                '--docs-dir', str(self.docs_dir),
                '--build-dir', str(self.build_dir)
            ], capture_output=True, text=True, timeout=600)  # 10 minute timeout
            
            return {
                'success': result.returncode == 0,
                'stdout': result.stdout,
                'stderr': result.stderr,
                'return_code': result.returncode
            }
            
        except subprocess.TimeoutExpired:
            return {
                'success': False,
                'error': 'Optimization timed out after 10 minutes'
            }
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }
    
    def run_validation_suite(self) -> Dict[str, any]:
        """Run the comprehensive validation suite"""
        try:
            result = subprocess.run([
                sys.executable, 'test_all_validation.py', str(self.docs_dir)
            ], capture_output=True, text=True, timeout=1800)  # 30 minute timeout
            
            return {
                'success': result.returncode == 0,
                'stdout': result.stdout,
                'stderr': result.stderr,
                'return_code': result.returncode
            }
            
        except subprocess.TimeoutExpired:
            return {
                'success': False,
                'error': 'Validation suite timed out after 30 minutes'
            }
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }
    
    def validate_optimizations(self) -> Dict[str, any]:
        """Validate that optimizations are working"""
        try:
            result = subprocess.run([
                sys.executable, 'validate_optimizations.py',
                '--build-dir', str(self.build_dir)
            ], capture_output=True, text=True, timeout=300)  # 5 minute timeout
            
            return {
                'success': result.returncode == 0,
                'stdout': result.stdout,
                'stderr': result.stderr,
                'return_code': result.returncode
            }
            
        except subprocess.TimeoutExpired:
            return {
                'success': False,
                'error': 'Validation timed out after 5 minutes'
            }
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }
    
    def run_responsive_design_tests(self) -> Dict[str, any]:
        """Run responsive design and accessibility tests"""
        try:
            # Check if server is running
            if not self.is_server_running():
                logger.warning("Documentation server not running. Starting local server...")
                server_process = self.start_local_server()
                time.sleep(5)  # Give server time to start
            else:
                server_process = None
            
            try:
                result = subprocess.run([
                    sys.executable, 'test_responsive_design.py',
                    '--base-url', self.base_url,
                    '--output-dir', str(self.results_dir)
                ], capture_output=True, text=True, timeout=1800)  # 30 minute timeout
                
                return {
                    'success': result.returncode == 0,
                    'stdout': result.stdout,
                    'stderr': result.stderr,
                    'return_code': result.returncode
                }
                
            finally:
                if server_process:
                    server_process.terminate()
                    
        except subprocess.TimeoutExpired:
            return {
                'success': False,
                'error': 'Responsive design tests timed out after 30 minutes'
            }
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }
    
    def run_performance_audit(self) -> Dict[str, any]:
        """Run performance audit using available tools"""
        results = {
            'lighthouse': None,
            'pagespeed': None,
            'manual_checks': None
        }
        
        # Try to run Lighthouse if available
        try:
            lighthouse_result = subprocess.run([
                'lighthouse', self.base_url,
                '--output=json',
                '--output-path=' + str(self.results_dir / 'lighthouse_report.json'),
                '--chrome-flags="--headless"',
                '--quiet'
            ], capture_output=True, text=True, timeout=300)
            
            results['lighthouse'] = {
                'success': lighthouse_result.returncode == 0,
                'available': True
            }
            
        except (subprocess.TimeoutExpired, FileNotFoundError):
            results['lighthouse'] = {
                'success': False,
                'available': False,
                'note': 'Lighthouse not available or timed out'
            }
        
        # Manual performance checks
        results['manual_checks'] = self.run_manual_performance_checks()
        
        return results
    
    def run_manual_performance_checks(self) -> Dict[str, any]:
        """Run manual performance checks"""
        checks = {
            'file_sizes': self.check_file_sizes(),
            'compression': self.check_compression_ratios(),
            'image_optimization': self.check_image_optimization(),
            'css_optimization': self.check_css_optimization(),
            'js_optimization': self.check_js_optimization()
        }
        
        return {
            'success': all(check.get('success', False) for check in checks.values()),
            'checks': checks
        }
    
    def check_file_sizes(self) -> Dict[str, any]:
        """Check file sizes for performance"""
        large_files = []
        total_size = 0
        file_count = 0
        
        for file_path in self.build_dir.rglob('*'):
            if file_path.is_file():
                size = file_path.stat().st_size
                total_size += size
                file_count += 1
                
                # Flag large files
                if size > 1024 * 1024:  # > 1MB
                    large_files.append({
                        'path': str(file_path.relative_to(self.build_dir)),
                        'size': size,
                        'size_mb': size / (1024 * 1024)
                    })
        
        return {
            'success': len(large_files) < 5,  # Less than 5 files > 1MB
            'total_size': total_size,
            'total_size_mb': total_size / (1024 * 1024),
            'file_count': file_count,
            'large_files': large_files[:10]  # Top 10 largest files
        }
    
    def check_compression_ratios(self) -> Dict[str, any]:
        """Check compression ratios"""
        compression_stats = []
        
        for file_path in self.build_dir.rglob('*.gz'):
            original_path = file_path.with_suffix('')
            if original_path.exists():
                original_size = original_path.stat().st_size
                compressed_size = file_path.stat().st_size
                ratio = compressed_size / original_size if original_size > 0 else 1
                
                compression_stats.append({
                    'file': str(original_path.relative_to(self.build_dir)),
                    'original_size': original_size,
                    'compressed_size': compressed_size,
                    'ratio': ratio,
                    'savings': original_size - compressed_size
                })
        
        avg_ratio = sum(stat['ratio'] for stat in compression_stats) / len(compression_stats) if compression_stats else 1
        
        return {
            'success': avg_ratio < 0.7,  # Average compression ratio < 70%
            'compressed_files': len(compression_stats),
            'average_ratio': avg_ratio,
            'total_savings': sum(stat['savings'] for stat in compression_stats),
            'stats': compression_stats[:10]
        }
    
    def check_image_optimization(self) -> Dict[str, any]:
        """Check image optimization"""
        image_stats = {
            'total_images': 0,
            'webp_images': 0,
            'large_images': [],
            'total_size': 0
        }
        
        image_extensions = ['.jpg', '.jpeg', '.png', '.gif', '.webp', '.svg']
        
        for ext in image_extensions:
            for image_path in self.build_dir.rglob(f'*{ext}'):
                size = image_path.stat().st_size
                image_stats['total_images'] += 1
                image_stats['total_size'] += size
                
                if ext == '.webp':
                    image_stats['webp_images'] += 1
                
                if size > 500 * 1024:  # > 500KB
                    image_stats['large_images'].append({
                        'path': str(image_path.relative_to(self.build_dir)),
                        'size': size,
                        'size_kb': size / 1024
                    })
        
        return {
            'success': (image_stats['webp_images'] > 0 and 
                       len(image_stats['large_images']) < 5),
            **image_stats
        }
    
    def check_css_optimization(self) -> Dict[str, any]:
        """Check CSS optimization"""
        css_stats = {
            'total_files': 0,
            'total_size': 0,
            'minified_files': 0,
            'large_files': []
        }
        
        for css_path in self.build_dir.rglob('*.css'):
            size = css_path.stat().st_size
            css_stats['total_files'] += 1
            css_stats['total_size'] += size
            
            # Check if minified (basic heuristic)
            with open(css_path, 'r', encoding='utf-8') as f:
                content = f.read()
                if len(content) > 1000 and content.count('\n') < len(content) / 100:
                    css_stats['minified_files'] += 1
            
            if size > 100 * 1024:  # > 100KB
                css_stats['large_files'].append({
                    'path': str(css_path.relative_to(self.build_dir)),
                    'size': size,
                    'size_kb': size / 1024
                })
        
        return {
            'success': (css_stats['minified_files'] > 0 and 
                       len(css_stats['large_files']) < 3),
            **css_stats
        }
    
    def check_js_optimization(self) -> Dict[str, any]:
        """Check JavaScript optimization"""
        js_stats = {
            'total_files': 0,
            'total_size': 0,
            'minified_files': 0,
            'large_files': []
        }
        
        for js_path in self.build_dir.rglob('*.js'):
            size = js_path.stat().st_size
            js_stats['total_files'] += 1
            js_stats['total_size'] += size
            
            # Check if minified (basic heuristic)
            with open(js_path, 'r', encoding='utf-8') as f:
                content = f.read()
                if len(content) > 1000 and content.count('\n') < len(content) / 100:
                    js_stats['minified_files'] += 1
            
            if size > 100 * 1024:  # > 100KB
                js_stats['large_files'].append({
                    'path': str(js_path.relative_to(self.build_dir)),
                    'size': size,
                    'size_kb': size / 1024
                })
        
        return {
            'success': (js_stats['minified_files'] > 0 and 
                       len(js_stats['large_files']) < 3),
            **js_stats
        }
    
    def is_server_running(self) -> bool:
        """Check if documentation server is running"""
        try:
            import requests
            response = requests.get(self.base_url, timeout=5)
            return response.status_code == 200
        except:
            return False
    
    def start_local_server(self):
        """Start local HTTP server for testing"""
        try:
            return subprocess.Popen([
                sys.executable, '-m', 'http.server', '8000'
            ], cwd=self.build_dir, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except Exception as e:
            logger.error(f"Failed to start local server: {str(e)}")
            return None
    
    def generate_summary(self, results: Dict[str, any]) -> Dict[str, any]:
        """Generate test summary"""
        summary = {
            'total_tests': 0,
            'passed_tests': 0,
            'failed_tests': 0,
            'skipped_tests': 0,
            'success_rate': 0,
            'critical_failures': [],
            'warnings': [],
            'recommendations': []
        }
        
        # Count test results
        test_sections = ['build', 'optimization', 'validation', 'optimization_validation', 'responsive_design', 'performance_audit']
        
        for section in test_sections:
            result = results.get(section)
            if result:
                summary['total_tests'] += 1
                
                if result.get('skipped'):
                    summary['skipped_tests'] += 1
                elif result.get('success'):
                    summary['passed_tests'] += 1
                else:
                    summary['failed_tests'] += 1
                    summary['critical_failures'].append(section)
        
        # Calculate success rate
        if summary['total_tests'] > 0:
            summary['success_rate'] = (summary['passed_tests'] / summary['total_tests']) * 100
        
        # Add recommendations based on results
        if results.get('build', {}).get('success') == False:
            summary['recommendations'].append("Fix documentation build errors before proceeding")
        
        if results.get('validation', {}).get('success') == False:
            summary['recommendations'].append("Address optimization validation failures")
        
        if results.get('responsive_design', {}).get('success') == False:
            summary['recommendations'].append("Fix responsive design and accessibility issues")
        
        return summary
    
    def generate_comprehensive_report(self, results: Dict[str, any]):
        """Generate comprehensive HTML report"""
        report_path = self.results_dir / 'comprehensive_test_report.html'
        
        summary = results.get('summary', {})
        duration = results.get('total_duration', 0)
        
        html_content = f'''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Comprehensive Test Report - CAIS Documentation</title>
    <style>
        body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; margin: 0; padding: 20px; background: #f8f9fa; }}
        .container {{ max-width: 1400px; margin: 0 auto; background: white; padding: 30px; border-radius: 8px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }}
        h1 {{ color: #2c3e50; border-bottom: 3px solid #3498db; padding-bottom: 10px; }}
        .summary {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 20px; margin: 20px 0; }}
        .stat-card {{ padding: 20px; border-radius: 8px; text-align: center; color: white; }}
        .stat-card.success {{ background: linear-gradient(135deg, #4CAF50 0%, #45a049 100%); }}
        .stat-card.warning {{ background: linear-gradient(135deg, #ff9800 0%, #f57c00 100%); }}
        .stat-card.error {{ background: linear-gradient(135deg, #f44336 0%, #da190b 100%); }}
        .stat-card.info {{ background: linear-gradient(135deg, #2196F3 0%, #1976D2 100%); }}
        .stat-number {{ font-size: 2em; font-weight: bold; margin-bottom: 5px; }}
        .stat-label {{ font-size: 0.9em; opacity: 0.9; }}
        .test-section {{ margin: 30px 0; padding: 20px; border: 1px solid #e1e8ed; border-radius: 8px; }}
        .test-section.success {{ border-left: 4px solid #4CAF50; }}
        .test-section.failed {{ border-left: 4px solid #f44336; }}
        .test-section.skipped {{ border-left: 4px solid #ffc107; }}
        .test-title {{ color: #2c3e50; margin-bottom: 15px; display: flex; justify-content: space-between; align-items: center; }}
        .status-badge {{ padding: 4px 12px; border-radius: 20px; font-size: 0.8em; font-weight: 600; }}
        .status-badge.success {{ background: #d4edda; color: #155724; }}
        .status-badge.failed {{ background: #f8d7da; color: #721c24; }}
        .status-badge.skipped {{ background: #fff3cd; color: #856404; }}
        .recommendations {{ background: #e7f3ff; border: 1px solid #b3d9ff; border-radius: 6px; padding: 20px; margin: 20px 0; }}
        .recommendations h3 {{ color: #0056b3; margin-top: 0; }}
        .recommendations ul {{ margin: 0; }}
        .recommendations li {{ margin: 8px 0; }}
        .collapsible {{ cursor: pointer; padding: 10px; background: #f8f9fa; border: 1px solid #e9ecef; border-radius: 4px; margin: 10px 0; }}
        .collapsible:hover {{ background: #e9ecef; }}
        .collapsible-content {{ display: none; padding: 15px; background: #ffffff; border: 1px solid #e9ecef; border-top: none; }}
        .collapsible.active + .collapsible-content {{ display: block; }}
        pre {{ background: #f8f9fa; padding: 15px; border-radius: 4px; overflow-x: auto; font-size: 0.9em; }}
        .metric {{ display: flex; justify-content: space-between; padding: 8px 0; border-bottom: 1px solid #f0f0f0; }}
        .metric:last-child {{ border-bottom: none; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>Comprehensive Test Report</h1>
        <p><strong>Generated:</strong> {time.strftime('%Y-%m-%d %H:%M:%S')}</p>
        <p><strong>Duration:</strong> {duration:.1f} seconds</p>
        
        <div class="summary">
            <div class="stat-card {'success' if summary.get('success_rate', 0) >= 80 else 'warning' if summary.get('success_rate', 0) >= 60 else 'error'}">
                <div class="stat-number">{summary.get('success_rate', 0):.1f}%</div>
                <div class="stat-label">Success Rate</div>
            </div>
            <div class="stat-card success">
                <div class="stat-number">{summary.get('passed_tests', 0)}</div>
                <div class="stat-label">Tests Passed</div>
            </div>
            <div class="stat-card {'error' if summary.get('failed_tests', 0) > 0 else 'success'}">
                <div class="stat-number">{summary.get('failed_tests', 0)}</div>
                <div class="stat-label">Tests Failed</div>
            </div>
            <div class="stat-card info">
                <div class="stat-number">{summary.get('total_tests', 0)}</div>
                <div class="stat-label">Total Tests</div>
            </div>
        </div>
'''
        
        # Add recommendations if any
        if summary.get('recommendations'):
            html_content += '''
        <div class="recommendations">
            <h3>Recommendations</h3>
            <ul>'''
            for rec in summary['recommendations']:
                html_content += f'<li>{rec}</li>'
            html_content += '''
            </ul>
        </div>'''
        
        # Add detailed test results
        test_sections = [
            ('build', 'Documentation Build'),
            ('optimization', 'Performance Optimization'),
            ('validation', 'Comprehensive Validation Suite'),
            ('optimization_validation', 'Optimization Validation'),
            ('responsive_design', 'Responsive Design & Accessibility'),
            ('performance_audit', 'Performance Audit')
        ]
        
        for test_key, test_name in test_sections:
            test_result = results.get(test_key, {})
            
            if test_result.get('skipped'):
                status_class = 'skipped'
                status_text = 'SKIPPED'
            elif test_result.get('success'):
                status_class = 'success'
                status_text = 'PASSED'
            else:
                status_class = 'failed'
                status_text = 'FAILED'
            
            html_content += f'''
        <div class="test-section {status_class}">
            <div class="test-title">
                <h2>{test_name}</h2>
                <span class="status-badge {status_class}">{status_text}</span>
            </div>
'''
            
            # Add test details
            if test_result.get('stdout'):
                html_content += f'''
            <div class="collapsible" onclick="toggleCollapsible(this)">
                📄 Output Details
            </div>
            <div class="collapsible-content">
                <pre>{test_result['stdout'][:2000]}{'...' if len(test_result['stdout']) > 2000 else ''}</pre>
            </div>'''
            
            if test_result.get('stderr'):
                html_content += f'''
            <div class="collapsible" onclick="toggleCollapsible(this)">
                ⚠️ Error Details
            </div>
            <div class="collapsible-content">
                <pre>{test_result['stderr'][:2000]}{'...' if len(test_result['stderr']) > 2000 else ''}</pre>
            </div>'''
            
            html_content += '</div>'
        
        html_content += '''
        <script>
            function toggleCollapsible(element) {
                element.classList.toggle('active');
            }
        </script>
    </div>
</body>
</html>'''
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        logger.info(f"Comprehensive report generated: {report_path}")

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Run comprehensive CAIS documentation tests")
    parser.add_argument("--docs-dir", default="docs", help="Documentation directory")
    parser.add_argument("--base-url", default="http://localhost:8000", help="Base URL for testing")
    parser.add_argument("--skip-build", action="store_true", help="Skip documentation build")
    parser.add_argument("--skip-optimization", action="store_true", help="Skip performance optimization")
    parser.add_argument("--quick", action="store_true", help="Run quick tests only (skip responsive design)")
    
    args = parser.parse_args()
    
    # Create test runner
    runner = ComprehensiveTestRunner(args.docs_dir, args.base_url)
    
    logger.info("Starting comprehensive test suite...")
    logger.info(f"Documentation directory: {args.docs_dir}")
    logger.info(f"Base URL: {args.base_url}")
    
    # Run tests
    results = runner.run_all_tests(
        skip_build=args.skip_build,
        skip_optimization=args.skip_optimization
    )
    
    # Print summary
    summary = results.get('summary', {})
    duration = results.get('total_duration', 0)
    
    print(f"\n{'='*80}")
    print("COMPREHENSIVE TEST SUMMARY")
    print(f"{'='*80}")
    print(f"Duration: {duration:.1f} seconds")
    print(f"Total tests: {summary.get('total_tests', 0)}")
    print(f"Passed: {summary.get('passed_tests', 0)}")
    print(f"Failed: {summary.get('failed_tests', 0)}")
    print(f"Skipped: {summary.get('skipped_tests', 0)}")
    print(f"Success rate: {summary.get('success_rate', 0):.1f}%")
    
    if summary.get('critical_failures'):
        print(f"\n❌ Critical failures: {', '.join(summary['critical_failures'])}")
    
    if summary.get('recommendations'):
        print(f"\n💡 Recommendations:")
        for rec in summary['recommendations']:
            print(f"  • {rec}")
    
    print(f"\n📊 Detailed reports available in: test_results/")
    
    # Return appropriate exit code
    if summary.get('failed_tests', 0) > 0:
        return 1
    else:
        print(f"\n✅ All tests completed successfully!")
        return 0

if __name__ == "__main__":
    sys.exit(main())
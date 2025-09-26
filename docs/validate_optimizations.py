#!/usr/bin/env python3
"""
Validation script for performance and accessibility optimizations.
Tests that all optimizations are working correctly.
"""

import os
import sys
import json
import time
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import subprocess
import re

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False
    logger.warning("requests not available. Install with: pip install requests")

class OptimizationValidator:
    """Validates that performance and accessibility optimizations are working"""
    
    def __init__(self, build_dir: str = "docs/build", base_url: str = "http://localhost:8000"):
        self.build_dir = Path(build_dir)
        self.base_url = base_url.rstrip('/')
        
    def validate_all(self) -> Dict[str, any]:
        """Run all validation tests"""
        logger.info("Starting optimization validation...")
        
        results = {
            'file_structure': self.validate_file_structure(),
            'compression': self.validate_compression(),
            'service_worker': self.validate_service_worker(),
            'web_manifest': self.validate_web_manifest(),
            'css_optimization': self.validate_css_optimization(),
            'js_optimization': self.validate_js_optimization(),
            'html_optimization': self.validate_html_optimization(),
            'image_optimization': self.validate_image_optimization(),
            'accessibility_features': self.validate_accessibility_features(),
            'performance_features': self.validate_performance_features(),
        }
        
        # Generate validation report
        self.generate_validation_report(results)
        
        logger.info("Optimization validation completed!")
        return results
    
    def validate_file_structure(self) -> Dict[str, any]:
        """Validate that all required files are present"""
        logger.info("Validating file structure...")
        
        required_files = [
            'sw.js',  # Service worker
            'manifest.json',  # Web manifest
            'offline.html',  # Offline page
            '_static/custom.css',
            '_static/performance_optimizations.css',
            '_static/accessibility_enhancements.js',
            '_static/search_enhancements.js',
            '_static/critical.css',
        ]
        
        results = {
            'missing_files': [],
            'present_files': [],
            'total_required': len(required_files)
        }
        
        for file_path in required_files:
            full_path = self.build_dir / file_path
            if full_path.exists():
                results['present_files'].append(file_path)
            else:
                results['missing_files'].append(file_path)
        
        results['success'] = len(results['missing_files']) == 0
        
        if results['missing_files']:
            logger.warning(f"Missing files: {results['missing_files']}")
        else:
            logger.info("All required files present")
        
        return results
    
    def validate_compression(self) -> Dict[str, any]:
        """Validate that compression is working"""
        logger.info("Validating compression...")
        
        results = {
            'gzip_files': 0,
            'brotli_files': 0,
            'uncompressed_large_files': [],
            'compression_ratios': {}
        }
        
        # Check for compressed versions of large files
        large_files = []
        for ext in ['.html', '.css', '.js']:
            for file_path in self.build_dir.rglob(f'*{ext}'):
                if file_path.stat().st_size > 1024:  # Files larger than 1KB
                    large_files.append(file_path)
        
        for file_path in large_files:
            gzip_path = file_path.with_suffix(file_path.suffix + '.gz')
            brotli_path = file_path.with_suffix(file_path.suffix + '.br')
            
            if gzip_path.exists():
                results['gzip_files'] += 1
                # Calculate compression ratio
                original_size = file_path.stat().st_size
                compressed_size = gzip_path.stat().st_size
                ratio = compressed_size / original_size
                results['compression_ratios'][str(file_path.relative_to(self.build_dir))] = ratio
            else:
                results['uncompressed_large_files'].append(str(file_path.relative_to(self.build_dir)))
            
            if brotli_path.exists():
                results['brotli_files'] += 1
        
        results['success'] = len(results['uncompressed_large_files']) == 0
        
        if results['uncompressed_large_files']:
            logger.warning(f"Large files without compression: {results['uncompressed_large_files'][:5]}")
        else:
            logger.info(f"Compression validated: {results['gzip_files']} gzip files")
        
        return results
    
    def validate_service_worker(self) -> Dict[str, any]:
        """Validate service worker functionality"""
        logger.info("Validating service worker...")
        
        sw_path = self.build_dir / 'sw.js'
        results = {
            'exists': sw_path.exists(),
            'valid_syntax': False,
            'cache_urls_count': 0,
            'has_install_handler': False,
            'has_fetch_handler': False,
            'has_activate_handler': False
        }
        
        if results['exists']:
            try:
                with open(sw_path, 'r', encoding='utf-8') as f:
                    sw_content = f.read()
                
                # Basic syntax validation
                if 'self.addEventListener' in sw_content:
                    results['valid_syntax'] = True
                
                # Check for required event handlers
                results['has_install_handler'] = "'install'" in sw_content
                results['has_fetch_handler'] = "'fetch'" in sw_content
                results['has_activate_handler'] = "'activate'" in sw_content
                
                # Count cached URLs
                cache_urls_match = re.search(r'STATIC_CACHE_URLS\s*=\s*\[(.*?)\]', sw_content, re.DOTALL)
                if cache_urls_match:
                    urls = cache_urls_match.group(1)
                    results['cache_urls_count'] = len(re.findall(r'"[^"]*"', urls))
                
            except Exception as e:
                logger.error(f"Error validating service worker: {str(e)}")
        
        results['success'] = (results['exists'] and results['valid_syntax'] and 
                            results['has_install_handler'] and results['has_fetch_handler'])
        
        if results['success']:
            logger.info(f"Service worker validated: {results['cache_urls_count']} cached URLs")
        else:
            logger.warning("Service worker validation failed")
        
        return results
    
    def validate_web_manifest(self) -> Dict[str, any]:
        """Validate web manifest"""
        logger.info("Validating web manifest...")
        
        manifest_path = self.build_dir / 'manifest.json'
        results = {
            'exists': manifest_path.exists(),
            'valid_json': False,
            'has_required_fields': False,
            'manifest_data': {}
        }
        
        if results['exists']:
            try:
                with open(manifest_path, 'r', encoding='utf-8') as f:
                    manifest_data = json.load(f)
                
                results['valid_json'] = True
                results['manifest_data'] = manifest_data
                
                # Check for required fields
                required_fields = ['name', 'short_name', 'start_url', 'display']
                results['has_required_fields'] = all(field in manifest_data for field in required_fields)
                
            except json.JSONDecodeError as e:
                logger.error(f"Invalid JSON in manifest: {str(e)}")
            except Exception as e:
                logger.error(f"Error validating manifest: {str(e)}")
        
        results['success'] = results['exists'] and results['valid_json'] and results['has_required_fields']
        
        if results['success']:
            logger.info("Web manifest validated successfully")
        else:
            logger.warning("Web manifest validation failed")
        
        return results
    
    def validate_css_optimization(self) -> Dict[str, any]:
        """Validate CSS optimization"""
        logger.info("Validating CSS optimization...")
        
        results = {
            'css_files': 0,
            'minified_files': 0,
            'total_size': 0,
            'has_critical_css': False,
            'has_performance_css': False
        }
        
        # Check CSS files
        css_files = list(self.build_dir.rglob('*.css'))
        results['css_files'] = len(css_files)
        
        for css_file in css_files:
            file_size = css_file.stat().st_size
            results['total_size'] += file_size
            
            # Check if file appears minified (basic check)
            with open(css_file, 'r', encoding='utf-8') as f:
                content = f.read()
                # If file has very few newlines relative to size, it's likely minified
                if len(content) > 1000 and content.count('\n') < len(content) / 100:
                    results['minified_files'] += 1
        
        # Check for specific optimization files
        critical_css_path = self.build_dir / '_static' / 'critical.css'
        performance_css_path = self.build_dir / '_static' / 'performance_optimizations.css'
        
        results['has_critical_css'] = critical_css_path.exists()
        results['has_performance_css'] = performance_css_path.exists()
        
        results['success'] = (results['has_critical_css'] and results['has_performance_css'] and 
                            results['minified_files'] > 0)
        
        if results['success']:
            logger.info(f"CSS optimization validated: {results['css_files']} files, {results['total_size']} bytes")
        else:
            logger.warning("CSS optimization validation failed")
        
        return results
    
    def validate_js_optimization(self) -> Dict[str, any]:
        """Validate JavaScript optimization"""
        logger.info("Validating JavaScript optimization...")
        
        results = {
            'js_files': 0,
            'minified_files': 0,
            'total_size': 0,
            'has_accessibility_js': False,
            'has_search_js': False
        }
        
        # Check JS files
        js_files = list(self.build_dir.rglob('*.js'))
        results['js_files'] = len(js_files)
        
        for js_file in js_files:
            file_size = js_file.stat().st_size
            results['total_size'] += file_size
            
            # Check if file appears minified (basic check)
            with open(js_file, 'r', encoding='utf-8') as f:
                content = f.read()
                # If file has very few newlines relative to size, it's likely minified
                if len(content) > 1000 and content.count('\n') < len(content) / 100:
                    results['minified_files'] += 1
        
        # Check for specific optimization files
        accessibility_js_path = self.build_dir / '_static' / 'accessibility_enhancements.js'
        search_js_path = self.build_dir / '_static' / 'search_enhancements.js'
        
        results['has_accessibility_js'] = accessibility_js_path.exists()
        results['has_search_js'] = search_js_path.exists()
        
        results['success'] = (results['has_accessibility_js'] and results['has_search_js'])
        
        if results['success']:
            logger.info(f"JavaScript optimization validated: {results['js_files']} files, {results['total_size']} bytes")
        else:
            logger.warning("JavaScript optimization validation failed")
        
        return results
    
    def validate_html_optimization(self) -> Dict[str, any]:
        """Validate HTML optimization"""
        logger.info("Validating HTML optimization...")
        
        results = {
            'html_files': 0,
            'files_with_preload': 0,
            'files_with_lazy_loading': 0,
            'files_with_skip_link': 0,
            'files_with_meta_viewport': 0,
            'total_size': 0
        }
        
        # Check HTML files
        html_files = list(self.build_dir.rglob('*.html'))
        results['html_files'] = len(html_files)
        
        for html_file in html_files:
            file_size = html_file.stat().st_size
            results['total_size'] += file_size
            
            try:
                with open(html_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # Check for optimizations
                if 'rel="preload"' in content:
                    results['files_with_preload'] += 1
                
                if 'loading="lazy"' in content:
                    results['files_with_lazy_loading'] += 1
                
                if 'skip-to-content' in content:
                    results['files_with_skip_link'] += 1
                
                if 'name="viewport"' in content:
                    results['files_with_meta_viewport'] += 1
                
            except Exception as e:
                logger.error(f"Error reading HTML file {html_file}: {str(e)}")
        
        results['success'] = (results['files_with_preload'] > 0 and 
                            results['files_with_meta_viewport'] > 0)
        
        if results['success']:
            logger.info(f"HTML optimization validated: {results['html_files']} files")
        else:
            logger.warning("HTML optimization validation failed")
        
        return results
    
    def validate_image_optimization(self) -> Dict[str, any]:
        """Validate image optimization"""
        logger.info("Validating image optimization...")
        
        results = {
            'image_files': 0,
            'webp_files': 0,
            'optimized_markers': 0,
            'total_size': 0,
            'large_images': []
        }
        
        # Check image files
        image_extensions = ['.jpg', '.jpeg', '.png', '.gif', '.webp', '.svg']
        image_files = []
        
        for ext in image_extensions:
            image_files.extend(self.build_dir.rglob(f'*{ext}'))
        
        results['image_files'] = len(image_files)
        
        for image_file in image_files:
            file_size = image_file.stat().st_size
            results['total_size'] += file_size
            
            # Check for WebP files
            if image_file.suffix.lower() == '.webp':
                results['webp_files'] += 1
            
            # Check for optimization markers
            marker_path = image_file.with_suffix(image_file.suffix + '.optimized')
            if marker_path.exists():
                results['optimized_markers'] += 1
            
            # Check for large images (> 500KB)
            if file_size > 500 * 1024:
                results['large_images'].append({
                    'path': str(image_file.relative_to(self.build_dir)),
                    'size': file_size
                })
        
        results['success'] = (results['webp_files'] > 0 or results['optimized_markers'] > 0)
        
        if results['success']:
            logger.info(f"Image optimization validated: {results['image_files']} images, {results['webp_files']} WebP")
        else:
            logger.warning("Image optimization validation failed")
        
        return results
    
    def validate_accessibility_features(self) -> Dict[str, any]:
        """Validate accessibility features"""
        logger.info("Validating accessibility features...")
        
        results = {
            'html_files_checked': 0,
            'files_with_skip_links': 0,
            'files_with_aria_labels': 0,
            'files_with_alt_text': 0,
            'files_with_focus_management': 0,
            'missing_alt_text': [],
            'accessibility_js_present': False
        }
        
        # Check for accessibility JavaScript
        accessibility_js_path = self.build_dir / '_static' / 'accessibility_enhancements.js'
        results['accessibility_js_present'] = accessibility_js_path.exists()
        
        # Check HTML files for accessibility features
        html_files = list(self.build_dir.rglob('*.html'))
        results['html_files_checked'] = len(html_files)
        
        for html_file in html_files[:10]:  # Check first 10 files
            try:
                with open(html_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # Check for accessibility features
                if 'skip-to-content' in content:
                    results['files_with_skip_links'] += 1
                
                if 'aria-label' in content or 'aria-labelledby' in content:
                    results['files_with_aria_labels'] += 1
                
                if 'alt=' in content:
                    results['files_with_alt_text'] += 1
                
                if 'tabindex' in content or 'focus' in content:
                    results['files_with_focus_management'] += 1
                
                # Check for images without alt text
                img_tags = re.findall(r'<img[^>]*>', content)
                for img_tag in img_tags:
                    if 'alt=' not in img_tag:
                        results['missing_alt_text'].append({
                            'file': str(html_file.relative_to(self.build_dir)),
                            'tag': img_tag[:100]
                        })
                
            except Exception as e:
                logger.error(f"Error checking accessibility in {html_file}: {str(e)}")
        
        results['success'] = (results['accessibility_js_present'] and 
                            results['files_with_skip_links'] > 0 and
                            results['files_with_aria_labels'] > 0)
        
        if results['success']:
            logger.info("Accessibility features validated successfully")
        else:
            logger.warning("Accessibility features validation failed")
        
        return results
    
    def validate_performance_features(self) -> Dict[str, any]:
        """Validate performance features"""
        logger.info("Validating performance features...")
        
        results = {
            'service_worker_present': False,
            'manifest_present': False,
            'critical_css_present': False,
            'preload_hints_present': False,
            'compression_enabled': False,
            'performance_css_present': False
        }
        
        # Check for performance files
        sw_path = self.build_dir / 'sw.js'
        manifest_path = self.build_dir / 'manifest.json'
        critical_css_path = self.build_dir / '_static' / 'critical.css'
        performance_css_path = self.build_dir / '_static' / 'performance_optimizations.css'
        
        results['service_worker_present'] = sw_path.exists()
        results['manifest_present'] = manifest_path.exists()
        results['critical_css_present'] = critical_css_path.exists()
        results['performance_css_present'] = performance_css_path.exists()
        
        # Check for preload hints in HTML
        html_files = list(self.build_dir.rglob('*.html'))
        for html_file in html_files[:5]:  # Check first 5 files
            try:
                with open(html_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                if 'rel="preload"' in content:
                    results['preload_hints_present'] = True
                    break
                    
            except Exception:
                pass
        
        # Check for compressed files
        gzip_files = list(self.build_dir.rglob('*.gz'))
        results['compression_enabled'] = len(gzip_files) > 0
        
        results['success'] = (results['service_worker_present'] and 
                            results['manifest_present'] and
                            results['critical_css_present'] and
                            results['performance_css_present'])
        
        if results['success']:
            logger.info("Performance features validated successfully")
        else:
            logger.warning("Performance features validation failed")
        
        return results
    
    def generate_validation_report(self, results: Dict[str, any]):
        """Generate validation report"""
        report_path = self.build_dir / 'validation_report.html'
        
        # Calculate overall success rate
        total_tests = len(results)
        successful_tests = sum(1 for result in results.values() if result.get('success', False))
        success_rate = (successful_tests / total_tests) * 100
        
        html_content = f'''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Optimization Validation Report - CAIS Documentation</title>
    <style>
        body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; margin: 0; padding: 20px; background: #f8f9fa; }}
        .container {{ max-width: 1200px; margin: 0 auto; background: white; padding: 30px; border-radius: 8px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }}
        h1 {{ color: #2c3e50; border-bottom: 3px solid #3498db; padding-bottom: 10px; }}
        .summary {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 20px; margin: 20px 0; }}
        .stat-card {{ padding: 20px; border-radius: 8px; text-align: center; color: white; }}
        .stat-card.success {{ background: linear-gradient(135deg, #4CAF50 0%, #45a049 100%); }}
        .stat-card.warning {{ background: linear-gradient(135deg, #ff9800 0%, #f57c00 100%); }}
        .stat-card.error {{ background: linear-gradient(135deg, #f44336 0%, #da190b 100%); }}
        .stat-number {{ font-size: 2em; font-weight: bold; margin-bottom: 5px; }}
        .stat-label {{ font-size: 0.9em; opacity: 0.9; }}
        .test-section {{ margin: 30px 0; padding: 20px; border: 1px solid #e1e8ed; border-radius: 8px; }}
        .test-section.success {{ border-left: 4px solid #4CAF50; }}
        .test-section.failed {{ border-left: 4px solid #f44336; }}
        .test-title {{ color: #2c3e50; margin-bottom: 15px; display: flex; justify-content: space-between; align-items: center; }}
        .status-badge {{ padding: 4px 12px; border-radius: 20px; font-size: 0.8em; font-weight: 600; }}
        .status-badge.success {{ background: #d4edda; color: #155724; }}
        .status-badge.failed {{ background: #f8d7da; color: #721c24; }}
        .metric {{ display: flex; justify-content: space-between; padding: 8px 0; border-bottom: 1px solid #f0f0f0; }}
        .metric:last-child {{ border-bottom: none; }}
        .metric-value {{ font-weight: 600; }}
        .metric-value.success {{ color: #28a745; }}
        .metric-value.warning {{ color: #ffc107; }}
        .metric-value.error {{ color: #dc3545; }}
        .details {{ margin-top: 15px; padding: 15px; background: #f8f9fa; border-radius: 4px; }}
        .issue-list {{ list-style: none; padding: 0; }}
        .issue-list li {{ padding: 8px; margin: 4px 0; background: #fff3cd; border-left: 3px solid #ffc107; }}
        .issue-list li.error {{ background: #f8d7da; border-left-color: #dc3545; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>Optimization Validation Report</h1>
        
        <div class="summary">
            <div class="stat-card success">
                <div class="stat-number">{success_rate:.1f}%</div>
                <div class="stat-label">Success Rate</div>
            </div>
            <div class="stat-card {'success' if successful_tests == total_tests else 'warning'}">
                <div class="stat-number">{successful_tests}/{total_tests}</div>
                <div class="stat-label">Tests Passed</div>
            </div>
            <div class="stat-card {'success' if results.get('file_structure', {}).get('success') else 'error'}">
                <div class="stat-number">{len(results.get('file_structure', {}).get('present_files', []))}</div>
                <div class="stat-label">Required Files</div>
            </div>
            <div class="stat-card {'success' if results.get('compression', {}).get('gzip_files', 0) > 0 else 'warning'}">
                <div class="stat-number">{results.get('compression', {}).get('gzip_files', 0)}</div>
                <div class="stat-label">Compressed Files</div>
            </div>
        </div>
'''
        
        # Add detailed results for each test
        test_sections = [
            ('file_structure', 'File Structure'),
            ('compression', 'Compression'),
            ('service_worker', 'Service Worker'),
            ('web_manifest', 'Web Manifest'),
            ('css_optimization', 'CSS Optimization'),
            ('js_optimization', 'JavaScript Optimization'),
            ('html_optimization', 'HTML Optimization'),
            ('image_optimization', 'Image Optimization'),
            ('accessibility_features', 'Accessibility Features'),
            ('performance_features', 'Performance Features'),
        ]
        
        for test_key, test_name in test_sections:
            test_result = results.get(test_key, {})
            success = test_result.get('success', False)
            status_class = 'success' if success else 'failed'
            status_text = 'PASSED' if success else 'FAILED'
            
            html_content += f'''
        <div class="test-section {status_class}">
            <div class="test-title">
                <h2>{test_name}</h2>
                <span class="status-badge {status_class}">{status_text}</span>
            </div>
            <div class="metrics">
'''
            
            # Add specific metrics for each test
            for key, value in test_result.items():
                if key not in ['success']:
                    if isinstance(value, (int, float)):
                        html_content += f'''
                <div class="metric">
                    <span>{key.replace('_', ' ').title()}:</span>
                    <span class="metric-value">{value}</span>
                </div>'''
                    elif isinstance(value, bool):
                        status = 'success' if value else 'error'
                        html_content += f'''
                <div class="metric">
                    <span>{key.replace('_', ' ').title()}:</span>
                    <span class="metric-value {status}">{'Yes' if value else 'No'}</span>
                </div>'''
                    elif isinstance(value, list) and len(value) > 0:
                        html_content += f'''
                <div class="metric">
                    <span>{key.replace('_', ' ').title()}:</span>
                    <span class="metric-value">{len(value)} items</span>
                </div>'''
            
            html_content += '''
            </div>
        </div>'''
        
        html_content += '''
    </div>
</body>
</html>'''
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        logger.info(f"Validation report generated: {report_path}")

def main():
    """Main function to run optimization validation"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Validate CAIS documentation optimizations")
    parser.add_argument("--build-dir", default="docs/build", help="Documentation build directory")
    parser.add_argument("--base-url", default="http://localhost:8000", help="Base URL for testing")
    
    args = parser.parse_args()
    
    # Create validator
    validator = OptimizationValidator(args.build_dir, args.base_url)
    
    # Check if build directory exists
    if not validator.build_dir.exists():
        logger.error(f"Build directory not found: {validator.build_dir}")
        logger.info("Run 'make html' and optimization scripts first")
        return 1
    
    logger.info("Starting optimization validation...")
    
    # Run validation
    results = validator.validate_all()
    
    # Print summary
    total_tests = len(results)
    successful_tests = sum(1 for result in results.values() if result.get('success', False))
    success_rate = (successful_tests / total_tests) * 100
    
    print(f"\n{'='*60}")
    print("OPTIMIZATION VALIDATION SUMMARY")
    print(f"{'='*60}")
    print(f"Total tests: {total_tests}")
    print(f"Passed: {successful_tests}")
    print(f"Failed: {total_tests - successful_tests}")
    print(f"Success rate: {success_rate:.1f}%")
    
    # Print failed tests
    failed_tests = [name for name, result in results.items() if not result.get('success', False)]
    if failed_tests:
        print(f"\n❌ Failed tests: {', '.join(failed_tests)}")
        return 1
    else:
        print(f"\n✅ All optimization validations passed!")
        return 0

if __name__ == "__main__":
    sys.exit(main())
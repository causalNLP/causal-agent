#!/usr/bin/env python3
"""
Performance optimization script for CAIS documentation.
Optimizes images, minifies CSS/JS, generates service worker, and implements caching strategies.
"""

import os
import sys
import json
import gzip
import shutil
import hashlib
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import logging
import subprocess
import re

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

try:
    from PIL import Image, ImageOpt
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False
    logger.warning("PIL not available. Install with: pip install Pillow")

try:
    import cssmin
    CSSMIN_AVAILABLE = True
except ImportError:
    CSSMIN_AVAILABLE = False
    logger.warning("cssmin not available. Install with: pip install cssmin")

try:
    import jsmin
    JSMIN_AVAILABLE = True
except ImportError:
    JSMIN_AVAILABLE = False
    logger.warning("jsmin not available. Install with: pip install jsmin")

class PerformanceOptimizer:
    """Main class for optimizing documentation performance"""
    
    def __init__(self, docs_dir: str = "docs", build_dir: str = "docs/build"):
        self.docs_dir = Path(docs_dir)
        self.build_dir = Path(build_dir)
        self.source_dir = self.docs_dir / "source"
        self.static_dir = self.source_dir / "_static"
        
        # Performance optimization settings
        self.image_quality = 85
        self.image_max_width = 1920
        self.enable_webp = True
        self.enable_gzip = True
        self.enable_brotli = False  # Requires brotli package
        
    def optimize_all(self) -> Dict[str, any]:
        """Run all performance optimizations"""
        logger.info("Starting performance optimization...")
        
        results = {
            'images': self.optimize_images(),
            'css': self.optimize_css(),
            'javascript': self.optimize_javascript(),
            'html': self.optimize_html(),
            'compression': self.enable_compression(),
            'service_worker': self.generate_service_worker(),
            'manifest': self.generate_web_manifest(),
            'critical_css': self.extract_critical_css(),
            'preload_hints': self.add_preload_hints(),
        }
        
        # Generate performance report
        self.generate_performance_report(results)
        
        logger.info("Performance optimization completed!")
        return results
    
    def optimize_images(self) -> Dict[str, any]:
        """Optimize images for web delivery"""
        logger.info("Optimizing images...")
        
        if not PIL_AVAILABLE:
            logger.warning("Skipping image optimization - PIL not available")
            return {'skipped': True, 'reason': 'PIL not available'}
        
        results = {
            'processed': 0,
            'saved_bytes': 0,
            'webp_generated': 0,
            'errors': []
        }
        
        # Find all images in the build directory
        image_extensions = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff'}
        image_files = []
        
        for ext in image_extensions:
            image_files.extend(self.build_dir.rglob(f'*{ext}'))
            image_files.extend(self.build_dir.rglob(f'*{ext.upper()}'))
        
        for image_path in image_files:
            try:
                original_size = image_path.stat().st_size
                
                # Skip if already optimized (check for .optimized marker)
                marker_path = image_path.with_suffix(image_path.suffix + '.optimized')
                if marker_path.exists():
                    continue
                
                # Open and optimize image
                with Image.open(image_path) as img:
                    # Convert RGBA to RGB if saving as JPEG
                    if image_path.suffix.lower() in ['.jpg', '.jpeg'] and img.mode in ['RGBA', 'LA']:
                        background = Image.new('RGB', img.size, (255, 255, 255))
                        background.paste(img, mask=img.split()[-1] if img.mode == 'RGBA' else None)
                        img = background
                    
                    # Resize if too large
                    if img.width > self.image_max_width:
                        ratio = self.image_max_width / img.width
                        new_height = int(img.height * ratio)
                        img = img.resize((self.image_max_width, new_height), Image.Resampling.LANCZOS)
                    
                    # Save optimized image
                    save_kwargs = {'optimize': True}
                    if image_path.suffix.lower() in ['.jpg', '.jpeg']:
                        save_kwargs['quality'] = self.image_quality
                        save_kwargs['progressive'] = True
                    elif image_path.suffix.lower() == '.png':
                        save_kwargs['optimize'] = True
                    
                    img.save(image_path, **save_kwargs)
                    
                    # Generate WebP version if enabled
                    if self.enable_webp and image_path.suffix.lower() in ['.jpg', '.jpeg', '.png']:
                        webp_path = image_path.with_suffix('.webp')
                        img.save(webp_path, 'WebP', quality=self.image_quality, optimize=True)
                        results['webp_generated'] += 1
                
                # Create optimization marker
                marker_path.touch()
                
                # Calculate savings
                new_size = image_path.stat().st_size
                saved = original_size - new_size
                results['saved_bytes'] += saved
                results['processed'] += 1
                
                if saved > 0:
                    logger.info(f"Optimized {image_path.name}: saved {saved} bytes ({saved/original_size*100:.1f}%)")
                
            except Exception as e:
                error_msg = f"Error optimizing {image_path}: {str(e)}"
                logger.error(error_msg)
                results['errors'].append(error_msg)
        
        logger.info(f"Image optimization complete: {results['processed']} images processed, {results['saved_bytes']} bytes saved")
        return results
    
    def optimize_css(self) -> Dict[str, any]:
        """Optimize CSS files"""
        logger.info("Optimizing CSS...")
        
        results = {
            'processed': 0,
            'saved_bytes': 0,
            'errors': []
        }
        
        # Find all CSS files
        css_files = list(self.build_dir.rglob('*.css'))
        
        for css_path in css_files:
            try:
                # Skip already minified files
                if '.min.' in css_path.name:
                    continue
                
                # Read original CSS
                with open(css_path, 'r', encoding='utf-8') as f:
                    original_css = f.read()
                
                original_size = len(original_css.encode('utf-8'))
                
                # Minify CSS
                if CSSMIN_AVAILABLE:
                    minified_css = cssmin.cssmin(original_css)
                else:
                    # Basic minification without cssmin
                    minified_css = self.basic_css_minify(original_css)
                
                # Write minified CSS
                with open(css_path, 'w', encoding='utf-8') as f:
                    f.write(minified_css)
                
                new_size = len(minified_css.encode('utf-8'))
                saved = original_size - new_size
                results['saved_bytes'] += saved
                results['processed'] += 1
                
                if saved > 0:
                    logger.info(f"Minified {css_path.name}: saved {saved} bytes ({saved/original_size*100:.1f}%)")
                
            except Exception as e:
                error_msg = f"Error optimizing CSS {css_path}: {str(e)}"
                logger.error(error_msg)
                results['errors'].append(error_msg)
        
        return results
    
    def basic_css_minify(self, css: str) -> str:
        """Basic CSS minification without external dependencies"""
        # Remove comments
        css = re.sub(r'/\*.*?\*/', '', css, flags=re.DOTALL)
        
        # Remove unnecessary whitespace
        css = re.sub(r'\s+', ' ', css)
        css = re.sub(r';\s*}', '}', css)
        css = re.sub(r'{\s*', '{', css)
        css = re.sub(r'}\s*', '}', css)
        css = re.sub(r':\s*', ':', css)
        css = re.sub(r';\s*', ';', css)
        css = re.sub(r',\s*', ',', css)
        
        return css.strip()
    
    def optimize_javascript(self) -> Dict[str, any]:
        """Optimize JavaScript files"""
        logger.info("Optimizing JavaScript...")
        
        results = {
            'processed': 0,
            'saved_bytes': 0,
            'errors': []
        }
        
        # Find all JS files
        js_files = list(self.build_dir.rglob('*.js'))
        
        for js_path in js_files:
            try:
                # Skip already minified files
                if '.min.' in js_path.name:
                    continue
                
                # Read original JavaScript
                with open(js_path, 'r', encoding='utf-8') as f:
                    original_js = f.read()
                
                original_size = len(original_js.encode('utf-8'))
                
                # Minify JavaScript
                if JSMIN_AVAILABLE:
                    minified_js = jsmin.jsmin(original_js)
                else:
                    # Basic minification without jsmin
                    minified_js = self.basic_js_minify(original_js)
                
                # Write minified JavaScript
                with open(js_path, 'w', encoding='utf-8') as f:
                    f.write(minified_js)
                
                new_size = len(minified_js.encode('utf-8'))
                saved = original_size - new_size
                results['saved_bytes'] += saved
                results['processed'] += 1
                
                if saved > 0:
                    logger.info(f"Minified {js_path.name}: saved {saved} bytes ({saved/original_size*100:.1f}%)")
                
            except Exception as e:
                error_msg = f"Error optimizing JS {js_path}: {str(e)}"
                logger.error(error_msg)
                results['errors'].append(error_msg)
        
        return results
    
    def basic_js_minify(self, js: str) -> str:
        """Basic JavaScript minification without external dependencies"""
        # Remove single-line comments (but preserve URLs)
        js = re.sub(r'//(?![^\r\n]*https?:).*', '', js)
        
        # Remove multi-line comments
        js = re.sub(r'/\*.*?\*/', '', js, flags=re.DOTALL)
        
        # Remove unnecessary whitespace
        js = re.sub(r'\s+', ' ', js)
        js = re.sub(r';\s*}', '}', js)
        js = re.sub(r'{\s*', '{', js)
        js = re.sub(r'}\s*', '}', js)
        js = re.sub(r';\s*', ';', js)
        
        return js.strip()
    
    def optimize_html(self) -> Dict[str, any]:
        """Optimize HTML files"""
        logger.info("Optimizing HTML...")
        
        results = {
            'processed': 0,
            'saved_bytes': 0,
            'errors': []
        }
        
        # Find all HTML files
        html_files = list(self.build_dir.rglob('*.html'))
        
        for html_path in html_files:
            try:
                # Read original HTML
                with open(html_path, 'r', encoding='utf-8') as f:
                    original_html = f.read()
                
                original_size = len(original_html.encode('utf-8'))
                
                # Optimize HTML
                optimized_html = self.optimize_html_content(original_html)
                
                # Write optimized HTML
                with open(html_path, 'w', encoding='utf-8') as f:
                    f.write(optimized_html)
                
                new_size = len(optimized_html.encode('utf-8'))
                saved = original_size - new_size
                results['saved_bytes'] += saved
                results['processed'] += 1
                
                if saved > 0:
                    logger.info(f"Optimized {html_path.name}: saved {saved} bytes ({saved/original_size*100:.1f}%)")
                
            except Exception as e:
                error_msg = f"Error optimizing HTML {html_path}: {str(e)}"
                logger.error(error_msg)
                results['errors'].append(error_msg)
        
        return results
    
    def optimize_html_content(self, html: str) -> str:
        """Optimize HTML content"""
        # Add performance optimizations to HTML
        optimizations = []
        
        # Add preconnect for external resources
        if 'fonts.googleapis.com' in html:
            optimizations.append('<link rel="preconnect" href="https://fonts.googleapis.com">')
            optimizations.append('<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>')
        
        # Add DNS prefetch for external domains
        external_domains = re.findall(r'https?://([^/]+)', html)
        for domain in set(external_domains):
            if domain not in ['localhost', '127.0.0.1']:
                optimizations.append(f'<link rel="dns-prefetch" href="//{domain}">')
        
        # Insert optimizations in head
        if optimizations and '<head>' in html:
            head_end = html.find('</head>')
            if head_end != -1:
                optimization_html = '\n    ' + '\n    '.join(optimizations) + '\n    '
                html = html[:head_end] + optimization_html + html[head_end:]
        
        # Add loading="lazy" to images
        html = re.sub(r'<img(?![^>]*loading=)', '<img loading="lazy"', html)
        
        # Add decoding="async" to images
        html = re.sub(r'<img(?![^>]*decoding=)', '<img decoding="async"', html)
        
        # Minify HTML (basic)
        html = re.sub(r'>\s+<', '><', html)
        html = re.sub(r'\s+', ' ', html)
        
        return html
    
    def enable_compression(self) -> Dict[str, any]:
        """Enable gzip compression for static files"""
        logger.info("Enabling compression...")
        
        results = {
            'gzip_files': 0,
            'brotli_files': 0,
            'saved_bytes': 0,
            'errors': []
        }
        
        # File types to compress
        compress_extensions = {'.html', '.css', '.js', '.json', '.xml', '.txt', '.svg'}
        
        # Find files to compress
        files_to_compress = []
        for ext in compress_extensions:
            files_to_compress.extend(self.build_dir.rglob(f'*{ext}'))
        
        for file_path in files_to_compress:
            try:
                # Skip already compressed files
                if file_path.suffix in ['.gz', '.br']:
                    continue
                
                original_size = file_path.stat().st_size
                
                # Skip very small files
                if original_size < 1024:  # Less than 1KB
                    continue
                
                # Gzip compression
                if self.enable_gzip:
                    gzip_path = file_path.with_suffix(file_path.suffix + '.gz')
                    with open(file_path, 'rb') as f_in:
                        with gzip.open(gzip_path, 'wb') as f_out:
                            shutil.copyfileobj(f_in, f_out)
                    
                    gzip_size = gzip_path.stat().st_size
                    results['saved_bytes'] += original_size - gzip_size
                    results['gzip_files'] += 1
                
                # Brotli compression (if available)
                if self.enable_brotli:
                    try:
                        import brotli
                        brotli_path = file_path.with_suffix(file_path.suffix + '.br')
                        with open(file_path, 'rb') as f:
                            compressed = brotli.compress(f.read())
                        with open(brotli_path, 'wb') as f:
                            f.write(compressed)
                        results['brotli_files'] += 1
                    except ImportError:
                        pass
                
            except Exception as e:
                error_msg = f"Error compressing {file_path}: {str(e)}"
                logger.error(error_msg)
                results['errors'].append(error_msg)
        
        return results
    
    def generate_service_worker(self) -> Dict[str, any]:
        """Generate service worker for offline caching"""
        logger.info("Generating service worker...")
        
        # Find all static assets to cache
        cache_files = []
        cache_extensions = {'.html', '.css', '.js', '.png', '.jpg', '.jpeg', '.gif', '.svg', '.webp', '.woff', '.woff2'}
        
        for ext in cache_extensions:
            for file_path in self.build_dir.rglob(f'*{ext}'):
                # Get relative path from build directory
                rel_path = file_path.relative_to(self.build_dir)
                cache_files.append('/' + str(rel_path).replace('\\', '/'))
        
        # Generate service worker content
        sw_content = f'''
// CAIS Documentation Service Worker
// Auto-generated by performance optimizer

const CACHE_NAME = 'cais-docs-v{self.get_cache_version()}';
const STATIC_CACHE_URLS = {json.dumps(cache_files[:100], indent=2)};  // Limit to first 100 files

// Install event - cache static assets
self.addEventListener('install', (event) => {{
    event.waitUntil(
        caches.open(CACHE_NAME)
            .then((cache) => {{
                console.log('Caching static assets');
                return cache.addAll(STATIC_CACHE_URLS);
            }})
            .then(() => {{
                self.skipWaiting();
            }})
    );
}});

// Activate event - clean up old caches
self.addEventListener('activate', (event) => {{
    event.waitUntil(
        caches.keys().then((cacheNames) => {{
            return Promise.all(
                cacheNames.map((cacheName) => {{
                    if (cacheName !== CACHE_NAME && cacheName.startsWith('cais-docs-')) {{
                        console.log('Deleting old cache:', cacheName);
                        return caches.delete(cacheName);
                    }}
                }})
            );
        }}).then(() => {{
            self.clients.claim();
        }})
    );
}});

// Fetch event - serve from cache with network fallback
self.addEventListener('fetch', (event) => {{
    // Only handle GET requests
    if (event.request.method !== 'GET') {{
        return;
    }}
    
    // Skip cross-origin requests
    if (!event.request.url.startsWith(self.location.origin)) {{
        return;
    }}
    
    event.respondWith(
        caches.match(event.request)
            .then((response) => {{
                // Return cached version if available
                if (response) {{
                    return response;
                }}
                
                // Otherwise fetch from network
                return fetch(event.request)
                    .then((response) => {{
                        // Don't cache non-successful responses
                        if (!response || response.status !== 200 || response.type !== 'basic') {{
                            return response;
                        }}
                        
                        // Clone the response
                        const responseToCache = response.clone();
                        
                        // Cache the response
                        caches.open(CACHE_NAME)
                            .then((cache) => {{
                                cache.put(event.request, responseToCache);
                            }});
                        
                        return response;
                    }})
                    .catch(() => {{
                        // Return offline page for HTML requests
                        if (event.request.headers.get('accept').includes('text/html')) {{
                            return caches.match('/offline.html');
                        }}
                    }});
            }})
    );
}});

// Background sync for analytics (if supported)
self.addEventListener('sync', (event) => {{
    if (event.tag === 'analytics-sync') {{
        event.waitUntil(syncAnalytics());
    }}
}});

function syncAnalytics() {{
    // Sync any pending analytics data
    return Promise.resolve();
}}
'''
        
        # Write service worker
        sw_path = self.build_dir / 'sw.js'
        with open(sw_path, 'w', encoding='utf-8') as f:
            f.write(sw_content)
        
        # Generate offline page
        self.generate_offline_page()
        
        return {
            'service_worker_path': str(sw_path),
            'cached_files': len(cache_files),
            'cache_version': self.get_cache_version()
        }
    
    def generate_offline_page(self):
        """Generate offline page for service worker"""
        offline_content = '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Offline - CAIS Documentation</title>
    <style>
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            margin: 0;
            padding: 20px;
            background: #f8f9fa;
            display: flex;
            align-items: center;
            justify-content: center;
            min-height: 100vh;
        }
        .offline-container {
            text-align: center;
            max-width: 500px;
            background: white;
            padding: 40px;
            border-radius: 8px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }
        .offline-icon {
            font-size: 4em;
            margin-bottom: 20px;
        }
        h1 {
            color: #2c3e50;
            margin-bottom: 20px;
        }
        p {
            color: #6c757d;
            line-height: 1.6;
            margin-bottom: 20px;
        }
        .retry-btn {
            background: #007bff;
            color: white;
            border: none;
            padding: 12px 24px;
            border-radius: 4px;
            cursor: pointer;
            font-size: 16px;
        }
        .retry-btn:hover {
            background: #0056b3;
        }
    </style>
</head>
<body>
    <div class="offline-container">
        <div class="offline-icon">📡</div>
        <h1>You're Offline</h1>
        <p>It looks like you're not connected to the internet. Some content may not be available, but you can still browse cached pages.</p>
        <button class="retry-btn" onclick="window.location.reload()">Try Again</button>
    </div>
</body>
</html>
'''
        
        offline_path = self.build_dir / 'offline.html'
        with open(offline_path, 'w', encoding='utf-8') as f:
            f.write(offline_content)
    
    def generate_web_manifest(self) -> Dict[str, any]:
        """Generate web app manifest"""
        logger.info("Generating web manifest...")
        
        manifest = {
            "name": "CAIS Documentation",
            "short_name": "CAIS Docs",
            "description": "Comprehensive documentation for Causal AI Scientist (CAIS)",
            "start_url": "/",
            "display": "standalone",
            "background_color": "#ffffff",
            "theme_color": "#007bff",
            "icons": [
                {
                    "src": "/favicon.ico",
                    "sizes": "16x16 32x32",
                    "type": "image/x-icon"
                }
            ],
            "categories": ["education", "reference", "documentation"],
            "lang": "en",
            "dir": "ltr",
            "orientation": "any"
        }
        
        manifest_path = self.build_dir / 'manifest.json'
        with open(manifest_path, 'w', encoding='utf-8') as f:
            json.dump(manifest, f, indent=2)
        
        return {
            'manifest_path': str(manifest_path),
            'manifest': manifest
        }
    
    def extract_critical_css(self) -> Dict[str, any]:
        """Extract critical CSS for above-the-fold content"""
        logger.info("Extracting critical CSS...")
        
        # This is a simplified version - in production, you'd use tools like critical or penthouse
        critical_css = '''
        /* Critical CSS for above-the-fold content */
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            margin: 0;
            padding: 0;
            line-height: 1.6;
            color: #333;
        }
        
        .wy-nav-top {
            background: #2980b9;
            color: white;
            padding: 10px 15px;
        }
        
        .wy-nav-side {
            width: 300px;
            background: #343131;
            position: fixed;
            height: 100%;
            overflow-y: auto;
        }
        
        .wy-nav-content-wrap {
            margin-left: 300px;
        }
        
        .wy-nav-content {
            padding: 1.618em 3.236em;
            max-width: none;
        }
        
        h1, h2, h3, h4, h5, h6 {
            margin-top: 0;
            font-weight: 700;
            color: #2c3e50;
        }
        
        .skip-to-content {
            position: absolute;
            top: -40px;
            left: 6px;
            background: #007bff;
            color: white;
            padding: 8px;
            text-decoration: none;
            border-radius: 0 0 4px 4px;
            z-index: 9999;
        }
        
        .skip-to-content:focus {
            top: 0;
        }
        '''
        
        critical_css_path = self.build_dir / '_static' / 'critical.css'
        critical_css_path.parent.mkdir(exist_ok=True)
        
        with open(critical_css_path, 'w', encoding='utf-8') as f:
            f.write(critical_css)
        
        return {
            'critical_css_path': str(critical_css_path),
            'size': len(critical_css)
        }
    
    def add_preload_hints(self) -> Dict[str, any]:
        """Add resource preload hints to HTML files"""
        logger.info("Adding preload hints...")
        
        results = {
            'processed': 0,
            'hints_added': 0
        }
        
        # Find all HTML files
        html_files = list(self.build_dir.rglob('*.html'))
        
        for html_path in html_files:
            try:
                with open(html_path, 'r', encoding='utf-8') as f:
                    html = f.read()
                
                hints_added = 0
                
                # Add preload for critical CSS
                if '<link rel="stylesheet"' in html and 'rel="preload"' not in html:
                    css_links = re.findall(r'<link rel="stylesheet"[^>]*href="([^"]*)"', html)
                    for css_href in css_links[:2]:  # Preload first 2 CSS files
                        preload_hint = f'<link rel="preload" href="{css_href}" as="style" onload="this.onload=null;this.rel=\'stylesheet\'">'
                        if preload_hint not in html:
                            html = html.replace('<head>', f'<head>\n    {preload_hint}')
                            hints_added += 1
                
                # Add preload for critical JavaScript
                js_links = re.findall(r'<script[^>]*src="([^"]*)"', html)
                for js_href in js_links[:1]:  # Preload first JS file
                    if 'jquery' in js_href.lower() or 'main' in js_href.lower():
                        preload_hint = f'<link rel="preload" href="{js_href}" as="script">'
                        if preload_hint not in html:
                            html = html.replace('<head>', f'<head>\n    {preload_hint}')
                            hints_added += 1
                
                # Add preload for web fonts
                if 'fonts.googleapis.com' in html:
                    font_preload = '<link rel="preload" href="https://fonts.googleapis.com/css?family=Lato:400,700,400italic,700italic&subset=latin,latin-ext" as="style">'
                    if font_preload not in html:
                        html = html.replace('<head>', f'<head>\n    {font_preload}')
                        hints_added += 1
                
                if hints_added > 0:
                    with open(html_path, 'w', encoding='utf-8') as f:
                        f.write(html)
                    
                    results['processed'] += 1
                    results['hints_added'] += hints_added
                
            except Exception as e:
                logger.error(f"Error adding preload hints to {html_path}: {str(e)}")
        
        return results
    
    def get_cache_version(self) -> str:
        """Generate cache version based on content hash"""
        # Simple version based on current timestamp
        import time
        return str(int(time.time()))
    
    def generate_performance_report(self, results: Dict[str, any]):
        """Generate performance optimization report"""
        report_path = self.build_dir / 'performance_report.html'
        
        total_saved = sum([
            results.get('images', {}).get('saved_bytes', 0),
            results.get('css', {}).get('saved_bytes', 0),
            results.get('javascript', {}).get('saved_bytes', 0),
            results.get('html', {}).get('saved_bytes', 0),
            results.get('compression', {}).get('saved_bytes', 0),
        ])
        
        html_content = f'''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Performance Optimization Report - CAIS Documentation</title>
    <style>
        body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; margin: 0; padding: 20px; background: #f8f9fa; }}
        .container {{ max-width: 1000px; margin: 0 auto; background: white; padding: 30px; border-radius: 8px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }}
        h1 {{ color: #2c3e50; border-bottom: 3px solid #3498db; padding-bottom: 10px; }}
        .summary {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 20px; margin: 20px 0; }}
        .stat-card {{ background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 20px; border-radius: 8px; text-align: center; }}
        .stat-number {{ font-size: 2em; font-weight: bold; margin-bottom: 5px; }}
        .stat-label {{ font-size: 0.9em; opacity: 0.9; }}
        .optimization-section {{ margin: 30px 0; padding: 20px; border: 1px solid #e1e8ed; border-radius: 8px; }}
        .optimization-title {{ color: #2c3e50; margin-bottom: 15px; }}
        .metric {{ display: flex; justify-content: space-between; padding: 8px 0; border-bottom: 1px solid #f0f0f0; }}
        .metric:last-child {{ border-bottom: none; }}
        .success {{ color: #28a745; }}
        .warning {{ color: #ffc107; }}
        .error {{ color: #dc3545; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>Performance Optimization Report</h1>
        
        <div class="summary">
            <div class="stat-card">
                <div class="stat-number">{total_saved // 1024}KB</div>
                <div class="stat-label">Total Saved</div>
            </div>
            <div class="stat-card">
                <div class="stat-number">{results.get('images', {}).get('processed', 0)}</div>
                <div class="stat-label">Images Optimized</div>
            </div>
            <div class="stat-card">
                <div class="stat-number">{results.get('compression', {}).get('gzip_files', 0)}</div>
                <div class="stat-label">Files Compressed</div>
            </div>
            <div class="stat-card">
                <div class="stat-number">{results.get('service_worker', {}).get('cached_files', 0)}</div>
                <div class="stat-label">Files Cached</div>
            </div>
        </div>
        
        <div class="optimization-section">
            <h2 class="optimization-title">Image Optimization</h2>
            <div class="metric">
                <span>Images Processed:</span>
                <span class="success">{results.get('images', {}).get('processed', 0)}</span>
            </div>
            <div class="metric">
                <span>Bytes Saved:</span>
                <span class="success">{results.get('images', {}).get('saved_bytes', 0)} bytes</span>
            </div>
            <div class="metric">
                <span>WebP Images Generated:</span>
                <span class="success">{results.get('images', {}).get('webp_generated', 0)}</span>
            </div>
        </div>
        
        <div class="optimization-section">
            <h2 class="optimization-title">CSS Optimization</h2>
            <div class="metric">
                <span>CSS Files Processed:</span>
                <span class="success">{results.get('css', {}).get('processed', 0)}</span>
            </div>
            <div class="metric">
                <span>Bytes Saved:</span>
                <span class="success">{results.get('css', {}).get('saved_bytes', 0)} bytes</span>
            </div>
        </div>
        
        <div class="optimization-section">
            <h2 class="optimization-title">JavaScript Optimization</h2>
            <div class="metric">
                <span>JS Files Processed:</span>
                <span class="success">{results.get('javascript', {}).get('processed', 0)}</span>
            </div>
            <div class="metric">
                <span>Bytes Saved:</span>
                <span class="success">{results.get('javascript', {}).get('saved_bytes', 0)} bytes</span>
            </div>
        </div>
        
        <div class="optimization-section">
            <h2 class="optimization-title">Compression</h2>
            <div class="metric">
                <span>Gzip Files Created:</span>
                <span class="success">{results.get('compression', {}).get('gzip_files', 0)}</span>
            </div>
            <div class="metric">
                <span>Compression Savings:</span>
                <span class="success">{results.get('compression', {}).get('saved_bytes', 0)} bytes</span>
            </div>
        </div>
        
        <div class="optimization-section">
            <h2 class="optimization-title">Caching & PWA</h2>
            <div class="metric">
                <span>Service Worker:</span>
                <span class="success">Generated</span>
            </div>
            <div class="metric">
                <span>Web Manifest:</span>
                <span class="success">Generated</span>
            </div>
            <div class="metric">
                <span>Cached Files:</span>
                <span class="success">{results.get('service_worker', {}).get('cached_files', 0)}</span>
            </div>
        </div>
        
        <div class="optimization-section">
            <h2 class="optimization-title">Preload Hints</h2>
            <div class="metric">
                <span>HTML Files Processed:</span>
                <span class="success">{results.get('preload_hints', {}).get('processed', 0)}</span>
            </div>
            <div class="metric">
                <span>Hints Added:</span>
                <span class="success">{results.get('preload_hints', {}).get('hints_added', 0)}</span>
            </div>
        </div>
    </div>
</body>
</html>
'''
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        logger.info(f"Performance report generated: {report_path}")

def main():
    """Main function to run performance optimizations"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Optimize CAIS documentation performance")
    parser.add_argument("--docs-dir", default="docs", help="Documentation source directory")
    parser.add_argument("--build-dir", default="docs/build", help="Documentation build directory")
    parser.add_argument("--skip-images", action="store_true", help="Skip image optimization")
    parser.add_argument("--skip-minify", action="store_true", help="Skip CSS/JS minification")
    parser.add_argument("--skip-compression", action="store_true", help="Skip file compression")
    
    args = parser.parse_args()
    
    # Create optimizer
    optimizer = PerformanceOptimizer(args.docs_dir, args.build_dir)
    
    # Check if build directory exists
    if not optimizer.build_dir.exists():
        logger.error(f"Build directory not found: {optimizer.build_dir}")
        logger.info("Run 'make html' in the docs directory first")
        return 1
    
    logger.info("Starting performance optimization...")
    
    # Run optimizations
    results = {}
    
    if not args.skip_images:
        results['images'] = optimizer.optimize_images()
    
    if not args.skip_minify:
        results['css'] = optimizer.optimize_css()
        results['javascript'] = optimizer.optimize_javascript()
    
    results['html'] = optimizer.optimize_html()
    
    if not args.skip_compression:
        results['compression'] = optimizer.enable_compression()
    
    results['service_worker'] = optimizer.generate_service_worker()
    results['manifest'] = optimizer.generate_web_manifest()
    results['critical_css'] = optimizer.extract_critical_css()
    results['preload_hints'] = optimizer.add_preload_hints()
    
    # Generate report
    optimizer.generate_performance_report(results)
    
    # Print summary
    total_saved = sum([
        results.get('images', {}).get('saved_bytes', 0),
        results.get('css', {}).get('saved_bytes', 0),
        results.get('javascript', {}).get('saved_bytes', 0),
        results.get('html', {}).get('saved_bytes', 0),
        results.get('compression', {}).get('saved_bytes', 0),
    ])
    
    print(f"\n{'='*60}")
    print("PERFORMANCE OPTIMIZATION SUMMARY")
    print(f"{'='*60}")
    print(f"Total bytes saved: {total_saved:,} ({total_saved/1024:.1f} KB)")
    print(f"Images optimized: {results.get('images', {}).get('processed', 0)}")
    print(f"CSS files minified: {results.get('css', {}).get('processed', 0)}")
    print(f"JS files minified: {results.get('javascript', {}).get('processed', 0)}")
    print(f"Files compressed: {results.get('compression', {}).get('gzip_files', 0)}")
    print(f"Service worker: Generated")
    print(f"Web manifest: Generated")
    print(f"\n✅ Performance optimization completed!")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
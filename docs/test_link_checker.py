#!/usr/bin/env python3
"""
Link checker for documentation.
Validates internal and external links in HTML documentation.
"""

import os
import sys
import re
import requests
import time
from pathlib import Path
from urllib.parse import urljoin, urlparse
from typing import List, Dict, Set, Tuple, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
import json

class LinkChecker:
    """Check internal and external links in documentation."""
    
    def __init__(self, html_dir: str, base_url: str = ""):
        self.html_dir = Path(html_dir)
        self.base_url = base_url
        self.internal_links = set()
        self.external_links = set()
        self.broken_links = []
        self.checked_external = {}  # Cache for external link checks
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Documentation Link Checker 1.0'
        })
        
    def run_all_checks(self) -> bool:
        """Run all link checking tests."""
        print("Starting link validation...")
        
        if not self.html_dir.exists():
            print(f"ERROR: HTML directory not found: {self.html_dir}")
            return False
            
        # Extract all links
        self._extract_links()
        
        # Check internal links
        internal_ok = self._check_internal_links()
        
        # Check external links (with rate limiting)
        external_ok = self._check_external_links()
        
        # Generate report
        self._generate_report()
        
        return internal_ok and external_ok
    
    def _extract_links(self):
        """Extract all links from HTML files."""
        print("Extracting links from HTML files...")
        
        html_files = list(self.html_dir.rglob("*.html"))
        print(f"Found {len(html_files)} HTML files")
        
        for html_file in html_files:
            try:
                with open(html_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                    
                # Extract href links
                href_links = re.findall(r'href="([^"]+)"', content)
                
                # Extract src links (images, scripts)
                src_links = re.findall(r'src="([^"]+)"', content)
                
                all_links = href_links + src_links
                
                for link in all_links:
                    # Skip empty links, anchors, and javascript
                    if not link or link.startswith('#') or link.startswith('javascript:'):
                        continue
                        
                    # Categorize links
                    if self._is_external_link(link):
                        self.external_links.add(link)
                    else:
                        # Store with source file for better error reporting
                        self.internal_links.add((str(html_file), link))
                        
            except Exception as e:
                print(f"WARNING: Could not process {html_file}: {e}")
                
        print(f"Extracted {len(self.internal_links)} internal links")
        print(f"Extracted {len(self.external_links)} external links")
    
    def _is_external_link(self, link: str) -> bool:
        """Check if a link is external."""
        return link.startswith(('http://', 'https://', 'ftp://'))
    
    def _check_internal_links(self) -> bool:
        """Check internal links for validity."""
        print("\nChecking internal links...")
        
        broken_internal = []
        
        for source_file, link in self.internal_links:
            # Resolve relative paths
            source_path = Path(source_file)
            
            # Handle different link types
            if link.startswith('/'):
                # Absolute path from root
                target_path = self.html_dir / link.lstrip('/')
            else:
                # Relative path from current file
                target_path = source_path.parent / link
                
            # Remove anchor fragments for file checking
            file_part = str(target_path).split('#')[0]
            target_file = Path(file_part)
            
            # Normalize path
            try:
                target_file = target_file.resolve()
            except:
                pass
                
            if not target_file.exists():
                broken_internal.append({
                    'source': source_path.name,
                    'link': link,
                    'target': str(target_file),
                    'type': 'internal'
                })
        
        if broken_internal:
            print(f"❌ Found {len(broken_internal)} broken internal links")
            for link in broken_internal[:10]:  # Show first 10
                print(f"  {link['source']} -> {link['link']}")
            self.broken_links.extend(broken_internal)
            return False
        else:
            print("✅ All internal links are valid")
            return True
    
    def _check_external_links(self, max_workers: int = 10) -> bool:
        """Check external links with concurrent requests."""
        print(f"\nChecking {len(self.external_links)} external links...")
        
        if not self.external_links:
            print("✅ No external links to check")
            return True
            
        broken_external = []
        
        # Filter out common domains that might block automated requests
        skip_domains = {
            'localhost',
            '127.0.0.1',
            'example.com',
            'test.com'
        }
        
        links_to_check = []
        for link in self.external_links:
            domain = urlparse(link).netloc
            if domain not in skip_domains:
                links_to_check.append(link)
            else:
                print(f"Skipping {link} (test domain)")
        
        print(f"Checking {len(links_to_check)} external links...")
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all link checks
            future_to_link = {
                executor.submit(self._check_single_external_link, link): link 
                for link in links_to_check
            }
            
            completed = 0
            for future in as_completed(future_to_link):
                link = future_to_link[future]
                completed += 1
                
                if completed % 10 == 0:
                    print(f"Progress: {completed}/{len(links_to_check)}")
                
                try:
                    result = future.result()
                    if not result['valid']:
                        broken_external.append(result)
                        
                except Exception as e:
                    broken_external.append({
                        'link': link,
                        'error': str(e),
                        'type': 'external',
                        'valid': False
                    })
        
        if broken_external:
            print(f"❌ Found {len(broken_external)} broken external links")
            for link in broken_external[:5]:  # Show first 5
                print(f"  {link['link']}: {link.get('error', 'Unknown error')}")
            self.broken_links.extend(broken_external)
            # Don't fail for external links as they might be temporarily down
            print("⚠️  External link failures don't fail the build")
            
        print(f"✅ External link check completed ({len(broken_external)} issues found)")
        return True  # Don't fail build for external links
    
    def _check_single_external_link(self, link: str) -> Dict:
        """Check a single external link."""
        if link in self.checked_external:
            return self.checked_external[link]
            
        try:
            # Add delay to avoid rate limiting
            time.sleep(0.1)
            
            response = self.session.head(
                link, 
                timeout=10, 
                allow_redirects=True
            )
            
            # If HEAD fails, try GET
            if response.status_code >= 400:
                response = self.session.get(
                    link, 
                    timeout=10, 
                    allow_redirects=True
                )
            
            result = {
                'link': link,
                'status_code': response.status_code,
                'valid': response.status_code < 400,
                'type': 'external'
            }
            
            if not result['valid']:
                result['error'] = f"HTTP {response.status_code}"
                
        except requests.exceptions.Timeout:
            result = {
                'link': link,
                'error': 'Timeout',
                'valid': False,
                'type': 'external'
            }
        except requests.exceptions.ConnectionError:
            result = {
                'link': link,
                'error': 'Connection Error',
                'valid': False,
                'type': 'external'
            }
        except Exception as e:
            result = {
                'link': link,
                'error': str(e),
                'valid': False,
                'type': 'external'
            }
        
        self.checked_external[link] = result
        return result
    
    def _generate_report(self):
        """Generate link checking report."""
        report_file = self.html_dir.parent / "link_check_report.json"
        
        report = {
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'summary': {
                'total_internal_links': len(self.internal_links),
                'total_external_links': len(self.external_links),
                'broken_links': len(self.broken_links),
                'broken_internal': len([l for l in self.broken_links if l.get('type') == 'internal']),
                'broken_external': len([l for l in self.broken_links if l.get('type') == 'external'])
            },
            'broken_links': self.broken_links
        }
        
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)
            
        print(f"\n📊 Link check report saved to: {report_file}")
        
        # Print summary
        print("\n" + "="*50)
        print("LINK CHECK SUMMARY")
        print("="*50)
        print(f"Internal links: {report['summary']['total_internal_links']}")
        print(f"External links: {report['summary']['total_external_links']}")
        print(f"Broken internal: {report['summary']['broken_internal']}")
        print(f"Broken external: {report['summary']['broken_external']}")
        
        if report['summary']['broken_links'] == 0:
            print("🎉 No broken links found!")
        else:
            print(f"⚠️  {report['summary']['broken_links']} broken links found")

def main():
    """Main function to run link checker."""
    if len(sys.argv) < 2:
        print("Usage: python test_link_checker.py <html_directory> [base_url]")
        sys.exit(1)
        
    html_dir = sys.argv[1]
    base_url = sys.argv[2] if len(sys.argv) > 2 else ""
    
    checker = LinkChecker(html_dir, base_url)
    success = checker.run_all_checks()
    
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()
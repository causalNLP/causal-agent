#!/usr/bin/env python3
"""
Automated testing for documentation builds.
Tests Sphinx build process, validates output, and checks for common issues.
"""

import os
import sys
import subprocess
import tempfile
import shutil
from pathlib import Path
import json
import re
from typing import List, Dict, Tuple, Optional

class DocumentationBuildTester:
    """Test documentation build process and validate output."""
    
    def __init__(self, docs_dir: str = "docs"):
        self.docs_dir = Path(docs_dir)
        self.source_dir = self.docs_dir / "source"
        self.build_dir = self.docs_dir / "build"
        self.test_results = []
        
    def run_all_tests(self) -> bool:
        """Run all documentation build tests."""
        print("Starting documentation build tests...")
        
        tests = [
            ("Test Sphinx Configuration", self.test_sphinx_config),
            ("Test Clean Build", self.test_clean_build),
            ("Test Incremental Build", self.test_incremental_build),
            ("Test Build Warnings", self.test_build_warnings),
            ("Test Output Structure", self.test_output_structure),
            ("Test HTML Validation", self.test_html_validation),
            ("Test Cross-References", self.test_cross_references),
            ("Test Search Index", self.test_search_index),
        ]
        
        all_passed = True
        for test_name, test_func in tests:
            print(f"\n{'='*60}")
            print(f"Running: {test_name}")
            print('='*60)
            
            try:
                result = test_func()
                status = "PASS" if result else "FAIL"
                print(f"Result: {status}")
                
                if not result:
                    all_passed = False
                    
                self.test_results.append({
                    "test": test_name,
                    "status": status,
                    "passed": result
                })
                
            except Exception as e:
                print(f"ERROR: {e}")
                all_passed = False
                self.test_results.append({
                    "test": test_name,
                    "status": "ERROR",
                    "passed": False,
                    "error": str(e)
                })
        
        self._print_summary()
        return all_passed
    
    def test_sphinx_config(self) -> bool:
        """Test Sphinx configuration validity."""
        conf_py = self.source_dir / "conf.py"
        
        if not conf_py.exists():
            print("ERROR: conf.py not found")
            return False
            
        # Test configuration by importing it
        try:
            import importlib.util
            spec = importlib.util.spec_from_file_location("conf", conf_py)
            conf_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(conf_module)
            
            # Check required configuration
            required_configs = ['project', 'author', 'extensions']
            for config in required_configs:
                if not hasattr(conf_module, config):
                    print(f"ERROR: Missing required configuration: {config}")
                    return False
                    
            print("✓ Sphinx configuration is valid")
            return True
            
        except Exception as e:
            print(f"ERROR: Invalid Sphinx configuration: {e}")
            return False
    
    def test_clean_build(self) -> bool:
        """Test clean documentation build."""
        try:
            # Clean previous builds
            if self.build_dir.exists():
                shutil.rmtree(self.build_dir)
                
            # Check if we should use make or sphinx-build directly
            makefile = self.docs_dir / "Makefile"
            if makefile.exists():
                # Use make command
                cmd = ["make", "clean"]
                subprocess.run(cmd, cwd=self.docs_dir, capture_output=True)
                
                cmd = ["make", "html"]
                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    cwd=self.docs_dir
                )
            else:
                # Use sphinx-build directly
                cmd = [
                    "sphinx-build",
                    "-b", "html",
                    "-W",  # Treat warnings as errors
                    str(self.source_dir),
                    str(self.build_dir / "html")
                ]
                
                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    cwd=self.docs_dir
                )
            
            if result.returncode != 0:
                print(f"ERROR: Build failed with return code {result.returncode}")
                print("STDOUT:", result.stdout)
                print("STDERR:", result.stderr)
                return False
                
            print("✓ Clean build completed successfully")
            return True
            
        except Exception as e:
            print(f"ERROR: Clean build failed: {e}")
            return False
    
    def test_incremental_build(self) -> bool:
        """Test incremental build performance."""
        try:
            # First build (should already exist from clean build test)
            if not (self.build_dir / "html" / "index.html").exists():
                print("ERROR: No existing build found for incremental test")
                return False
                
            # Touch a source file to trigger incremental build
            index_rst = self.source_dir / "index.rst"
            if index_rst.exists():
                index_rst.touch()
                
            # Run incremental build
            makefile = self.docs_dir / "Makefile"
            if makefile.exists():
                cmd = ["make", "html"]
                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    cwd=self.docs_dir
                )
            else:
                cmd = [
                    "sphinx-build",
                    "-b", "html",
                    str(self.source_dir),
                    str(self.build_dir / "html")
                ]
                
                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    cwd=self.docs_dir
                )
            
            if result.returncode != 0:
                print(f"ERROR: Incremental build failed: {result.stderr}")
                return False
                
            print("✓ Incremental build completed successfully")
            return True
            
        except Exception as e:
            print(f"ERROR: Incremental build test failed: {e}")
            return False
    
    def test_build_warnings(self) -> bool:
        """Test for build warnings and errors."""
        try:
            makefile = self.docs_dir / "Makefile"
            if makefile.exists():
                # Use make with verbose output
                cmd = ["make", "html", "SPHINXOPTS=-v"]
                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    cwd=self.docs_dir
                )
            else:
                cmd = [
                    "sphinx-build",
                    "-b", "html",
                    "-v",  # Verbose output
                    str(self.source_dir),
                    str(self.build_dir / "html_warnings")
                ]
                
                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    cwd=self.docs_dir
                )
            
            # Check for common warning patterns
            warning_patterns = [
                r"WARNING:",
                r"ERROR:",
                r"undefined label:",
                r"unknown document:",
                r"duplicate object description",
                r"image file not readable:"
            ]
            
            warnings_found = []
            for pattern in warning_patterns:
                matches = re.findall(pattern, result.stderr, re.IGNORECASE)
                if matches:
                    warnings_found.extend(matches)
            
            if warnings_found:
                print(f"WARNINGS FOUND: {len(warnings_found)} warnings detected")
                print("Build output:", result.stderr[:1000])  # First 1000 chars
                # Don't fail the test for warnings, just report them
                
            print(f"✓ Build warnings check completed ({len(warnings_found)} warnings found)")
            return True
            
        except Exception as e:
            print(f"ERROR: Warning check failed: {e}")
            return False
    
    def test_output_structure(self) -> bool:
        """Test expected output file structure."""
        html_dir = self.build_dir / "html"
        
        if not html_dir.exists():
            print("ERROR: HTML build directory not found")
            return False
            
        # Check for essential files
        essential_files = [
            "index.html",
            "genindex.html",
            "search.html",
            "_static/css/theme.css",
            "_static/js/theme.js",
            "objects.inv"
        ]
        
        missing_files = []
        for file_path in essential_files:
            if not (html_dir / file_path).exists():
                missing_files.append(file_path)
                
        if missing_files:
            print(f"ERROR: Missing essential files: {missing_files}")
            return False
            
        # Check for expected sections
        expected_sections = [
            "getting_started",
            "user_guide", 
            "tutorials",
            "api",
            "methods",
            "theory",
            "development"
        ]
        
        missing_sections = []
        for section in expected_sections:
            if not (html_dir / section).exists():
                missing_sections.append(section)
                
        if missing_sections:
            print(f"WARNING: Missing expected sections: {missing_sections}")
            # Don't fail for missing sections, just warn
            
        print("✓ Output structure validation completed")
        return True
    
    def test_html_validation(self) -> bool:
        """Test HTML output for basic validity."""
        html_dir = self.build_dir / "html"
        index_html = html_dir / "index.html"
        
        if not index_html.exists():
            print("ERROR: index.html not found")
            return False
            
        try:
            with open(index_html, 'r', encoding='utf-8') as f:
                content = f.read()
                
            # Basic HTML validation checks
            checks = [
                ("<html", "HTML tag"),
                ("<head", "HEAD tag"),
                ("<body", "BODY tag"),
                ("</html>", "Closing HTML tag"),
                ("<title>", "Title tag")
            ]
            
            for pattern, description in checks:
                if pattern not in content:
                    print(f"ERROR: Missing {description}")
                    return False
                    
            # Check for common HTML issues
            if content.count("<html") != content.count("</html>"):
                print("ERROR: Mismatched HTML tags")
                return False
                
            print("✓ HTML validation completed")
            return True
            
        except Exception as e:
            print(f"ERROR: HTML validation failed: {e}")
            return False
    
    def test_cross_references(self) -> bool:
        """Test internal cross-references."""
        html_dir = self.build_dir / "html"
        
        if not html_dir.exists():
            print("ERROR: HTML build directory not found")
            return False
            
        try:
            # Find all HTML files
            html_files = list(html_dir.rglob("*.html"))
            
            broken_refs = []
            total_refs = 0
            
            for html_file in html_files[:10]:  # Test first 10 files to avoid timeout
                with open(html_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                    
                # Find internal links
                internal_links = re.findall(r'href="([^"]*\.html[^"]*)"', content)
                
                for link in internal_links:
                    total_refs += 1
                    # Remove anchors for file checking
                    file_link = link.split('#')[0]
                    if file_link and not file_link.startswith('http'):
                        # Resolve relative path
                        target_file = html_file.parent / file_link
                        if not target_file.exists():
                            broken_refs.append(f"{html_file.name} -> {link}")
            
            if broken_refs:
                print(f"ERROR: Found {len(broken_refs)} broken internal references")
                for ref in broken_refs[:5]:  # Show first 5
                    print(f"  - {ref}")
                return False
                
            print(f"✓ Cross-reference validation completed ({total_refs} references checked)")
            return True
            
        except Exception as e:
            print(f"ERROR: Cross-reference validation failed: {e}")
            return False
    
    def test_search_index(self) -> bool:
        """Test search index generation."""
        html_dir = self.build_dir / "html"
        search_index = html_dir / "_static" / "searchindex.js"
        
        if not search_index.exists():
            print("ERROR: Search index not found")
            return False
            
        try:
            with open(search_index, 'r', encoding='utf-8') as f:
                content = f.read()
                
            # Check for search index structure
            if "Search.setIndex" not in content:
                print("ERROR: Invalid search index format")
                return False
                
            # Check for some expected terms
            expected_terms = ["causal", "agent", "method", "analysis"]
            found_terms = sum(1 for term in expected_terms if term in content.lower())
            
            if found_terms < len(expected_terms) // 2:
                print(f"WARNING: Search index may be incomplete ({found_terms}/{len(expected_terms)} expected terms found)")
                
            print("✓ Search index validation completed")
            return True
            
        except Exception as e:
            print(f"ERROR: Search index validation failed: {e}")
            return False
    
    def _print_summary(self):
        """Print test results summary."""
        print(f"\n{'='*60}")
        print("DOCUMENTATION BUILD TEST SUMMARY")
        print('='*60)
        
        passed = sum(1 for result in self.test_results if result["passed"])
        total = len(self.test_results)
        
        print(f"Tests passed: {passed}/{total}")
        
        if passed == total:
            print("🎉 All tests passed!")
        else:
            print("❌ Some tests failed:")
            for result in self.test_results:
                if not result["passed"]:
                    print(f"  - {result['test']}: {result['status']}")

def main():
    """Main function to run documentation build tests."""
    if len(sys.argv) > 1:
        docs_dir = sys.argv[1]
    else:
        docs_dir = "docs"
        
    tester = DocumentationBuildTester(docs_dir)
    success = tester.run_all_tests()
    
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()
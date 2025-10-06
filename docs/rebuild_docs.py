#!/usr/bin/env python3
"""
Simple script to rebuild documentation after content changes.
"""

import os
import sys
import shutil
import subprocess
from pathlib import Path

def clean_build():
    """Clean the build directory."""
    build_path = Path("build")
    if build_path.exists():
        print("🧹 Cleaning build directory...")
        shutil.rmtree(build_path)
        print("✅ Build directory cleaned")
    else:
        print("ℹ️ Build directory already clean")

def build_docs():
    """Build the documentation."""
    print("🔨 Building documentation...")
    
    try:
        # Try using make first
        result = subprocess.run(['make', 'html'], capture_output=True, text=True)
        if result.returncode == 0:
            print("✅ Documentation built successfully with make")
            return True
        else:
            print("⚠️ Make failed, trying sphinx-build directly...")
            
    except FileNotFoundError:
        print("⚠️ Make not found, trying sphinx-build directly...")
    
    # Try sphinx-build directly
    try:
        result = subprocess.run([
            'sphinx-build', '-b', 'html', 'source', 'build/html'
        ], capture_output=True, text=True)
        
        if result.returncode == 0:
            print("✅ Documentation built successfully with sphinx-build")
            return True
        else:
            print(f"❌ Sphinx build failed: {result.stderr}")
            return False
            
    except FileNotFoundError:
        print("❌ sphinx-build not found. Please install Sphinx:")
        print("   pip install -r requirements.txt")
        return False

def start_server():
    """Start a local server to view the documentation."""
    html_path = Path("build/html")
    if not html_path.exists():
        print("❌ HTML build not found. Build failed.")
        return
    
    print("🌐 Starting local server...")
    print("📖 Documentation will be available at: http://localhost:8000")
    print("🛑 Press Ctrl+C to stop the server")
    
    try:
        os.chdir(html_path)
        subprocess.run([sys.executable, '-m', 'http.server', '8000'])
    except KeyboardInterrupt:
        print("\n🛑 Server stopped")

def check_dependencies():
    """Check if required dependencies are available."""
    print("🔍 Checking dependencies...")
    
    missing = []
    
    # Check for sphinx
    try:
        import sphinx
        print(f"✅ Sphinx {sphinx.__version__} found")
    except ImportError:
        missing.append("sphinx")
    
    # Check for theme
    try:
        import sphinx_rtd_theme
        print("✅ ReadTheDocs theme found")
    except ImportError:
        missing.append("sphinx_rtd_theme")
    
    if missing:
        print(f"❌ Missing dependencies: {', '.join(missing)}")
        print("📦 Install with: pip install -r requirements.txt")
        return False
    
    return True

def main():
    """Main rebuild function."""
    print("🚀 Documentation Rebuild Script")
    print("=" * 40)
    
    # Check if we're in the right directory
    if not Path("source").exists():
        print("❌ Please run this script from the docs/ directory")
        sys.exit(1)
    
    # Check dependencies
    if not check_dependencies():
        sys.exit(1)
    
    # Clean and build
    clean_build()
    
    if build_docs():
        print("\n🎉 Documentation rebuilt successfully!")
        
        # Ask if user wants to start server
        try:
            response = input("\n🌐 Start local server to view docs? (y/n): ").lower().strip()
            if response in ['y', 'yes']:
                start_server()
            else:
                print("📁 Documentation is available in: build/html/index.html")
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
    else:
        print("\n❌ Documentation build failed!")
        sys.exit(1)

if __name__ == "__main__":
    main()
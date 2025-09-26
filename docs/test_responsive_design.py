#!/usr/bin/env python3
"""
Responsive design and accessibility testing for CAIS documentation.
Tests various viewport sizes, accessibility compliance, and performance metrics.
"""

import os
import sys
import json
import time
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, asdict
from urllib.parse import urljoin
import subprocess

try:
    from selenium import webdriver
    from selenium.webdriver.common.by import By
    from selenium.webdriver.support.ui import WebDriverWait
    from selenium.webdriver.support import expected_conditions as EC
    from selenium.webdriver.chrome.options import Options as ChromeOptions
    from selenium.webdriver.firefox.options import Options as FirefoxOptions
    from selenium.webdriver.common.action_chains import ActionChains
    from selenium.webdriver.common.keys import Keys
    from selenium.common.exceptions import TimeoutException, NoSuchElementException
    SELENIUM_AVAILABLE = True
except ImportError:
    SELENIUM_AVAILABLE = False
    print("Warning: Selenium not available. Install with: pip install selenium")

try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False
    print("Warning: requests not available. Install with: pip install requests")

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class ViewportSize:
    """Represents a viewport size for testing"""
    name: str
    width: int
    height: int
    device_type: str

@dataclass
class AccessibilityIssue:
    """Represents an accessibility issue found during testing"""
    type: str
    severity: str
    element: str
    description: str
    wcag_guideline: str

@dataclass
class PerformanceMetrics:
    """Represents performance metrics for a page"""
    load_time: float
    dom_content_loaded: float
    first_contentful_paint: Optional[float]
    largest_contentful_paint: Optional[float]
    cumulative_layout_shift: Optional[float]
    first_input_delay: Optional[float]

@dataclass
class TestResult:
    """Represents the result of a responsive design test"""
    url: str
    viewport: ViewportSize
    browser: str
    passed: bool
    issues: List[str]
    accessibility_issues: List[AccessibilityIssue]
    performance_metrics: PerformanceMetrics
    screenshot_path: Optional[str]

class ResponsiveDesignTester:
    """Main class for testing responsive design and accessibility"""
    
    # Standard viewport sizes for testing
    VIEWPORT_SIZES = [
        ViewportSize("Mobile Small", 320, 568, "mobile"),
        ViewportSize("Mobile Medium", 375, 667, "mobile"),
        ViewportSize("Mobile Large", 414, 896, "mobile"),
        ViewportSize("Tablet Portrait", 768, 1024, "tablet"),
        ViewportSize("Tablet Landscape", 1024, 768, "tablet"),
        ViewportSize("Desktop Small", 1280, 720, "desktop"),
        ViewportSize("Desktop Medium", 1440, 900, "desktop"),
        ViewportSize("Desktop Large", 1920, 1080, "desktop"),
        ViewportSize("Desktop XL", 2560, 1440, "desktop"),
    ]
    
    # Browsers to test
    BROWSERS = ["chrome", "firefox"]
    
    # Key pages to test
    TEST_PAGES = [
        "",  # Homepage
        "getting_started/",
        "methods/",
        "methods/decision_tree.html",
        "tutorials/",
        "api/",
        "search.html?q=causal+inference",
    ]
    
    def __init__(self, base_url: str = "http://localhost:8000", output_dir: str = "test_results"):
        self.base_url = base_url.rstrip('/')
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.drivers = {}
        
    def setup_driver(self, browser: str, viewport: ViewportSize) -> webdriver.Remote:
        """Setup WebDriver for the specified browser and viewport"""
        if not SELENIUM_AVAILABLE:
            raise RuntimeError("Selenium is required for responsive design testing")
            
        if browser == "chrome":
            options = ChromeOptions()
            options.add_argument("--headless")
            options.add_argument("--no-sandbox")
            options.add_argument("--disable-dev-shm-usage")
            options.add_argument("--disable-gpu")
            options.add_argument(f"--window-size={viewport.width},{viewport.height}")
            options.add_experimental_option('useAutomationExtension', False)
            options.add_experimental_option("excludeSwitches", ["enable-automation"])
            
            driver = webdriver.Chrome(options=options)
            
        elif browser == "firefox":
            options = FirefoxOptions()
            options.add_argument("--headless")
            options.add_argument(f"--width={viewport.width}")
            options.add_argument(f"--height={viewport.height}")
            
            driver = webdriver.Firefox(options=options)
            
        else:
            raise ValueError(f"Unsupported browser: {browser}")
        
        driver.set_window_size(viewport.width, viewport.height)
        return driver
    
    def test_page_responsive_design(self, url: str, viewport: ViewportSize, browser: str) -> TestResult:
        """Test responsive design for a single page"""
        logger.info(f"Testing {url} on {browser} at {viewport.name} ({viewport.width}x{viewport.height})")
        
        driver = self.setup_driver(browser, viewport)
        issues = []
        accessibility_issues = []
        screenshot_path = None
        
        try:
            # Load the page and measure performance
            start_time = time.time()
            driver.get(url)
            
            # Wait for page to load
            WebDriverWait(driver, 10).until(
                EC.presence_of_element_located((By.TAG_NAME, "body"))
            )
            
            load_time = time.time() - start_time
            
            # Get performance metrics
            performance_metrics = self.get_performance_metrics(driver, load_time)
            
            # Test responsive design elements
            issues.extend(self.test_navigation_responsiveness(driver, viewport))
            issues.extend(self.test_content_layout(driver, viewport))
            issues.extend(self.test_interactive_elements(driver, viewport))
            issues.extend(self.test_typography_scaling(driver, viewport))
            issues.extend(self.test_image_responsiveness(driver, viewport))
            
            # Test accessibility
            accessibility_issues.extend(self.test_accessibility_compliance(driver))
            
            # Take screenshot
            screenshot_path = self.take_screenshot(driver, url, viewport, browser)
            
            passed = len(issues) == 0 and len([issue for issue in accessibility_issues if issue.severity == "error"]) == 0
            
        except Exception as e:
            logger.error(f"Error testing {url}: {str(e)}")
            issues.append(f"Test failed with error: {str(e)}")
            performance_metrics = PerformanceMetrics(0, 0, None, None, None, None)
            passed = False
            
        finally:
            driver.quit()
        
        return TestResult(
            url=url,
            viewport=viewport,
            browser=browser,
            passed=passed,
            issues=issues,
            accessibility_issues=accessibility_issues,
            performance_metrics=performance_metrics,
            screenshot_path=screenshot_path
        )
    
    def get_performance_metrics(self, driver: webdriver.Remote, load_time: float) -> PerformanceMetrics:
        """Get performance metrics from the browser"""
        try:
            # Get navigation timing
            nav_timing = driver.execute_script("""
                return {
                    domContentLoaded: performance.timing.domContentLoadedEventEnd - performance.timing.navigationStart,
                    loadComplete: performance.timing.loadEventEnd - performance.timing.navigationStart
                };
            """)
            
            # Try to get Core Web Vitals (may not be available in all browsers)
            try:
                web_vitals = driver.execute_script("""
                    return new Promise((resolve) => {
                        if ('PerformanceObserver' in window) {
                            const observer = new PerformanceObserver((list) => {
                                const entries = list.getEntries();
                                const metrics = {};
                                
                                entries.forEach((entry) => {
                                    if (entry.entryType === 'paint') {
                                        if (entry.name === 'first-contentful-paint') {
                                            metrics.firstContentfulPaint = entry.startTime;
                                        }
                                    } else if (entry.entryType === 'largest-contentful-paint') {
                                        metrics.largestContentfulPaint = entry.startTime;
                                    } else if (entry.entryType === 'layout-shift') {
                                        if (!metrics.cumulativeLayoutShift) {
                                            metrics.cumulativeLayoutShift = 0;
                                        }
                                        metrics.cumulativeLayoutShift += entry.value;
                                    }
                                });
                                
                                resolve(metrics);
                            });
                            
                            observer.observe({entryTypes: ['paint', 'largest-contentful-paint', 'layout-shift']});
                            
                            // Resolve after 2 seconds if no metrics are collected
                            setTimeout(() => resolve({}), 2000);
                        } else {
                            resolve({});
                        }
                    });
                """)
            except:
                web_vitals = {}
            
            return PerformanceMetrics(
                load_time=load_time,
                dom_content_loaded=nav_timing.get('domContentLoaded', 0) / 1000,
                first_contentful_paint=web_vitals.get('firstContentfulPaint'),
                largest_contentful_paint=web_vitals.get('largestContentfulPaint'),
                cumulative_layout_shift=web_vitals.get('cumulativeLayoutShift'),
                first_input_delay=None  # Requires user interaction to measure
            )
            
        except Exception as e:
            logger.warning(f"Could not get performance metrics: {str(e)}")
            return PerformanceMetrics(load_time, 0, None, None, None, None)
    
    def test_navigation_responsiveness(self, driver: webdriver.Remote, viewport: ViewportSize) -> List[str]:
        """Test navigation responsiveness"""
        issues = []
        
        try:
            # Check if navigation is accessible
            nav_elements = driver.find_elements(By.CSS_SELECTOR, ".wy-nav-side, nav, [role='navigation']")
            
            if not nav_elements:
                issues.append("No navigation elements found")
                return issues
            
            nav = nav_elements[0]
            
            # Check if navigation is visible or has mobile toggle
            if viewport.device_type == "mobile":
                # On mobile, navigation should be collapsible or hidden by default
                nav_visible = nav.is_displayed()
                mobile_toggle = driver.find_elements(By.CSS_SELECTOR, ".wy-nav-top, .mobile-nav-toggle, [aria-label*='menu']")
                
                if nav_visible and not mobile_toggle:
                    issues.append("Navigation should be collapsible on mobile devices")
                    
            else:
                # On desktop, navigation should be visible
                if not nav.is_displayed():
                    issues.append("Navigation should be visible on desktop")
            
            # Check navigation links are clickable
            nav_links = nav.find_elements(By.TAG_NAME, "a")
            for link in nav_links[:5]:  # Test first 5 links
                if link.is_displayed():
                    link_rect = link.rect
                    if link_rect['width'] < 44 or link_rect['height'] < 44:
                        issues.append(f"Navigation link too small for touch: {link.text[:30]}")
                        
        except Exception as e:
            issues.append(f"Navigation test failed: {str(e)}")
        
        return issues
    
    def test_content_layout(self, driver: webdriver.Remote, viewport: ViewportSize) -> List[str]:
        """Test content layout responsiveness"""
        issues = []
        
        try:
            # Check for horizontal scrolling (should be avoided)
            body_width = driver.execute_script("return document.body.scrollWidth")
            viewport_width = driver.execute_script("return window.innerWidth")
            
            if body_width > viewport_width + 10:  # Allow small tolerance
                issues.append(f"Horizontal scrolling detected: body width {body_width}px > viewport {viewport_width}px")
            
            # Check content container width
            content_containers = driver.find_elements(By.CSS_SELECTOR, ".wy-nav-content, main, .content, .container")
            
            for container in content_containers:
                if container.is_displayed():
                    container_rect = container.rect
                    if container_rect['width'] > viewport_width:
                        issues.append(f"Content container too wide: {container_rect['width']}px")
            
            # Check for overlapping elements
            overlapping = driver.execute_script("""
                const elements = document.querySelectorAll('*');
                const overlaps = [];
                
                for (let i = 0; i < elements.length; i++) {
                    const rect1 = elements[i].getBoundingClientRect();
                    if (rect1.width === 0 || rect1.height === 0) continue;
                    
                    for (let j = i + 1; j < elements.length; j++) {
                        const rect2 = elements[j].getBoundingClientRect();
                        if (rect2.width === 0 || rect2.height === 0) continue;
                        
                        // Check if elements overlap significantly
                        const overlapX = Math.max(0, Math.min(rect1.right, rect2.right) - Math.max(rect1.left, rect2.left));
                        const overlapY = Math.max(0, Math.min(rect1.bottom, rect2.bottom) - Math.max(rect1.top, rect2.top));
                        
                        if (overlapX > rect1.width * 0.8 && overlapY > rect1.height * 0.8) {
                            overlaps.push({
                                element1: elements[i].tagName + (elements[i].className ? '.' + elements[i].className.split(' ')[0] : ''),
                                element2: elements[j].tagName + (elements[j].className ? '.' + elements[j].className.split(' ')[0] : '')
                            });
                        }
                    }
                }
                
                return overlaps.slice(0, 5); // Return first 5 overlaps
            """)
            
            for overlap in overlapping:
                issues.append(f"Overlapping elements detected: {overlap['element1']} and {overlap['element2']}")
                
        except Exception as e:
            issues.append(f"Layout test failed: {str(e)}")
        
        return issues
    
    def test_interactive_elements(self, driver: webdriver.Remote, viewport: ViewportSize) -> List[str]:
        """Test interactive elements for touch-friendliness"""
        issues = []
        
        try:
            # Find interactive elements
            interactive_selectors = [
                "button", "a", "input", "textarea", "select",
                "[role='button']", "[tabindex]", ".btn", ".filter-btn"
            ]
            
            for selector in interactive_selectors:
                elements = driver.find_elements(By.CSS_SELECTOR, selector)
                
                for element in elements[:10]:  # Test first 10 of each type
                    if element.is_displayed():
                        rect = element.rect
                        
                        # Check minimum touch target size (44x44px recommended)
                        if viewport.device_type in ["mobile", "tablet"]:
                            if rect['width'] < 44 or rect['height'] < 44:
                                element_text = element.text[:30] if element.text else element.get_attribute('aria-label') or selector
                                issues.append(f"Touch target too small: {element_text} ({rect['width']}x{rect['height']}px)")
                        
                        # Check if element is clickable
                        try:
                            if element.is_enabled():
                                # Try to hover (simulates touch on mobile)
                                ActionChains(driver).move_to_element(element).perform()
                        except Exception:
                            issues.append(f"Interactive element not accessible: {element.text[:30]}")
                            
        except Exception as e:
            issues.append(f"Interactive elements test failed: {str(e)}")
        
        return issues
    
    def test_typography_scaling(self, driver: webdriver.Remote, viewport: ViewportSize) -> List[str]:
        """Test typography scaling and readability"""
        issues = []
        
        try:
            # Check font sizes
            text_elements = driver.find_elements(By.CSS_SELECTOR, "p, h1, h2, h3, h4, h5, h6, li, span")
            
            for element in text_elements[:20]:  # Test first 20 text elements
                if element.is_displayed() and element.text.strip():
                    font_size = driver.execute_script(
                        "return window.getComputedStyle(arguments[0]).fontSize", element
                    )
                    
                    font_size_px = float(font_size.replace('px', ''))
                    
                    # Check minimum font size for readability
                    min_font_size = 14 if viewport.device_type == "mobile" else 12
                    
                    if font_size_px < min_font_size:
                        issues.append(f"Font too small: {font_size_px}px (minimum {min_font_size}px)")
                    
                    # Check line height
                    line_height = driver.execute_script(
                        "return window.getComputedStyle(arguments[0]).lineHeight", element
                    )
                    
                    if line_height != "normal" and line_height.endswith('px'):
                        line_height_px = float(line_height.replace('px', ''))
                        if line_height_px / font_size_px < 1.2:
                            issues.append(f"Line height too small: {line_height_px / font_size_px:.2f} (minimum 1.2)")
                            
        except Exception as e:
            issues.append(f"Typography test failed: {str(e)}")
        
        return issues
    
    def test_image_responsiveness(self, driver: webdriver.Remote, viewport: ViewportSize) -> List[str]:
        """Test image responsiveness"""
        issues = []
        
        try:
            images = driver.find_elements(By.TAG_NAME, "img")
            
            for img in images:
                if img.is_displayed():
                    rect = img.rect
                    
                    # Check if image overflows container
                    if rect['width'] > viewport.width:
                        issues.append(f"Image too wide: {rect['width']}px > viewport {viewport.width}px")
                    
                    # Check if image has proper alt text
                    alt_text = img.get_attribute('alt')
                    if not alt_text and not img.get_attribute('aria-label'):
                        issues.append("Image missing alt text")
                    
                    # Check if image is properly sized
                    natural_width = driver.execute_script("return arguments[0].naturalWidth", img)
                    if natural_width and rect['width'] > natural_width * 2:
                        issues.append(f"Image upscaled significantly: displayed {rect['width']}px vs natural {natural_width}px")
                        
        except Exception as e:
            issues.append(f"Image responsiveness test failed: {str(e)}")
        
        return issues
    
    def test_accessibility_compliance(self, driver: webdriver.Remote) -> List[AccessibilityIssue]:
        """Test accessibility compliance"""
        issues = []
        
        try:
            # Test 1: Check for proper heading hierarchy
            headings = driver.find_elements(By.CSS_SELECTOR, "h1, h2, h3, h4, h5, h6")
            previous_level = 0
            
            for heading in headings:
                if heading.is_displayed():
                    level = int(heading.tag_name[1])
                    if level > previous_level + 1:
                        issues.append(AccessibilityIssue(
                            type="heading_hierarchy",
                            severity="warning",
                            element=heading.tag_name,
                            description=f"Heading level skipped: {heading.text[:50]}",
                            wcag_guideline="1.3.1"
                        ))
                    previous_level = level
            
            # Test 2: Check for images without alt text
            images = driver.find_elements(By.CSS_SELECTOR, "img:not([alt]), img[alt='']")
            for img in images:
                if img.is_displayed():
                    issues.append(AccessibilityIssue(
                        type="missing_alt_text",
                        severity="error",
                        element="img",
                        description=f"Image missing alt text: {img.get_attribute('src')[:50]}",
                        wcag_guideline="1.1.1"
                    ))
            
            # Test 3: Check for form inputs without labels
            inputs = driver.find_elements(By.CSS_SELECTOR, "input:not([aria-label]):not([aria-labelledby])")
            for input_elem in inputs:
                if input_elem.is_displayed():
                    input_id = input_elem.get_attribute('id')
                    if input_id:
                        label = driver.find_elements(By.CSS_SELECTOR, f"label[for='{input_id}']")
                        if not label:
                            issues.append(AccessibilityIssue(
                                type="missing_label",
                                severity="error",
                                element="input",
                                description=f"Input missing label: {input_elem.get_attribute('name') or input_elem.get_attribute('type')}",
                                wcag_guideline="1.3.1"
                            ))
            
            # Test 4: Check for sufficient color contrast (basic check)
            # This is a simplified check - full contrast testing requires more sophisticated tools
            elements_to_check = driver.find_elements(By.CSS_SELECTOR, "p, h1, h2, h3, h4, h5, h6, a, button")
            for element in elements_to_check[:10]:  # Check first 10 elements
                if element.is_displayed():
                    try:
                        color = driver.execute_script(
                            "return window.getComputedStyle(arguments[0]).color", element
                        )
                        bg_color = driver.execute_script(
                            "return window.getComputedStyle(arguments[0]).backgroundColor", element
                        )
                        
                        # Simple check for very light text on light background
                        if color.startswith('rgb(') and bg_color.startswith('rgb('):
                            # Extract RGB values (simplified)
                            color_values = [int(x) for x in color.replace('rgb(', '').replace(')', '').split(',')]
                            bg_values = [int(x) for x in bg_color.replace('rgb(', '').replace(')', '').split(',')]
                            
                            # Calculate simple brightness difference
                            color_brightness = sum(color_values) / 3
                            bg_brightness = sum(bg_values) / 3
                            
                            if abs(color_brightness - bg_brightness) < 50:  # Very similar brightness
                                issues.append(AccessibilityIssue(
                                    type="low_contrast",
                                    severity="warning",
                                    element=element.tag_name,
                                    description=f"Potential low contrast: {element.text[:30]}",
                                    wcag_guideline="1.4.3"
                                ))
                    except:
                        pass  # Skip if color parsing fails
            
            # Test 5: Check for keyboard accessibility
            focusable_elements = driver.find_elements(By.CSS_SELECTOR, 
                "a, button, input, textarea, select, [tabindex]:not([tabindex='-1'])")
            
            for element in focusable_elements[:5]:  # Test first 5 focusable elements
                if element.is_displayed():
                    try:
                        element.send_keys(Keys.TAB)
                        if driver.switch_to.active_element != element:
                            issues.append(AccessibilityIssue(
                                type="keyboard_inaccessible",
                                severity="error",
                                element=element.tag_name,
                                description=f"Element not keyboard accessible: {element.text[:30]}",
                                wcag_guideline="2.1.1"
                            ))
                    except:
                        pass  # Skip if focus test fails
                        
        except Exception as e:
            logger.warning(f"Accessibility test failed: {str(e)}")
        
        return issues
    
    def take_screenshot(self, driver: webdriver.Remote, url: str, viewport: ViewportSize, browser: str) -> str:
        """Take a screenshot of the current page"""
        try:
            # Create screenshot directory
            screenshot_dir = self.output_dir / "screenshots"
            screenshot_dir.mkdir(exist_ok=True)
            
            # Generate filename
            page_name = url.split('/')[-1] or "homepage"
            filename = f"{page_name}_{browser}_{viewport.name.replace(' ', '_')}.png"
            screenshot_path = screenshot_dir / filename
            
            # Take screenshot
            driver.save_screenshot(str(screenshot_path))
            
            return str(screenshot_path)
            
        except Exception as e:
            logger.warning(f"Could not take screenshot: {str(e)}")
            return None
    
    def run_all_tests(self) -> List[TestResult]:
        """Run all responsive design tests"""
        results = []
        
        if not SELENIUM_AVAILABLE:
            logger.error("Selenium is required for testing. Install with: pip install selenium")
            return results
        
        # Test if local server is running
        if not self.is_server_running():
            logger.error(f"Documentation server not running at {self.base_url}")
            logger.info("Start the server with: cd docs && python -m http.server 8000")
            return results
        
        total_tests = len(self.TEST_PAGES) * len(self.VIEWPORT_SIZES) * len(self.BROWSERS)
        current_test = 0
        
        for page in self.TEST_PAGES:
            url = urljoin(self.base_url + '/', page)
            
            for viewport in self.VIEWPORT_SIZES:
                for browser in self.BROWSERS:
                    current_test += 1
                    logger.info(f"Running test {current_test}/{total_tests}")
                    
                    try:
                        result = self.test_page_responsive_design(url, viewport, browser)
                        results.append(result)
                    except Exception as e:
                        logger.error(f"Test failed: {str(e)}")
                        results.append(TestResult(
                            url=url,
                            viewport=viewport,
                            browser=browser,
                            passed=False,
                            issues=[f"Test failed: {str(e)}"],
                            accessibility_issues=[],
                            performance_metrics=PerformanceMetrics(0, 0, None, None, None, None),
                            screenshot_path=None
                        ))
        
        return results
    
    def is_server_running(self) -> bool:
        """Check if the documentation server is running"""
        if not REQUESTS_AVAILABLE:
            return True  # Assume it's running if we can't check
            
        try:
            response = requests.get(self.base_url, timeout=5)
            return response.status_code == 200
        except:
            return False
    
    def generate_report(self, results: List[TestResult]) -> str:
        """Generate a comprehensive test report"""
        report_path = self.output_dir / "responsive_design_report.html"
        
        # Calculate summary statistics
        total_tests = len(results)
        passed_tests = len([r for r in results if r.passed])
        failed_tests = total_tests - passed_tests
        
        # Group results by page
        pages = {}
        for result in results:
            page = result.url.split('/')[-1] or "homepage"
            if page not in pages:
                pages[page] = []
            pages[page].append(result)
        
        # Generate HTML report
        html_content = f"""
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>CAIS Documentation - Responsive Design Test Report</title>
            <style>
                body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; margin: 0; padding: 20px; background: #f8f9fa; }}
                .container {{ max-width: 1200px; margin: 0 auto; background: white; padding: 30px; border-radius: 8px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }}
                h1 {{ color: #2c3e50; border-bottom: 3px solid #3498db; padding-bottom: 10px; }}
                h2 {{ color: #34495e; margin-top: 30px; }}
                .summary {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 20px; margin: 20px 0; }}
                .stat-card {{ background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 20px; border-radius: 8px; text-align: center; }}
                .stat-card.success {{ background: linear-gradient(135deg, #4CAF50 0%, #45a049 100%); }}
                .stat-card.error {{ background: linear-gradient(135deg, #f44336 0%, #da190b 100%); }}
                .stat-card.warning {{ background: linear-gradient(135deg, #ff9800 0%, #f57c00 100%); }}
                .stat-number {{ font-size: 2em; font-weight: bold; margin-bottom: 5px; }}
                .stat-label {{ font-size: 0.9em; opacity: 0.9; }}
                .test-results {{ margin-top: 30px; }}
                .page-section {{ margin-bottom: 40px; border: 1px solid #e1e8ed; border-radius: 8px; overflow: hidden; }}
                .page-header {{ background: #f8f9fa; padding: 15px 20px; border-bottom: 1px solid #e1e8ed; }}
                .page-title {{ margin: 0; color: #2c3e50; }}
                .viewport-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 15px; padding: 20px; }}
                .viewport-card {{ border: 1px solid #e1e8ed; border-radius: 6px; padding: 15px; }}
                .viewport-card.passed {{ border-left: 4px solid #4CAF50; }}
                .viewport-card.failed {{ border-left: 4px solid #f44336; }}
                .viewport-header {{ display: flex; justify-content: space-between; align-items: center; margin-bottom: 10px; }}
                .viewport-name {{ font-weight: 600; color: #2c3e50; }}
                .status-badge {{ padding: 4px 8px; border-radius: 4px; font-size: 0.8em; font-weight: 600; }}
                .status-badge.passed {{ background: #d4edda; color: #155724; }}
                .status-badge.failed {{ background: #f8d7da; color: #721c24; }}
                .issues-list {{ margin-top: 10px; }}
                .issue {{ background: #fff3cd; border: 1px solid #ffeaa7; border-radius: 4px; padding: 8px; margin: 5px 0; font-size: 0.9em; }}
                .issue.error {{ background: #f8d7da; border-color: #f5c6cb; }}
                .issue.warning {{ background: #fff3cd; border-color: #ffeaa7; }}
                .performance-metrics {{ margin-top: 10px; font-size: 0.9em; color: #6c757d; }}
                .screenshot {{ max-width: 100%; height: auto; border-radius: 4px; margin-top: 10px; }}
                .accessibility-summary {{ background: #e7f3ff; border: 1px solid #b3d9ff; border-radius: 6px; padding: 15px; margin: 20px 0; }}
                .browser-tabs {{ display: flex; margin-bottom: 15px; }}
                .browser-tab {{ padding: 8px 16px; background: #f8f9fa; border: 1px solid #e1e8ed; cursor: pointer; }}
                .browser-tab.active {{ background: #007bff; color: white; }}
                .browser-content {{ display: none; }}
                .browser-content.active {{ display: block; }}
            </style>
        </head>
        <body>
            <div class="container">
                <h1>CAIS Documentation - Responsive Design Test Report</h1>
                
                <div class="summary">
                    <div class="stat-card">
                        <div class="stat-number">{total_tests}</div>
                        <div class="stat-label">Total Tests</div>
                    </div>
                    <div class="stat-card success">
                        <div class="stat-number">{passed_tests}</div>
                        <div class="stat-label">Passed</div>
                    </div>
                    <div class="stat-card error">
                        <div class="stat-number">{failed_tests}</div>
                        <div class="stat-label">Failed</div>
                    </div>
                    <div class="stat-card warning">
                        <div class="stat-number">{len([r for r in results if r.accessibility_issues])}</div>
                        <div class="stat-label">Accessibility Issues</div>
                    </div>
                </div>
                
                <div class="accessibility-summary">
                    <h3>Accessibility Summary</h3>
                    <p>Total accessibility issues found: {sum(len(r.accessibility_issues) for r in results)}</p>
                    <ul>
                        <li>Errors: {sum(len([i for i in r.accessibility_issues if i.severity == 'error']) for r in results)}</li>
                        <li>Warnings: {sum(len([i for i in r.accessibility_issues if i.severity == 'warning']) for r in results)}</li>
                    </ul>
                </div>
        """
        
        # Add detailed results for each page
        for page_name, page_results in pages.items():
            html_content += f"""
                <div class="page-section">
                    <div class="page-header">
                        <h2 class="page-title">{page_name.title() if page_name != 'homepage' else 'Homepage'}</h2>
                    </div>
                    
                    <div class="browser-tabs">
                        <div class="browser-tab active" onclick="showBrowser('{page_name}', 'chrome')">Chrome</div>
                        <div class="browser-tab" onclick="showBrowser('{page_name}', 'firefox')">Firefox</div>
                    </div>
            """
            
            # Group by browser
            for browser in self.BROWSERS:
                browser_results = [r for r in page_results if r.browser == browser]
                
                html_content += f"""
                    <div class="browser-content {'active' if browser == 'chrome' else ''}" id="{page_name}-{browser}">
                        <div class="viewport-grid">
                """
                
                for result in browser_results:
                    status_class = "passed" if result.passed else "failed"
                    status_text = "PASSED" if result.passed else "FAILED"
                    
                    html_content += f"""
                        <div class="viewport-card {status_class}">
                            <div class="viewport-header">
                                <span class="viewport-name">{result.viewport.name}</span>
                                <span class="status-badge {status_class}">{status_text}</span>
                            </div>
                            <div class="viewport-details">
                                <strong>{result.viewport.width}x{result.viewport.height}</strong> ({result.viewport.device_type})
                            </div>
                    """
                    
                    if result.issues:
                        html_content += '<div class="issues-list">'
                        for issue in result.issues:
                            html_content += f'<div class="issue">{issue}</div>'
                        html_content += '</div>'
                    
                    if result.accessibility_issues:
                        html_content += '<div class="issues-list">'
                        for issue in result.accessibility_issues:
                            issue_class = "error" if issue.severity == "error" else "warning"
                            html_content += f'<div class="issue {issue_class}">A11Y {issue.severity.upper()}: {issue.description}</div>'
                        html_content += '</div>'
                    
                    # Performance metrics
                    perf = result.performance_metrics
                    html_content += f"""
                        <div class="performance-metrics">
                            <strong>Performance:</strong> Load: {perf.load_time:.2f}s, 
                            DOM: {perf.dom_content_loaded:.2f}s
                    """
                    if perf.first_contentful_paint:
                        html_content += f", FCP: {perf.first_contentful_paint:.0f}ms"
                    html_content += "</div>"
                    
                    # Screenshot
                    if result.screenshot_path:
                        html_content += f'<img src="{result.screenshot_path}" class="screenshot" alt="Screenshot of {result.viewport.name}">'
                    
                    html_content += "</div>"
                
                html_content += """
                        </div>
                    </div>
                """
            
            html_content += "</div>"
        
        html_content += """
                <script>
                    function showBrowser(page, browser) {
                        // Hide all browser contents for this page
                        const contents = document.querySelectorAll(`[id^="${page}-"]`);
                        contents.forEach(content => {
                            content.classList.remove('active');
                        });
                        
                        // Show selected browser content
                        document.getElementById(`${page}-${browser}`).classList.add('active');
                        
                        // Update tab styles
                        const tabs = document.querySelectorAll('.browser-tab');
                        tabs.forEach(tab => tab.classList.remove('active'));
                        event.target.classList.add('active');
                    }
                </script>
            </div>
        </body>
        </html>
        """
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        return str(report_path)
    
    def save_results_json(self, results: List[TestResult]) -> str:
        """Save results as JSON for further processing"""
        json_path = self.output_dir / "responsive_design_results.json"
        
        # Convert results to JSON-serializable format
        json_results = []
        for result in results:
            json_result = asdict(result)
            # Convert accessibility issues
            json_result['accessibility_issues'] = [asdict(issue) for issue in result.accessibility_issues]
            json_result['performance_metrics'] = asdict(result.performance_metrics)
            json_results.append(json_result)
        
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(json_results, f, indent=2)
        
        return str(json_path)

def main():
    """Main function to run responsive design tests"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Test responsive design and accessibility for CAIS documentation")
    parser.add_argument("--base-url", default="http://localhost:8000", help="Base URL of the documentation site")
    parser.add_argument("--output-dir", default="test_results", help="Output directory for test results")
    parser.add_argument("--browsers", nargs="+", default=["chrome", "firefox"], help="Browsers to test")
    parser.add_argument("--pages", nargs="+", help="Specific pages to test (default: all)")
    parser.add_argument("--viewports", nargs="+", help="Specific viewports to test (default: all)")
    
    args = parser.parse_args()
    
    # Create tester instance
    tester = ResponsiveDesignTester(args.base_url, args.output_dir)
    
    # Override browsers if specified
    if args.browsers:
        tester.BROWSERS = args.browsers
    
    # Override pages if specified
    if args.pages:
        tester.TEST_PAGES = args.pages
    
    # Override viewports if specified
    if args.viewports:
        viewport_names = [v.name for v in tester.VIEWPORT_SIZES]
        tester.VIEWPORT_SIZES = [v for v in tester.VIEWPORT_SIZES if v.name in args.viewports]
    
    logger.info("Starting responsive design tests...")
    logger.info(f"Testing {len(tester.TEST_PAGES)} pages across {len(tester.VIEWPORT_SIZES)} viewports in {len(tester.BROWSERS)} browsers")
    
    # Run tests
    results = tester.run_all_tests()
    
    if not results:
        logger.error("No tests were run. Check that Selenium is installed and the server is running.")
        return 1
    
    # Generate reports
    logger.info("Generating reports...")
    html_report = tester.generate_report(results)
    json_report = tester.save_results_json(results)
    
    # Print summary
    total_tests = len(results)
    passed_tests = len([r for r in results if r.passed])
    failed_tests = total_tests - passed_tests
    
    print(f"\n{'='*60}")
    print("RESPONSIVE DESIGN TEST SUMMARY")
    print(f"{'='*60}")
    print(f"Total tests: {total_tests}")
    print(f"Passed: {passed_tests}")
    print(f"Failed: {failed_tests}")
    print(f"Success rate: {(passed_tests/total_tests)*100:.1f}%")
    print(f"\nReports generated:")
    print(f"  HTML: {html_report}")
    print(f"  JSON: {json_report}")
    
    if failed_tests > 0:
        print(f"\n⚠️  {failed_tests} tests failed. Check the HTML report for details.")
        return 1
    else:
        print(f"\n✅ All tests passed!")
        return 0

if __name__ == "__main__":
    sys.exit(main())
/**
 * Accessibility enhancements for CAIS documentation
 * Implements WCAG 2.1 AA compliance features
 */

(function() {
    'use strict';

    // Accessibility configuration
    const A11Y_CONFIG = {
        skipLinkText: 'Skip to main content',
        ariaLiveRegion: 'aria-live-region',
        focusableElements: 'a[href], button, input, textarea, select, details, [tabindex]:not([tabindex="-1"])',
        headingLevels: ['h1', 'h2', 'h3', 'h4', 'h5', 'h6']
    };

    let currentFocusIndex = -1;
    let focusableElements = [];

    /**
     * Initialize accessibility enhancements
     */
    function initializeAccessibility() {
        addSkipToContentLink();
        addAriaLiveRegion();
        enhanceKeyboardNavigation();
        improveScreenReaderSupport();
        addFocusManagement();
        enhanceFormAccessibility();
        addLandmarkRoles();
        improveColorContrast();
        addAccessibilityControls();
        validateAccessibility();
    }

    /**
     * Add skip to content link for keyboard users
     */
    function addSkipToContentLink() {
        const skipLink = document.createElement('a');
        skipLink.href = '#main-content';
        skipLink.className = 'skip-to-content';
        skipLink.textContent = A11Y_CONFIG.skipLinkText;
        skipLink.setAttribute('aria-label', 'Skip navigation and go to main content');
        
        // Insert at the beginning of the body
        document.body.insertBefore(skipLink, document.body.firstChild);
        
        // Ensure main content has proper ID
        let mainContent = document.getElementById('main-content');
        if (!mainContent) {
            mainContent = document.querySelector('.wy-nav-content') || 
                         document.querySelector('main') || 
                         document.querySelector('[role="main"]');
            if (mainContent) {
                mainContent.id = 'main-content';
                mainContent.setAttribute('tabindex', '-1');
            }
        }
    }

    /**
     * Add ARIA live region for dynamic content announcements
     */
    function addAriaLiveRegion() {
        const liveRegion = document.createElement('div');
        liveRegion.id = A11Y_CONFIG.ariaLiveRegion;
        liveRegion.setAttribute('aria-live', 'polite');
        liveRegion.setAttribute('aria-atomic', 'true');
        liveRegion.className = 'sr-only';
        document.body.appendChild(liveRegion);
    }

    /**
     * Announce messages to screen readers
     */
    function announceToScreenReader(message, priority = 'polite') {
        const liveRegion = document.getElementById(A11Y_CONFIG.ariaLiveRegion);
        if (liveRegion) {
            liveRegion.setAttribute('aria-live', priority);
            liveRegion.textContent = message;
            
            // Clear after announcement
            setTimeout(() => {
                liveRegion.textContent = '';
            }, 1000);
        }
    }

    /**
     * Enhance keyboard navigation throughout the site
     */
    function enhanceKeyboardNavigation() {
        // Update focusable elements list
        updateFocusableElements();
        
        // Add keyboard event listeners
        document.addEventListener('keydown', handleKeyboardNavigation);
        
        // Update focusable elements when DOM changes
        const observer = new MutationObserver(updateFocusableElements);
        observer.observe(document.body, {
            childList: true,
            subtree: true,
            attributes: true,
            attributeFilter: ['tabindex', 'disabled', 'hidden']
        });
    }

    /**
     * Handle keyboard navigation events
     */
    function handleKeyboardNavigation(e) {
        switch(e.key) {
            case 'Tab':
                handleTabNavigation(e);
                break;
            case 'Escape':
                handleEscapeKey(e);
                break;
            case 'Enter':
            case ' ':
                handleActivation(e);
                break;
            case 'ArrowDown':
            case 'ArrowUp':
                handleArrowNavigation(e);
                break;
            case 'Home':
            case 'End':
                handleHomeEndNavigation(e);
                break;
        }
    }

    /**
     * Handle Tab navigation with proper focus management
     */
    function handleTabNavigation(e) {
        const activeElement = document.activeElement;
        
        // Handle modal or dialog focus trapping
        const modal = activeElement.closest('[role="dialog"], .modal');
        if (modal) {
            trapFocusInModal(e, modal);
        }
        
        // Announce current focus to screen readers
        if (activeElement && activeElement.getAttribute('aria-label')) {
            announceToScreenReader(`Focused on ${activeElement.getAttribute('aria-label')}`);
        }
    }

    /**
     * Handle Escape key for closing modals and menus
     */
    function handleEscapeKey(e) {
        // Close any open modals
        const openModal = document.querySelector('[role="dialog"][aria-hidden="false"], .modal:not(.hidden)');
        if (openModal) {
            closeModal(openModal);
            e.preventDefault();
        }
        
        // Close any open menus
        const openMenu = document.querySelector('[role="menu"][aria-expanded="true"]');
        if (openMenu) {
            closeMenu(openMenu);
            e.preventDefault();
        }
        
        // Clear search suggestions
        const searchSuggestions = document.getElementById('search-suggestions');
        if (searchSuggestions && !searchSuggestions.classList.contains('hidden')) {
            searchSuggestions.classList.add('hidden');
            e.preventDefault();
        }
    }

    /**
     * Handle Enter and Space key activation
     */
    function handleActivation(e) {
        const target = e.target;
        
        // Handle custom button-like elements
        if (target.getAttribute('role') === 'button' && !target.disabled) {
            target.click();
            e.preventDefault();
        }
        
        // Handle expandable sections
        if (target.classList.contains('expandable-header')) {
            toggleExpandableSection(target);
            e.preventDefault();
        }
    }

    /**
     * Handle arrow key navigation in lists and menus
     */
    function handleArrowNavigation(e) {
        const target = e.target;
        const parent = target.closest('[role="menu"], [role="listbox"], .search-results-list');
        
        if (parent) {
            const items = parent.querySelectorAll('[role="menuitem"], [role="option"], .search-result-item');
            const currentIndex = Array.from(items).indexOf(target);
            
            let nextIndex;
            if (e.key === 'ArrowDown') {
                nextIndex = (currentIndex + 1) % items.length;
            } else {
                nextIndex = (currentIndex - 1 + items.length) % items.length;
            }
            
            items[nextIndex].focus();
            e.preventDefault();
        }
    }

    /**
     * Handle Home and End key navigation
     */
    function handleHomeEndNavigation(e) {
        const target = e.target;
        const parent = target.closest('[role="menu"], [role="listbox"], .search-results-list');
        
        if (parent) {
            const items = parent.querySelectorAll('[role="menuitem"], [role="option"], .search-result-item');
            
            if (e.key === 'Home') {
                items[0].focus();
            } else {
                items[items.length - 1].focus();
            }
            e.preventDefault();
        }
    }

    /**
     * Improve screen reader support with proper ARIA labels and descriptions
     */
    function improveScreenReaderSupport() {
        // Add ARIA labels to navigation elements
        const nav = document.querySelector('.wy-nav-side');
        if (nav) {
            nav.setAttribute('role', 'navigation');
            nav.setAttribute('aria-label', 'Main navigation');
        }
        
        // Add ARIA labels to search
        const searchForm = document.getElementById('rtd-search-form');
        if (searchForm) {
            const searchInput = searchForm.querySelector('input[name="q"]');
            if (searchInput) {
                searchInput.setAttribute('aria-label', 'Search documentation');
                searchInput.setAttribute('aria-describedby', 'search-help');
                
                // Add search help text
                const searchHelp = document.createElement('div');
                searchHelp.id = 'search-help';
                searchHelp.className = 'sr-only';
                searchHelp.textContent = 'Search through the CAIS documentation. Use arrow keys to navigate results.';
                searchForm.appendChild(searchHelp);
            }
        }
        
        // Improve heading structure
        improveHeadingStructure();
        
        // Add ARIA labels to code blocks
        document.querySelectorAll('.highlight, .literal-block').forEach((codeBlock, index) => {
            codeBlock.setAttribute('role', 'region');
            codeBlock.setAttribute('aria-label', `Code example ${index + 1}`);
        });
        
        // Add ARIA labels to tables
        document.querySelectorAll('table').forEach((table, index) => {
            if (!table.querySelector('caption')) {
                table.setAttribute('aria-label', `Data table ${index + 1}`);
            }
        });
    }

    /**
     * Improve heading structure for screen readers
     */
    function improveHeadingStructure() {
        const headings = document.querySelectorAll(A11Y_CONFIG.headingLevels.join(', '));
        let previousLevel = 0;
        
        headings.forEach(heading => {
            const level = parseInt(heading.tagName.charAt(1));
            
            // Add ARIA attributes for better navigation
            heading.setAttribute('tabindex', '-1');
            
            // Add heading navigation
            if (level > previousLevel + 1) {
                console.warn(`Heading level skipped: ${heading.textContent} (h${level} after h${previousLevel})`);
            }
            
            previousLevel = level;
        });
    }

    /**
     * Add comprehensive focus management
     */
    function addFocusManagement() {
        // Track focus for better UX
        let focusVisible = false;
        
        document.addEventListener('keydown', (e) => {
            if (e.key === 'Tab') {
                focusVisible = true;
            }
        });
        
        document.addEventListener('mousedown', () => {
            focusVisible = false;
        });
        
        document.addEventListener('focusin', (e) => {
            if (focusVisible) {
                e.target.classList.add('focus-visible');
            }
        });
        
        document.addEventListener('focusout', (e) => {
            e.target.classList.remove('focus-visible');
        });
        
        // Manage focus for dynamic content
        const observer = new MutationObserver((mutations) => {
            mutations.forEach((mutation) => {
                if (mutation.type === 'childList') {
                    mutation.addedNodes.forEach((node) => {
                        if (node.nodeType === Node.ELEMENT_NODE) {
                            manageFocusForNewContent(node);
                        }
                    });
                }
            });
        });
        
        observer.observe(document.body, {
            childList: true,
            subtree: true
        });
    }

    /**
     * Manage focus for newly added content
     */
    function manageFocusForNewContent(element) {
        // If it's a modal or important content, focus it
        if (element.matches('[role="dialog"], .modal, .alert')) {
            const focusTarget = element.querySelector('[autofocus]') || 
                              element.querySelector('button, input, textarea, select') ||
                              element;
            
            setTimeout(() => {
                focusTarget.focus();
                announceToScreenReader(`${element.getAttribute('aria-label') || 'New content'} opened`);
            }, 100);
        }
        
        // Update focusable elements
        updateFocusableElements();
    }

    /**
     * Enhance form accessibility
     */
    function enhanceFormAccessibility() {
        // Add proper labels and descriptions to form elements
        document.querySelectorAll('input, textarea, select').forEach(input => {
            // Ensure all inputs have labels
            if (!input.getAttribute('aria-label') && !input.getAttribute('aria-labelledby')) {
                const label = document.querySelector(`label[for="${input.id}"]`);
                if (!label && input.placeholder) {
                    input.setAttribute('aria-label', input.placeholder);
                }
            }
            
            // Add required field indicators
            if (input.required) {
                input.setAttribute('aria-required', 'true');
                
                // Add visual indicator
                const label = document.querySelector(`label[for="${input.id}"]`);
                if (label && !label.querySelector('.required-indicator')) {
                    const indicator = document.createElement('span');
                    indicator.className = 'required-indicator';
                    indicator.textContent = ' *';
                    indicator.setAttribute('aria-label', 'required');
                    label.appendChild(indicator);
                }
            }
            
            // Add error handling
            input.addEventListener('invalid', (e) => {
                const errorMessage = e.target.validationMessage;
                announceToScreenReader(`Error: ${errorMessage}`, 'assertive');
            });
        });
        
        // Enhance feedback forms
        const feedbackForms = document.querySelectorAll('.feedback-form, .feedback-modal-body');
        feedbackForms.forEach(form => {
            form.setAttribute('role', 'form');
            form.setAttribute('aria-label', 'Documentation feedback form');
        });
    }

    /**
     * Add proper landmark roles for better navigation
     */
    function addLandmarkRoles() {
        // Main content
        const main = document.querySelector('.wy-nav-content, main');
        if (main && !main.getAttribute('role')) {
            main.setAttribute('role', 'main');
        }
        
        // Navigation
        const nav = document.querySelector('.wy-nav-side');
        if (nav && !nav.getAttribute('role')) {
            nav.setAttribute('role', 'navigation');
            nav.setAttribute('aria-label', 'Main navigation');
        }
        
        // Search
        const search = document.getElementById('rtd-search-form');
        if (search && !search.getAttribute('role')) {
            search.setAttribute('role', 'search');
        }
        
        // Breadcrumbs
        const breadcrumbs = document.querySelector('.wy-breadcrumbs');
        if (breadcrumbs) {
            breadcrumbs.setAttribute('role', 'navigation');
            breadcrumbs.setAttribute('aria-label', 'Breadcrumb navigation');
        }
        
        // Footer
        const footer = document.querySelector('.wy-nav-content footer, footer');
        if (footer && !footer.getAttribute('role')) {
            footer.setAttribute('role', 'contentinfo');
        }
    }

    /**
     * Improve color contrast and visual accessibility
     */
    function improveColorContrast() {
        // Check for user preferences
        const prefersHighContrast = window.matchMedia('(prefers-contrast: high)').matches;
        const prefersDarkMode = window.matchMedia('(prefers-color-scheme: dark)').matches;
        
        if (prefersHighContrast) {
            document.body.classList.add('high-contrast');
        }
        
        if (prefersDarkMode) {
            document.body.classList.add('dark-mode');
        }
        
        // Listen for changes in user preferences
        window.matchMedia('(prefers-contrast: high)').addEventListener('change', (e) => {
            document.body.classList.toggle('high-contrast', e.matches);
        });
        
        window.matchMedia('(prefers-color-scheme: dark)').addEventListener('change', (e) => {
            document.body.classList.toggle('dark-mode', e.matches);
        });
    }

    /**
     * Add accessibility control panel
     */
    function addAccessibilityControls() {
        const controlPanel = document.createElement('div');
        controlPanel.className = 'accessibility-controls';
        controlPanel.innerHTML = `
            <button class="a11y-toggle" aria-label="Accessibility options" aria-expanded="false">
                <span class="sr-only">Accessibility</span>
                ♿
            </button>
            <div class="a11y-panel hidden" role="region" aria-label="Accessibility controls">
                <h3>Accessibility Options</h3>
                <div class="a11y-options">
                    <label>
                        <input type="checkbox" id="high-contrast-toggle"> High Contrast
                    </label>
                    <label>
                        <input type="checkbox" id="large-text-toggle"> Large Text
                    </label>
                    <label>
                        <input type="checkbox" id="reduce-motion-toggle"> Reduce Motion
                    </label>
                    <label>
                        <input type="range" id="font-size-slider" min="12" max="24" value="16" aria-label="Font size">
                        Font Size: <span id="font-size-value">16px</span>
                    </label>
                </div>
            </div>
        `;
        
        document.body.appendChild(controlPanel);
        
        // Add event listeners for accessibility controls
        const toggle = controlPanel.querySelector('.a11y-toggle');
        const panel = controlPanel.querySelector('.a11y-panel');
        
        toggle.addEventListener('click', () => {
            const isExpanded = toggle.getAttribute('aria-expanded') === 'true';
            toggle.setAttribute('aria-expanded', !isExpanded);
            panel.classList.toggle('hidden');
        });
        
        // High contrast toggle
        const highContrastToggle = document.getElementById('high-contrast-toggle');
        highContrastToggle.addEventListener('change', (e) => {
            document.body.classList.toggle('high-contrast', e.target.checked);
            localStorage.setItem('high-contrast', e.target.checked);
        });
        
        // Large text toggle
        const largeTextToggle = document.getElementById('large-text-toggle');
        largeTextToggle.addEventListener('change', (e) => {
            document.body.classList.toggle('large-text', e.target.checked);
            localStorage.setItem('large-text', e.target.checked);
        });
        
        // Reduce motion toggle
        const reduceMotionToggle = document.getElementById('reduce-motion-toggle');
        reduceMotionToggle.addEventListener('change', (e) => {
            document.body.classList.toggle('reduce-motion', e.target.checked);
            localStorage.setItem('reduce-motion', e.target.checked);
        });
        
        // Font size slider
        const fontSizeSlider = document.getElementById('font-size-slider');
        const fontSizeValue = document.getElementById('font-size-value');
        
        fontSizeSlider.addEventListener('input', (e) => {
            const fontSize = e.target.value + 'px';
            document.documentElement.style.setProperty('--base-font-size', fontSize);
            fontSizeValue.textContent = fontSize;
            localStorage.setItem('font-size', fontSize);
        });
        
        // Load saved preferences
        loadAccessibilityPreferences();
    }

    /**
     * Load saved accessibility preferences
     */
    function loadAccessibilityPreferences() {
        const highContrast = localStorage.getItem('high-contrast') === 'true';
        const largeText = localStorage.getItem('large-text') === 'true';
        const reduceMotion = localStorage.getItem('reduce-motion') === 'true';
        const fontSize = localStorage.getItem('font-size') || '16px';
        
        if (highContrast) {
            document.body.classList.add('high-contrast');
            const toggle = document.getElementById('high-contrast-toggle');
            if (toggle) toggle.checked = true;
        }
        
        if (largeText) {
            document.body.classList.add('large-text');
            const toggle = document.getElementById('large-text-toggle');
            if (toggle) toggle.checked = true;
        }
        
        if (reduceMotion) {
            document.body.classList.add('reduce-motion');
            const toggle = document.getElementById('reduce-motion-toggle');
            if (toggle) toggle.checked = true;
        }
        
        document.documentElement.style.setProperty('--base-font-size', fontSize);
        const slider = document.getElementById('font-size-slider');
        const value = document.getElementById('font-size-value');
        if (slider && value) {
            slider.value = parseInt(fontSize);
            value.textContent = fontSize;
        }
    }

    /**
     * Validate accessibility compliance
     */
    function validateAccessibility() {
        const issues = [];
        
        // Check for images without alt text
        document.querySelectorAll('img:not([alt])').forEach(img => {
            issues.push(`Image missing alt text: ${img.src}`);
        });
        
        // Check for form inputs without labels
        document.querySelectorAll('input:not([aria-label]):not([aria-labelledby])').forEach(input => {
            if (!document.querySelector(`label[for="${input.id}"]`)) {
                issues.push(`Input missing label: ${input.name || input.type}`);
            }
        });
        
        // Check for proper heading hierarchy
        const headings = document.querySelectorAll('h1, h2, h3, h4, h5, h6');
        let previousLevel = 0;
        headings.forEach(heading => {
            const level = parseInt(heading.tagName.charAt(1));
            if (level > previousLevel + 1) {
                issues.push(`Heading level skipped: ${heading.textContent}`);
            }
            previousLevel = level;
        });
        
        // Check for sufficient color contrast (basic check)
        const lowContrastElements = document.querySelectorAll('.low-contrast');
        lowContrastElements.forEach(element => {
            issues.push(`Potential low contrast element: ${element.textContent.substring(0, 50)}`);
        });
        
        // Log issues in development
        if (issues.length > 0 && window.location.hostname === 'localhost') {
            console.group('Accessibility Issues Found:');
            issues.forEach(issue => console.warn(issue));
            console.groupEnd();
        }
        
        return issues;
    }

    /**
     * Update list of focusable elements
     */
    function updateFocusableElements() {
        focusableElements = Array.from(document.querySelectorAll(A11Y_CONFIG.focusableElements))
            .filter(el => !el.disabled && !el.hidden && el.offsetParent !== null);
    }

    /**
     * Trap focus within a modal
     */
    function trapFocusInModal(e, modal) {
        const focusableInModal = modal.querySelectorAll(A11Y_CONFIG.focusableElements);
        const firstFocusable = focusableInModal[0];
        const lastFocusable = focusableInModal[focusableInModal.length - 1];
        
        if (e.shiftKey) {
            if (document.activeElement === firstFocusable) {
                lastFocusable.focus();
                e.preventDefault();
            }
        } else {
            if (document.activeElement === lastFocusable) {
                firstFocusable.focus();
                e.preventDefault();
            }
        }
    }

    /**
     * Close modal and return focus
     */
    function closeModal(modal) {
        modal.setAttribute('aria-hidden', 'true');
        modal.classList.add('hidden');
        
        // Return focus to trigger element
        const trigger = document.querySelector(`[aria-controls="${modal.id}"]`);
        if (trigger) {
            trigger.focus();
        }
        
        announceToScreenReader('Modal closed');
    }

    /**
     * Close menu and return focus
     */
    function closeMenu(menu) {
        menu.setAttribute('aria-expanded', 'false');
        
        // Return focus to trigger element
        const trigger = document.querySelector(`[aria-controls="${menu.id}"]`);
        if (trigger) {
            trigger.focus();
        }
    }

    /**
     * Toggle expandable section
     */
    function toggleExpandableSection(header) {
        const section = header.nextElementSibling;
        const isExpanded = header.getAttribute('aria-expanded') === 'true';
        
        header.setAttribute('aria-expanded', !isExpanded);
        section.classList.toggle('hidden');
        
        announceToScreenReader(`Section ${isExpanded ? 'collapsed' : 'expanded'}`);
    }

    // Global accessibility utilities
    window.a11y = {
        announce: announceToScreenReader,
        validateAccessibility: validateAccessibility,
        updateFocusableElements: updateFocusableElements
    };

    // Initialize when DOM is ready
    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', initializeAccessibility);
    } else {
        initializeAccessibility();
    }

})();
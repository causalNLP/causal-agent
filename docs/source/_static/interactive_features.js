/**
 * Interactive Features for CAIS Documentation
 * Provides copy-to-clipboard, feedback collection, tooltips, and expandable sections
 */

(function() {
    'use strict';

    // Configuration
    const CONFIG = {
        copyButtonClass: 'copy-code-btn',
        tooltipClass: 'tooltip',
        expandableClass: 'expandable-section',
        feedbackClass: 'feedback-widget',
        interactiveExampleClass: 'interactive-example',
        animationDuration: 300
    };

    // Feedback API endpoint (mock for now)
    const FEEDBACK_API = '/api/feedback';

    /**
     * Initialize all interactive features
     */
    function initializeInteractiveFeatures() {
        try {
            // Prevent Mermaid conflicts by ensuring our content doesn't interfere
            preventMermaidConflicts();
            
            initializeCopyToClipboard();
            initializeFeedbackSystem();
            initializeTooltips();
            initializeExpandableSections();
            initializeInteractiveExamples();
            initializeAccessibilityFeatures();
        } catch (error) {
            console.error('Error initializing interactive features:', error);
        }
    }

    /**
     * Prevent conflicts with Mermaid diagrams
     */
    function preventMermaidConflicts() {
        // Mark our interactive elements to prevent Mermaid processing
        const interactiveElements = document.querySelectorAll('.interactive-example, .interactive-decision-tree');
        interactiveElements.forEach(element => {
            element.setAttribute('data-no-mermaid', 'true');
        });
    }

    /**
     * Add copy-to-clipboard functionality for code examples
     */
    function initializeCopyToClipboard() {
        // Find all code blocks and add copy buttons
        const codeBlocks = document.querySelectorAll('pre, .highlight, .literal-block, .code-block');
        
        codeBlocks.forEach(block => {
            if (block.querySelector(`.${CONFIG.copyButtonClass}`)) return; // Already has button
            
            const copyButton = createCopyButton();
            const container = createCodeContainer(block, copyButton);
            
            copyButton.addEventListener('click', () => copyCodeToClipboard(block, copyButton));
        });

        // Handle notebook code cells specifically
        const notebookCells = document.querySelectorAll('.nbinput .highlight, .nboutput .highlight');
        notebookCells.forEach(cell => {
            if (cell.querySelector(`.${CONFIG.copyButtonClass}`)) return;
            
            const copyButton = createCopyButton('Copy cell');
            const container = createCodeContainer(cell, copyButton);
            
            copyButton.addEventListener('click', () => copyCodeToClipboard(cell, copyButton));
        });
    }

    /**
     * Create a copy button element
     */
    function createCopyButton(text = 'Copy code') {
        const button = document.createElement('button');
        button.className = `${CONFIG.copyButtonClass} btn-copy`;
        button.innerHTML = `
            <svg class="copy-icon" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                <rect x="9" y="9" width="13" height="13" rx="2" ry="2"></rect>
                <path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"></path>
            </svg>
            <span class="copy-text">${text}</span>
        `;
        button.setAttribute('aria-label', text);
        button.setAttribute('title', text);
        return button;
    }

    /**
     * Create a container for code block with copy button
     */
    function createCodeContainer(codeBlock, copyButton) {
        // Check if already wrapped
        if (codeBlock.parentElement.classList.contains('code-container')) {
            return codeBlock.parentElement;
        }

        const container = document.createElement('div');
        container.className = 'code-container';
        
        const header = document.createElement('div');
        header.className = 'code-header';
        header.appendChild(copyButton);
        
        // Wrap the code block
        codeBlock.parentNode.insertBefore(container, codeBlock);
        container.appendChild(header);
        container.appendChild(codeBlock);
        
        return container;
    }

    /**
     * Copy code content to clipboard
     */
    async function copyCodeToClipboard(codeBlock, button) {
        try {
            // Extract text content, handling different code block types
            let textToCopy = '';
            
            if (codeBlock.querySelector('code')) {
                textToCopy = codeBlock.querySelector('code').textContent;
            } else {
                textToCopy = codeBlock.textContent;
            }
            
            // Clean up the text (remove line numbers, prompts, etc.)
            textToCopy = cleanCodeText(textToCopy);
            
            await navigator.clipboard.writeText(textToCopy);
            
            // Show success feedback
            showCopySuccess(button);
            
            // Track copy event
            trackEvent('code_copy', {
                'code_length': textToCopy.length,
                'code_type': getCodeType(codeBlock)
            });
            
        } catch (err) {
            console.error('Failed to copy code:', err);
            showCopyError(button);
        }
    }

    /**
     * Clean code text for copying
     */
    function cleanCodeText(text) {
        return text
            .replace(/^In \[\d+\]:\s*/gm, '') // Remove Jupyter input prompts
            .replace(/^Out\[\d+\]:\s*/gm, '') // Remove Jupyter output prompts
            .replace(/^\s*\d+\s+/gm, '') // Remove line numbers
            .replace(/^\s*>>>\s*/gm, '') // Remove Python prompts
            .replace(/^\s*\.\.\.\s*/gm, '') // Remove Python continuation prompts
            .trim();
    }

    /**
     * Determine code type for analytics
     */
    function getCodeType(codeBlock) {
        const classes = codeBlock.className;
        if (classes.includes('python')) return 'python';
        if (classes.includes('bash')) return 'bash';
        if (classes.includes('json')) return 'json';
        if (classes.includes('yaml')) return 'yaml';
        if (codeBlock.closest('.nbinput')) return 'jupyter_input';
        if (codeBlock.closest('.nboutput')) return 'jupyter_output';
        return 'unknown';
    }

    /**
     * Show copy success feedback
     */
    function showCopySuccess(button) {
        const originalContent = button.innerHTML;
        button.innerHTML = `
            <svg class="check-icon" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                <polyline points="20,6 9,17 4,12"></polyline>
            </svg>
            <span class="copy-text">Copied!</span>
        `;
        button.classList.add('copy-success');
        
        setTimeout(() => {
            button.innerHTML = originalContent;
            button.classList.remove('copy-success');
        }, 2000);
    }

    /**
     * Show copy error feedback
     */
    function showCopyError(button) {
        const originalContent = button.innerHTML;
        button.innerHTML = `
            <svg class="error-icon" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                <circle cx="12" cy="12" r="10"></circle>
                <line x1="15" y1="9" x2="9" y2="15"></line>
                <line x1="9" y1="9" x2="15" y2="15"></line>
            </svg>
            <span class="copy-text">Failed</span>
        `;
        button.classList.add('copy-error');
        
        setTimeout(() => {
            button.innerHTML = originalContent;
            button.classList.remove('copy-error');
        }, 2000);
    }

    /**
     * Initialize feedback collection system
     */
    function initializeFeedbackSystem() {
        createFeedbackWidget();
        initializePageFeedback();
        initializeContentFeedback();
    }

    /**
     * Create the main feedback widget
     */
    function createFeedbackWidget() {
        const widget = document.createElement('div');
        widget.id = 'feedback-widget';
        widget.className = CONFIG.feedbackClass;
        widget.innerHTML = `
            <div class="feedback-trigger">
                <button class="feedback-btn" aria-label="Provide feedback">
                    <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                        <path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z"></path>
                    </svg>
                    Feedback
                </button>
            </div>
            <div class="feedback-panel hidden">
                <div class="feedback-header">
                    <h3>Help us improve this page</h3>
                    <button class="feedback-close" aria-label="Close feedback">×</button>
                </div>
                <div class="feedback-content">
                    <div class="feedback-rating">
                        <p>How helpful was this page?</p>
                        <div class="rating-buttons">
                            <button class="rating-btn" data-rating="1" aria-label="Not helpful">😞</button>
                            <button class="rating-btn" data-rating="2" aria-label="Somewhat helpful">😐</button>
                            <button class="rating-btn" data-rating="3" aria-label="Helpful">🙂</button>
                            <button class="rating-btn" data-rating="4" aria-label="Very helpful">😊</button>
                            <button class="rating-btn" data-rating="5" aria-label="Extremely helpful">🤩</button>
                        </div>
                    </div>
                    <div class="feedback-form hidden">
                        <textarea class="feedback-text" placeholder="Tell us more about your experience (optional)..." rows="4"></textarea>
                        <div class="feedback-categories">
                            <label><input type="checkbox" value="accuracy"> Information accuracy</label>
                            <label><input type="checkbox" value="clarity"> Clarity of explanation</label>
                            <label><input type="checkbox" value="examples"> Code examples</label>
                            <label><input type="checkbox" value="navigation"> Navigation/structure</label>
                            <label><input type="checkbox" value="missing"> Missing information</label>
                        </div>
                        <div class="feedback-actions">
                            <button class="feedback-submit">Submit Feedback</button>
                            <button class="feedback-cancel">Cancel</button>
                        </div>
                    </div>
                    <div class="feedback-thanks hidden">
                        <p>Thank you for your feedback! 🙏</p>
                        <p>Your input helps us improve the documentation.</p>
                    </div>
                </div>
            </div>
        `;

        document.body.appendChild(widget);
        initializeFeedbackEvents(widget);
    }

    /**
     * Initialize feedback widget events
     */
    function initializeFeedbackEvents(widget) {
        const trigger = widget.querySelector('.feedback-btn');
        const panel = widget.querySelector('.feedback-panel');
        const closeBtn = widget.querySelector('.feedback-close');
        const ratingBtns = widget.querySelectorAll('.rating-btn');
        const submitBtn = widget.querySelector('.feedback-submit');
        const cancelBtn = widget.querySelector('.feedback-cancel');

        trigger.addEventListener('click', () => toggleFeedbackPanel(panel));
        closeBtn.addEventListener('click', () => closeFeedbackPanel(panel));
        
        ratingBtns.forEach(btn => {
            btn.addEventListener('click', () => handleRatingClick(btn, widget));
        });
        
        submitBtn.addEventListener('click', () => submitFeedback(widget));
        cancelBtn.addEventListener('click', () => closeFeedbackPanel(panel));

        // Close on outside click
        document.addEventListener('click', (e) => {
            if (!widget.contains(e.target)) {
                closeFeedbackPanel(panel);
            }
        });
    }

    /**
     * Toggle feedback panel visibility
     */
    function toggleFeedbackPanel(panel) {
        panel.classList.toggle('hidden');
        if (!panel.classList.contains('hidden')) {
            panel.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
        }
    }

    /**
     * Close feedback panel
     */
    function closeFeedbackPanel(panel) {
        panel.classList.add('hidden');
        // Reset form state
        setTimeout(() => {
            panel.querySelector('.feedback-form').classList.add('hidden');
            panel.querySelector('.feedback-rating').classList.remove('hidden');
            panel.querySelector('.feedback-thanks').classList.add('hidden');
            panel.querySelectorAll('.rating-btn').forEach(btn => btn.classList.remove('selected'));
        }, 300);
    }

    /**
     * Handle rating button click
     */
    function handleRatingClick(button, widget) {
        const rating = parseInt(button.dataset.rating);
        
        // Update UI
        widget.querySelectorAll('.rating-btn').forEach(btn => btn.classList.remove('selected'));
        button.classList.add('selected');
        
        // Show form for detailed feedback
        setTimeout(() => {
            widget.querySelector('.feedback-rating').classList.add('hidden');
            widget.querySelector('.feedback-form').classList.remove('hidden');
        }, 500);
        
        // Store rating
        widget.dataset.rating = rating;
    }

    /**
     * Submit feedback
     */
    async function submitFeedback(widget) {
        const rating = parseInt(widget.dataset.rating);
        const text = widget.querySelector('.feedback-text').value;
        const categories = Array.from(widget.querySelectorAll('.feedback-categories input:checked'))
            .map(input => input.value);
        
        const feedbackData = {
            page: window.location.pathname,
            rating: rating,
            comment: text,
            categories: categories,
            timestamp: new Date().toISOString(),
            userAgent: navigator.userAgent
        };

        try {
            // In a real implementation, this would send to your feedback API
            console.log('Feedback submitted:', feedbackData);
            
            // Track feedback event
            trackEvent('feedback_submitted', {
                'rating': rating,
                'has_comment': text.length > 0,
                'categories_count': categories.length
            });
            
            // Show thanks message
            widget.querySelector('.feedback-form').classList.add('hidden');
            widget.querySelector('.feedback-thanks').classList.remove('hidden');
            
            // Auto-close after delay
            setTimeout(() => {
                closeFeedbackPanel(widget.querySelector('.feedback-panel'));
            }, 3000);
            
        } catch (error) {
            console.error('Failed to submit feedback:', error);
            alert('Failed to submit feedback. Please try again.');
        }
    }

    /**
     * Initialize page-level feedback
     */
    function initializePageFeedback() {
        // Add quick feedback buttons to page bottom
        const pageContent = document.querySelector('.wy-nav-content-wrap');
        if (!pageContent) return;

        const quickFeedback = document.createElement('div');
        quickFeedback.className = 'quick-feedback';
        quickFeedback.innerHTML = `
            <div class="quick-feedback-question">
                Was this page helpful?
            </div>
            <div class="quick-feedback-buttons">
                <button class="quick-yes" data-helpful="true">👍 Yes</button>
                <button class="quick-no" data-helpful="false">👎 No</button>
            </div>
            <div class="quick-feedback-thanks hidden">
                Thanks for your feedback!
            </div>
        `;

        pageContent.appendChild(quickFeedback);

        // Handle quick feedback
        quickFeedback.addEventListener('click', (e) => {
            if (e.target.matches('.quick-yes, .quick-no')) {
                const helpful = e.target.dataset.helpful === 'true';
                
                trackEvent('quick_feedback', {
                    'helpful': helpful,
                    'page': window.location.pathname
                });
                
                quickFeedback.querySelector('.quick-feedback-buttons').classList.add('hidden');
                quickFeedback.querySelector('.quick-feedback-thanks').classList.remove('hidden');
            }
        });
    }

    /**
     * Initialize content-specific feedback
     */
    function initializeContentFeedback() {
        // Add feedback buttons to specific content sections
        const sections = document.querySelectorAll('h2, h3');
        sections.forEach(section => {
            if (section.id) {
                const feedbackBtn = document.createElement('button');
                feedbackBtn.className = 'section-feedback-btn';
                feedbackBtn.innerHTML = '💬';
                feedbackBtn.title = 'Feedback on this section';
                feedbackBtn.setAttribute('aria-label', 'Provide feedback on this section');
                
                feedbackBtn.addEventListener('click', () => {
                    openSectionFeedback(section.id, section.textContent);
                });
                
                section.appendChild(feedbackBtn);
            }
        });
    }

    /**
     * Open section-specific feedback
     */
    function openSectionFeedback(sectionId, sectionTitle) {
        const modal = createFeedbackModal(sectionId, sectionTitle);
        document.body.appendChild(modal);
        
        // Focus management
        const firstInput = modal.querySelector('textarea');
        if (firstInput) firstInput.focus();
    }

    /**
     * Create feedback modal for sections
     */
    function createFeedbackModal(sectionId, sectionTitle) {
        const modal = document.createElement('div');
        modal.className = 'feedback-modal';
        modal.innerHTML = `
            <div class="feedback-modal-overlay"></div>
            <div class="feedback-modal-content">
                <div class="feedback-modal-header">
                    <h3>Feedback: ${sectionTitle}</h3>
                    <button class="feedback-modal-close" aria-label="Close">×</button>
                </div>
                <div class="feedback-modal-body">
                    <textarea placeholder="What would you like to tell us about this section?" rows="4"></textarea>
                    <div class="feedback-modal-actions">
                        <button class="feedback-modal-submit">Send Feedback</button>
                        <button class="feedback-modal-cancel">Cancel</button>
                    </div>
                </div>
            </div>
        `;

        // Event listeners
        modal.querySelector('.feedback-modal-close').addEventListener('click', () => {
            document.body.removeChild(modal);
        });
        
        modal.querySelector('.feedback-modal-cancel').addEventListener('click', () => {
            document.body.removeChild(modal);
        });
        
        modal.querySelector('.feedback-modal-submit').addEventListener('click', () => {
            const feedback = modal.querySelector('textarea').value;
            if (feedback.trim()) {
                trackEvent('section_feedback', {
                    'section_id': sectionId,
                    'feedback_length': feedback.length
                });
                alert('Thank you for your feedback!');
            }
            document.body.removeChild(modal);
        });
        
        modal.querySelector('.feedback-modal-overlay').addEventListener('click', () => {
            document.body.removeChild(modal);
        });

        return modal;
    }

    /**
     * Initialize tooltips for complex concepts
     */
    function initializeTooltips() {
        // Find elements with tooltip data
        const tooltipElements = document.querySelectorAll('[data-tooltip]');
        
        tooltipElements.forEach(element => {
            createTooltip(element);
        });

        // Add tooltips to technical terms
        addTechnicalTermTooltips();
        
        // Add tooltips to method names
        addMethodTooltips();
    }

    /**
     * Create tooltip for an element
     */
    function createTooltip(element) {
        const tooltipText = element.dataset.tooltip;
        const tooltipType = element.dataset.tooltipType || 'info';
        
        let tooltip = null;
        
        element.addEventListener('mouseenter', (e) => {
            tooltip = showTooltip(e.target, tooltipText, tooltipType);
        });
        
        element.addEventListener('mouseleave', () => {
            if (tooltip) {
                hideTooltip(tooltip);
                tooltip = null;
            }
        });
        
        element.addEventListener('focus', (e) => {
            tooltip = showTooltip(e.target, tooltipText, tooltipType);
        });
        
        element.addEventListener('blur', () => {
            if (tooltip) {
                hideTooltip(tooltip);
                tooltip = null;
            }
        });
    }

    /**
     * Show tooltip
     */
    function showTooltip(element, text, type = 'info') {
        const tooltip = document.createElement('div');
        tooltip.className = `${CONFIG.tooltipClass} tooltip-${type}`;
        tooltip.innerHTML = text;
        tooltip.setAttribute('role', 'tooltip');
        
        document.body.appendChild(tooltip);
        
        // Position tooltip
        const rect = element.getBoundingClientRect();
        const tooltipRect = tooltip.getBoundingClientRect();
        
        let left = rect.left + (rect.width / 2) - (tooltipRect.width / 2);
        let top = rect.top - tooltipRect.height - 10;
        
        // Adjust if tooltip goes off screen
        if (left < 10) left = 10;
        if (left + tooltipRect.width > window.innerWidth - 10) {
            left = window.innerWidth - tooltipRect.width - 10;
        }
        if (top < 10) {
            top = rect.bottom + 10;
            tooltip.classList.add('tooltip-below');
        }
        
        tooltip.style.left = `${left}px`;
        tooltip.style.top = `${top}px`;
        tooltip.classList.add('tooltip-visible');
        
        return tooltip;
    }

    /**
     * Hide tooltip
     */
    function hideTooltip(tooltip) {
        if (tooltip && tooltip.parentNode) {
            tooltip.classList.remove('tooltip-visible');
            setTimeout(() => {
                if (tooltip.parentNode) {
                    tooltip.parentNode.removeChild(tooltip);
                }
            }, 200);
        }
    }

    /**
     * Add tooltips to technical terms
     */
    function addTechnicalTermTooltips() {
        const technicalTerms = {
            'causal inference': 'The process of determining cause-and-effect relationships from data',
            'treatment effect': 'The difference in outcomes between treated and untreated units',
            'confounding': 'When a third variable affects both the treatment and outcome, creating spurious associations',
            'randomization': 'Random assignment of units to treatment and control groups',
            'instrumental variable': 'A variable that affects the treatment but only affects the outcome through the treatment',
            'propensity score': 'The probability of receiving treatment given observed characteristics',
            'difference-in-differences': 'A method comparing changes over time between treatment and control groups',
            'regression discontinuity': 'A method exploiting arbitrary cutoffs in treatment assignment',
            'backdoor criterion': 'A set of variables that, when controlled for, eliminate confounding',
            'unconfoundedness': 'The assumption that all confounders are observed and controlled for'
        };

        Object.entries(technicalTerms).forEach(([term, definition]) => {
            const regex = new RegExp(`\\b${term}\\b`, 'gi');
            const walker = document.createTreeWalker(
                document.body,
                NodeFilter.SHOW_TEXT,
                null,
                false
            );

            const textNodes = [];
            let node;
            while (node = walker.nextNode()) {
                if (node.parentElement.tagName !== 'SCRIPT' && 
                    node.parentElement.tagName !== 'STYLE' &&
                    !node.parentElement.classList.contains('tooltip')) {
                    textNodes.push(node);
                }
            }

            textNodes.forEach(textNode => {
                if (regex.test(textNode.textContent)) {
                    const parent = textNode.parentElement;
                    const html = textNode.textContent.replace(regex, 
                        `<span class="technical-term" data-tooltip="${definition}" data-tooltip-type="definition">$&</span>`
                    );
                    
                    const wrapper = document.createElement('span');
                    wrapper.innerHTML = html;
                    parent.replaceChild(wrapper, textNode);
                    
                    // Initialize tooltips for new elements
                    wrapper.querySelectorAll('[data-tooltip]').forEach(createTooltip);
                }
            });
        });
    }

    /**
     * Add tooltips to method names
     */
    function addMethodTooltips() {
        const methodTooltips = {
            'RCT': 'Randomized Controlled Trial - The gold standard for causal inference',
            'DiD': 'Difference-in-Differences - Compares changes over time between groups',
            'IV': 'Instrumental Variables - Uses instruments to identify causal effects',
            'RDD': 'Regression Discontinuity Design - Exploits arbitrary cutoffs',
            'PSM': 'Propensity Score Matching - Matches similar units based on propensity scores',
            'PSW': 'Propensity Score Weighting - Weights observations by inverse propensity scores'
        };

        Object.entries(methodTooltips).forEach(([method, description]) => {
            // Find elements containing the method name using a different approach
            const codeElements = document.querySelectorAll('code, .method-name');
            codeElements.forEach(element => {
                if (element.textContent.includes(method) && !element.dataset.tooltip) {
                    element.dataset.tooltip = description;
                    element.dataset.tooltipType = 'method';
                    createTooltip(element);
                }
            });
        });
    }

    /**
     * Initialize expandable sections
     */
    function initializeExpandableSections() {
        // Find sections marked as expandable
        const expandableSections = document.querySelectorAll(`.${CONFIG.expandableClass}, [data-expandable]`);
        
        expandableSections.forEach(section => {
            makeExpandable(section);
        });

        // Auto-create expandable sections for long content
        createAutoExpandableSections();
    }

    /**
     * Make a section expandable
     */
    function makeExpandable(section) {
        const title = section.querySelector('h1, h2, h3, h4, h5, h6') || 
                     section.querySelector('.expandable-title');
        
        if (!title) return;

        const content = section.querySelector('.expandable-content') || 
                       getExpandableContent(section, title);
        
        if (!content) return;

        // Create expand/collapse button
        const toggleBtn = document.createElement('button');
        toggleBtn.className = 'expandable-toggle';
        toggleBtn.innerHTML = `
            <svg class="expand-icon" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                <polyline points="6,9 12,15 18,9"></polyline>
            </svg>
        `;
        toggleBtn.setAttribute('aria-label', 'Toggle section');
        toggleBtn.setAttribute('aria-expanded', 'true');

        title.appendChild(toggleBtn);
        title.classList.add('expandable-header');
        content.classList.add('expandable-content');

        // Initially collapsed if specified
        if (section.dataset.collapsed === 'true') {
            content.classList.add('collapsed');
            toggleBtn.setAttribute('aria-expanded', 'false');
            toggleBtn.classList.add('collapsed');
        }

        // Toggle functionality
        toggleBtn.addEventListener('click', (e) => {
            e.preventDefault();
            toggleSection(content, toggleBtn);
        });

        // Keyboard support
        title.addEventListener('keydown', (e) => {
            if (e.key === 'Enter' || e.key === ' ') {
                e.preventDefault();
                toggleSection(content, toggleBtn);
            }
        });
    }

    /**
     * Get expandable content for a section
     */
    function getExpandableContent(section, title) {
        const content = document.createElement('div');
        content.className = 'expandable-content';
        
        // Move all content after title into expandable content
        let nextElement = title.nextElementSibling;
        while (nextElement && !nextElement.matches('h1, h2, h3, h4, h5, h6')) {
            const current = nextElement;
            nextElement = nextElement.nextElementSibling;
            content.appendChild(current);
        }
        
        if (content.children.length > 0) {
            section.appendChild(content);
            return content;
        }
        
        return null;
    }

    /**
     * Toggle expandable section
     */
    function toggleSection(content, toggleBtn) {
        const isCollapsed = content.classList.contains('collapsed');
        
        if (isCollapsed) {
            content.classList.remove('collapsed');
            toggleBtn.classList.remove('collapsed');
            toggleBtn.setAttribute('aria-expanded', 'true');
        } else {
            content.classList.add('collapsed');
            toggleBtn.classList.add('collapsed');
            toggleBtn.setAttribute('aria-expanded', 'false');
        }

        // Track expansion events
        trackEvent('section_toggle', {
            'action': isCollapsed ? 'expand' : 'collapse',
            'section_id': content.closest('[id]')?.id || 'unknown'
        });
    }

    /**
     * Create auto-expandable sections for long content
     */
    function createAutoExpandableSections() {
        // Find long paragraphs or lists that could benefit from expansion
        const longContent = document.querySelectorAll('p, ul, ol, .admonition');
        
        longContent.forEach(element => {
            if (element.scrollHeight > 300 && !element.closest('.expandable-section')) {
                const wrapper = document.createElement('div');
                wrapper.className = 'auto-expandable-section';
                wrapper.dataset.collapsed = 'true';
                
                const title = document.createElement('div');
                title.className = 'expandable-title';
                title.textContent = getContentSummary(element);
                
                element.parentNode.insertBefore(wrapper, element);
                wrapper.appendChild(title);
                wrapper.appendChild(element);
                
                makeExpandable(wrapper);
            }
        });
    }

    /**
     * Get content summary for auto-expandable sections
     */
    function getContentSummary(element) {
        const text = element.textContent.trim();
        if (text.length > 100) {
            return text.substring(0, 100) + '...';
        }
        return 'Show details';
    }

    /**
     * Initialize interactive examples
     */
    function initializeInteractiveExamples() {
        // Disabled to prevent Mermaid conflicts
        // Interactive examples have been moved to separate pages
        console.log('Interactive examples disabled to prevent conflicts');
    }

    /**
     * Create interactive decision tree example
     */
    function createInteractiveDecisionTree(container) {
        container.innerHTML = `
            <div class="interactive-decision-tree">
                <div class="tree-header">
                    <h3>Interactive Decision Tree</h3>
                    <p>Answer questions about your dataset to see which causal method CAIS would recommend.</p>
                </div>
                <div class="tree-questions">
                    <div class="question active" data-question="1">
                        <h4>Do you have random assignment to treatment?</h4>
                        <div class="question-options">
                            <button class="option-btn" data-answer="yes" data-next="rct">Yes - Random assignment</button>
                            <button class="option-btn" data-answer="no" data-next="2">No - Observational data</button>
                        </div>
                    </div>
                    <div class="question" data-question="2">
                        <h4>Do you have a time dimension with treatment occurring at a specific time?</h4>
                        <div class="question-options">
                            <button class="option-btn" data-answer="yes" data-next="3">Yes - Panel data</button>
                            <button class="option-btn" data-answer="no" data-next="4">No - Cross-sectional</button>
                        </div>
                    </div>
                    <div class="question" data-question="3">
                        <h4>Do you have both treated and control units observed before and after treatment?</h4>
                        <div class="question-options">
                            <button class="option-btn" data-answer="yes" data-next="did">Yes - Difference-in-Differences</button>
                            <button class="option-btn" data-answer="no" data-next="4">No - Other methods</button>
                        </div>
                    </div>
                    <div class="question" data-question="4">
                        <h4>Do you have an instrumental variable?</h4>
                        <div class="question-options">
                            <button class="option-btn" data-answer="yes" data-next="iv">Yes - Instrumental Variable</button>
                            <button class="option-btn" data-answer="no" data-next="5">No - Continue</button>
                        </div>
                    </div>
                    <div class="question" data-question="5">
                        <h4>Is treatment assignment based on a continuous score with a cutoff?</h4>
                        <div class="question-options">
                            <button class="option-btn" data-answer="yes" data-next="rdd">Yes - Regression Discontinuity</button>
                            <button class="option-btn" data-answer="no" data-next="observational">No - Observational methods</button>
                        </div>
                    </div>
                </div>
                <div class="tree-results">
                    <div class="result" data-result="rct">
                        <h4>🎯 Recommended: Randomized Controlled Trial (RCT)</h4>
                        <p>Your data has random assignment, making RCT analysis the gold standard for causal inference.</p>
                        <div class="result-details">
                            <strong>Why this method:</strong>
                            <ul>
                                <li>Random assignment eliminates confounding</li>
                                <li>Provides unbiased treatment effect estimates</li>
                                <li>No additional assumptions needed</li>
                            </ul>
                        </div>
                    </div>
                    <div class="result" data-result="did">
                        <h4>📊 Recommended: Difference-in-Differences (DiD)</h4>
                        <p>Your panel data structure allows for DiD analysis to control for time-invariant confounders.</p>
                        <div class="result-details">
                            <strong>Why this method:</strong>
                            <ul>
                                <li>Controls for time-invariant unobserved confounders</li>
                                <li>Uses pre-treatment trends to identify effects</li>
                                <li>Robust to selection on observables and unobservables</li>
                            </ul>
                        </div>
                    </div>
                    <div class="result" data-result="iv">
                        <h4>🔧 Recommended: Instrumental Variables (IV)</h4>
                        <p>Your instrumental variable can help identify causal effects even with unobserved confounding.</p>
                        <div class="result-details">
                            <strong>Why this method:</strong>
                            <ul>
                                <li>Handles unobserved confounding</li>
                                <li>Uses exogenous variation in treatment</li>
                                <li>Provides local average treatment effects</li>
                            </ul>
                        </div>
                    </div>
                    <div class="result" data-result="rdd">
                        <h4>📏 Recommended: Regression Discontinuity Design (RDD)</h4>
                        <p>The cutoff-based assignment creates quasi-random variation near the threshold.</p>
                        <div class="result-details">
                            <strong>Why this method:</strong>
                            <ul>
                                <li>Exploits arbitrary cutoffs for identification</li>
                                <li>Provides local treatment effects at the cutoff</li>
                                <li>Robust to functional form assumptions</li>
                            </ul>
                        </div>
                    </div>
                    <div class="result" data-result="observational">
                        <h4>🔍 Recommended: Propensity Score Methods</h4>
                        <p>For observational data, propensity score matching or weighting can help control for confounding.</p>
                        <div class="result-details">
                            <strong>Why this method:</strong>
                            <ul>
                                <li>Balances treatment and control groups</li>
                                <li>Controls for observed confounders</li>
                                <li>Provides average treatment effects</li>
                            </ul>
                        </div>
                    </div>
                </div>
                <div class="tree-controls">
                    <button class="reset-tree">Start Over</button>
                </div>
            </div>
        `;

        initializeDecisionTreeInteraction(container);
    }

    /**
     * Initialize decision tree interaction
     */
    function initializeDecisionTreeInteraction(container) {
        const questions = container.querySelectorAll('.question');
        const results = container.querySelectorAll('.result');
        const resetBtn = container.querySelector('.reset-tree');

        container.addEventListener('click', (e) => {
            if (e.target.matches('.option-btn')) {
                const nextStep = e.target.dataset.next;
                const currentQuestion = e.target.closest('.question');
                
                // Hide current question
                currentQuestion.classList.remove('active');
                
                // Show next step
                if (nextStep.match(/^\d+$/)) {
                    // Next question
                    const nextQuestion = container.querySelector(`[data-question="${nextStep}"]`);
                    if (nextQuestion) {
                        nextQuestion.classList.add('active');
                    }
                } else {
                    // Show result
                    const result = container.querySelector(`[data-result="${nextStep}"]`);
                    if (result) {
                        result.classList.add('active');
                        container.querySelector('.tree-results').classList.add('active');
                    }
                }

                // Track decision path
                trackEvent('decision_tree_step', {
                    'question': currentQuestion.dataset.question,
                    'answer': e.target.dataset.answer,
                    'next_step': nextStep
                });
            }
        });

        resetBtn.addEventListener('click', () => {
            // Reset all questions and results
            questions.forEach(q => q.classList.remove('active'));
            results.forEach(r => r.classList.remove('active'));
            container.querySelector('.tree-results').classList.remove('active');
            
            // Show first question
            container.querySelector('[data-question="1"]').classList.add('active');
            
            trackEvent('decision_tree_reset');
        });
    }

    /**
     * Create method comparison interactive example
     */
    function createMethodComparison(container) {
        // Implementation for method comparison would go here
        container.innerHTML = `
            <div class="method-comparison">
                <h3>Method Comparison Tool</h3>
                <p>Compare different causal inference methods side by side.</p>
                <div class="comparison-controls">
                    <select class="method-select" data-side="left">
                        <option value="rct">Randomized Controlled Trial</option>
                        <option value="did">Difference-in-Differences</option>
                        <option value="iv">Instrumental Variables</option>
                        <option value="rdd">Regression Discontinuity</option>
                        <option value="psm">Propensity Score Matching</option>
                    </select>
                    <span class="vs">VS</span>
                    <select class="method-select" data-side="right">
                        <option value="did">Difference-in-Differences</option>
                        <option value="rct">Randomized Controlled Trial</option>
                        <option value="iv">Instrumental Variables</option>
                        <option value="rdd">Regression Discontinuity</option>
                        <option value="psm">Propensity Score Matching</option>
                    </select>
                </div>
                <div class="comparison-results">
                    <div class="method-details left">
                        <h4>Randomized Controlled Trial</h4>
                        <div class="method-info">Loading...</div>
                    </div>
                    <div class="method-details right">
                        <h4>Difference-in-Differences</h4>
                        <div class="method-info">Loading...</div>
                    </div>
                </div>
            </div>
        `;
    }

    /**
     * Create dataset analyzer interactive example
     */
    function createDatasetAnalyzer(container) {
        // Implementation for dataset analyzer would go here
        container.innerHTML = `
            <div class="dataset-analyzer">
                <h3>Dataset Analyzer</h3>
                <p>Upload or describe your dataset to see what CAIS would recommend.</p>
                <div class="analyzer-input">
                    <textarea placeholder="Describe your dataset: variables, structure, research question..."></textarea>
                    <button class="analyze-btn">Analyze Dataset</button>
                </div>
                <div class="analyzer-results hidden">
                    <h4>Analysis Results</h4>
                    <div class="results-content"></div>
                </div>
            </div>
        `;
    }

    /**
     * Create generic interactive example
     */
    function createGenericInteractiveExample(container) {
        // Add basic interactivity to any example
        const codeBlocks = container.querySelectorAll('pre, code');
        codeBlocks.forEach(block => {
            block.addEventListener('click', () => {
                block.classList.toggle('highlighted');
            });
        });
    }

    /**
     * Initialize accessibility features
     */
    function initializeAccessibilityFeatures() {
        // Add skip links
        addSkipLinks();
        
        // Improve focus management
        improveFocusManagement();
        
        // Add ARIA labels where needed
        addAriaLabels();
        
        // Ensure keyboard navigation
        ensureKeyboardNavigation();
    }

    /**
     * Add skip links for accessibility
     */
    function addSkipLinks() {
        const skipLinks = document.createElement('div');
        skipLinks.className = 'skip-links';
        skipLinks.innerHTML = `
            <a href="#main-content" class="skip-link">Skip to main content</a>
            <a href="#navigation" class="skip-link">Skip to navigation</a>
            <a href="#search" class="skip-link">Skip to search</a>
        `;
        
        document.body.insertBefore(skipLinks, document.body.firstChild);
    }

    /**
     * Improve focus management
     */
    function improveFocusManagement() {
        // Ensure interactive elements are focusable
        const interactiveElements = document.querySelectorAll(
            '.copy-code-btn, .expandable-toggle, .rating-btn, .filter-btn'
        );
        
        interactiveElements.forEach(element => {
            if (!element.hasAttribute('tabindex')) {
                element.setAttribute('tabindex', '0');
            }
        });
    }

    /**
     * Add ARIA labels where needed
     */
    function addAriaLabels() {
        // Add labels to buttons without text
        const iconButtons = document.querySelectorAll('button:not([aria-label]):empty, button:not([aria-label]) svg:only-child');
        iconButtons.forEach(button => {
            if (!button.getAttribute('aria-label')) {
                button.setAttribute('aria-label', 'Button');
            }
        });
    }

    /**
     * Ensure keyboard navigation works properly
     */
    function ensureKeyboardNavigation() {
        // Add keyboard support to custom interactive elements
        document.addEventListener('keydown', (e) => {
            if (e.target.matches('.expandable-toggle, .rating-btn, .filter-btn')) {
                if (e.key === 'Enter' || e.key === ' ') {
                    e.preventDefault();
                    e.target.click();
                }
            }
        });
    }

    /**
     * Track events for analytics
     */
    function trackEvent(eventName, parameters = {}) {
        // Google Analytics 4
        if (typeof gtag !== 'undefined') {
            gtag('event', eventName, {
                'event_category': 'documentation_interaction',
                ...parameters
            });
        }
        
        // Console log for development
        console.log('Event tracked:', eventName, parameters);
    }

    // Initialize when DOM is ready
    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', initializeInteractiveFeatures);
    } else {
        initializeInteractiveFeatures();
    }

    // Expose some functions globally for external use
    window.CAISInteractive = {
        initializeCopyToClipboard,
        initializeFeedbackSystem,
        initializeTooltips,
        initializeExpandableSections,
        trackEvent
    };

})();
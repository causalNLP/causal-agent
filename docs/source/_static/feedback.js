/**
 * Documentation Feedback Collection System
 * Collects user feedback and analytics for documentation improvement
 */

class DocumentationFeedback {
    constructor() {
        this.feedbackEndpoint = 'https://api.github.com/repos/causal-ai-scientist/causal-ai-scientist/issues';
        this.analyticsEndpoint = '/api/feedback'; // Custom endpoint if available
        this.init();
    }

    init() {
        this.createFeedbackWidget();
        this.setupEventListeners();
        this.trackUserBehavior();
    }

    createFeedbackWidget() {
        // Create floating feedback button
        const feedbackButton = document.createElement('div');
        feedbackButton.id = 'feedback-button';
        feedbackButton.innerHTML = `
            <button class="feedback-btn" title="Send Feedback">
                <svg width="20" height="20" viewBox="0 0 24 24" fill="currentColor">
                    <path d="M20 2H4c-1.1 0-2 .9-2 2v12c0 1.1.9 2 2 2h4l4 4 4-4h4c1.1 0 2-.9 2-2V4c0-1.1-.9-2-2-2zm-2 12H6v-2h12v2zm0-3H6V9h12v2zm0-3H6V6h12v2z"/>
                </svg>
                Feedback
            </button>
        `;
        
        feedbackButton.style.cssText = `
            position: fixed;
            bottom: 20px;
            right: 20px;
            z-index: 1000;
            background: #2980b9;
            color: white;
            border: none;
            border-radius: 25px;
            padding: 10px 15px;
            cursor: pointer;
            box-shadow: 0 2px 10px rgba(0,0,0,0.2);
            font-size: 14px;
            display: flex;
            align-items: center;
            gap: 5px;
            transition: all 0.3s ease;
        `;

        document.body.appendChild(feedbackButton);

        // Create feedback modal
        this.createFeedbackModal();
    }

    createFeedbackModal() {
        const modal = document.createElement('div');
        modal.id = 'feedback-modal';
        modal.innerHTML = `
            <div class="feedback-modal-overlay">
                <div class="feedback-modal-content">
                    <div class="feedback-modal-header">
                        <h3>Help us improve the documentation</h3>
                        <button class="feedback-close">&times;</button>
                    </div>
                    <div class="feedback-modal-body">
                        <div class="feedback-tabs">
                            <button class="feedback-tab active" data-tab="quick">Quick Feedback</button>
                            <button class="feedback-tab" data-tab="detailed">Detailed Report</button>
                        </div>
                        
                        <div class="feedback-content" id="quick-feedback">
                            <p>How helpful was this page?</p>
                            <div class="rating-buttons">
                                <button class="rating-btn" data-rating="1">😞 Not helpful</button>
                                <button class="rating-btn" data-rating="2">😐 Somewhat helpful</button>
                                <button class="rating-btn" data-rating="3">😊 Very helpful</button>
                            </div>
                            <textarea placeholder="Any additional comments? (optional)" id="quick-comment"></textarea>
                            <button class="submit-feedback" data-type="quick">Submit Feedback</button>
                        </div>
                        
                        <div class="feedback-content hidden" id="detailed-feedback">
                            <form id="detailed-feedback-form">
                                <div class="form-group">
                                    <label>Issue Type:</label>
                                    <select id="issue-type">
                                        <option value="content">Content Issue</option>
                                        <option value="technical">Technical Error</option>
                                        <option value="suggestion">Suggestion</option>
                                        <option value="missing">Missing Information</option>
                                        <option value="other">Other</option>
                                    </select>
                                </div>
                                <div class="form-group">
                                    <label>Page URL:</label>
                                    <input type="text" id="page-url" readonly>
                                </div>
                                <div class="form-group">
                                    <label>Description:</label>
                                    <textarea id="issue-description" placeholder="Please describe the issue or suggestion in detail..." required></textarea>
                                </div>
                                <div class="form-group">
                                    <label>Expected Behavior (optional):</label>
                                    <textarea id="expected-behavior" placeholder="What did you expect to see or happen?"></textarea>
                                </div>
                                <div class="form-group">
                                    <label>Your Role:</label>
                                    <select id="user-role">
                                        <option value="">Select your role (optional)</option>
                                        <option value="student">Student</option>
                                        <option value="researcher">Researcher</option>
                                        <option value="data-scientist">Data Scientist</option>
                                        <option value="developer">Developer</option>
                                        <option value="other">Other</option>
                                    </select>
                                </div>
                                <button type="submit" class="submit-feedback" data-type="detailed">Submit Detailed Report</button>
                            </form>
                        </div>
                    </div>
                </div>
            </div>
        `;

        modal.style.cssText = `
            display: none;
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            z-index: 2000;
        `;

        document.body.appendChild(modal);
        this.addModalStyles();
    }

    addModalStyles() {
        const style = document.createElement('style');
        style.textContent = `
            .feedback-modal-overlay {
                background: rgba(0, 0, 0, 0.5);
                width: 100%;
                height: 100%;
                display: flex;
                justify-content: center;
                align-items: center;
                padding: 20px;
            }
            
            .feedback-modal-content {
                background: white;
                border-radius: 8px;
                max-width: 600px;
                width: 100%;
                max-height: 80vh;
                overflow-y: auto;
                box-shadow: 0 4px 20px rgba(0, 0, 0, 0.3);
            }
            
            .feedback-modal-header {
                padding: 20px;
                border-bottom: 1px solid #eee;
                display: flex;
                justify-content: space-between;
                align-items: center;
            }
            
            .feedback-modal-header h3 {
                margin: 0;
                color: #333;
            }
            
            .feedback-close {
                background: none;
                border: none;
                font-size: 24px;
                cursor: pointer;
                color: #999;
            }
            
            .feedback-modal-body {
                padding: 20px;
            }
            
            .feedback-tabs {
                display: flex;
                margin-bottom: 20px;
                border-bottom: 1px solid #eee;
            }
            
            .feedback-tab {
                background: none;
                border: none;
                padding: 10px 20px;
                cursor: pointer;
                border-bottom: 2px solid transparent;
                color: #666;
            }
            
            .feedback-tab.active {
                color: #2980b9;
                border-bottom-color: #2980b9;
            }
            
            .feedback-content.hidden {
                display: none;
            }
            
            .rating-buttons {
                display: flex;
                gap: 10px;
                margin: 15px 0;
                flex-wrap: wrap;
            }
            
            .rating-btn {
                background: #f8f9fa;
                border: 2px solid #dee2e6;
                border-radius: 6px;
                padding: 10px 15px;
                cursor: pointer;
                transition: all 0.2s;
                flex: 1;
                min-width: 120px;
            }
            
            .rating-btn:hover, .rating-btn.selected {
                background: #e3f2fd;
                border-color: #2196f3;
            }
            
            .form-group {
                margin-bottom: 15px;
            }
            
            .form-group label {
                display: block;
                margin-bottom: 5px;
                font-weight: bold;
                color: #333;
            }
            
            .form-group input, .form-group select, .form-group textarea {
                width: 100%;
                padding: 8px 12px;
                border: 1px solid #ddd;
                border-radius: 4px;
                font-size: 14px;
                box-sizing: border-box;
            }
            
            .form-group textarea {
                min-height: 80px;
                resize: vertical;
            }
            
            #quick-comment {
                width: 100%;
                min-height: 60px;
                margin: 15px 0;
                padding: 10px;
                border: 1px solid #ddd;
                border-radius: 4px;
                resize: vertical;
                box-sizing: border-box;
            }
            
            .submit-feedback {
                background: #2980b9;
                color: white;
                border: none;
                padding: 12px 24px;
                border-radius: 6px;
                cursor: pointer;
                font-size: 14px;
                transition: background 0.2s;
            }
            
            .submit-feedback:hover {
                background: #21618c;
            }
            
            .submit-feedback:disabled {
                background: #bdc3c7;
                cursor: not-allowed;
            }
            
            @media (max-width: 768px) {
                .feedback-modal-content {
                    margin: 10px;
                    max-height: 90vh;
                }
                
                .rating-buttons {
                    flex-direction: column;
                }
                
                .rating-btn {
                    min-width: auto;
                }
            }
        `;
        document.head.appendChild(style);
    }

    setupEventListeners() {
        // Feedback button click
        document.getElementById('feedback-button').addEventListener('click', () => {
            this.showFeedbackModal();
        });

        // Modal close
        document.querySelector('.feedback-close').addEventListener('click', () => {
            this.hideFeedbackModal();
        });

        // Modal overlay click
        document.querySelector('.feedback-modal-overlay').addEventListener('click', (e) => {
            if (e.target === e.currentTarget) {
                this.hideFeedbackModal();
            }
        });

        // Tab switching
        document.querySelectorAll('.feedback-tab').forEach(tab => {
            tab.addEventListener('click', (e) => {
                this.switchTab(e.target.dataset.tab);
            });
        });

        // Rating buttons
        document.querySelectorAll('.rating-btn').forEach(btn => {
            btn.addEventListener('click', (e) => {
                document.querySelectorAll('.rating-btn').forEach(b => b.classList.remove('selected'));
                e.target.classList.add('selected');
            });
        });

        // Form submissions
        document.querySelectorAll('.submit-feedback').forEach(btn => {
            btn.addEventListener('click', (e) => {
                this.submitFeedback(e.target.dataset.type);
            });
        });

        // Escape key to close modal
        document.addEventListener('keydown', (e) => {
            if (e.key === 'Escape') {
                this.hideFeedbackModal();
            }
        });
    }

    showFeedbackModal() {
        const modal = document.getElementById('feedback-modal');
        modal.style.display = 'block';
        document.getElementById('page-url').value = window.location.href;
        
        // Track modal open
        if (typeof gtag !== 'undefined') {
            gtag('event', 'feedback_modal_open', {
                'page_url': window.location.href,
                'documentation_section': this.getDocumentationSection()
            });
        }
    }

    hideFeedbackModal() {
        document.getElementById('feedback-modal').style.display = 'none';
    }

    switchTab(tabName) {
        // Update tab buttons
        document.querySelectorAll('.feedback-tab').forEach(tab => {
            tab.classList.remove('active');
        });
        document.querySelector(`[data-tab="${tabName}"]`).classList.add('active');

        // Update content
        document.querySelectorAll('.feedback-content').forEach(content => {
            content.classList.add('hidden');
        });
        document.getElementById(`${tabName}-feedback`).classList.remove('hidden');
    }

    async submitFeedback(type) {
        const submitBtn = document.querySelector(`[data-type="${type}"]`);
        submitBtn.disabled = true;
        submitBtn.textContent = 'Submitting...';

        try {
            let feedbackData;
            
            if (type === 'quick') {
                const selectedRating = document.querySelector('.rating-btn.selected');
                const comment = document.getElementById('quick-comment').value;
                
                if (!selectedRating) {
                    alert('Please select a rating');
                    return;
                }
                
                feedbackData = {
                    type: 'quick',
                    rating: selectedRating.dataset.rating,
                    comment: comment,
                    page_url: window.location.href,
                    page_title: document.title,
                    user_agent: navigator.userAgent,
                    timestamp: new Date().toISOString()
                };
            } else {
                const form = document.getElementById('detailed-feedback-form');
                const formData = new FormData(form);
                
                feedbackData = {
                    type: 'detailed',
                    issue_type: document.getElementById('issue-type').value,
                    page_url: document.getElementById('page-url').value,
                    description: document.getElementById('issue-description').value,
                    expected_behavior: document.getElementById('expected-behavior').value,
                    user_role: document.getElementById('user-role').value,
                    user_agent: navigator.userAgent,
                    timestamp: new Date().toISOString()
                };
                
                if (!feedbackData.description.trim()) {
                    alert('Please provide a description');
                    return;
                }
            }

            // Send to analytics
            if (typeof gtag !== 'undefined') {
                gtag('event', 'feedback_submitted', {
                    'feedback_type': type,
                    'page_url': feedbackData.page_url,
                    'documentation_section': this.getDocumentationSection()
                });
            }

            // Store locally for now (in production, send to backend)
            this.storeFeedbackLocally(feedbackData);
            
            // Create GitHub issue for detailed feedback
            if (type === 'detailed') {
                await this.createGitHubIssue(feedbackData);
            }

            this.showSuccessMessage();
            this.hideFeedbackModal();
            
        } catch (error) {
            console.error('Error submitting feedback:', error);
            alert('Error submitting feedback. Please try again.');
        } finally {
            submitBtn.disabled = false;
            submitBtn.textContent = type === 'quick' ? 'Submit Feedback' : 'Submit Detailed Report';
        }
    }

    storeFeedbackLocally(feedbackData) {
        const existingFeedback = JSON.parse(localStorage.getItem('cais_feedback') || '[]');
        existingFeedback.push(feedbackData);
        localStorage.setItem('cais_feedback', JSON.stringify(existingFeedback));
    }

    async createGitHubIssue(feedbackData) {
        // This would require a backend service to create GitHub issues
        // For now, we'll just log the data
        console.log('GitHub issue data:', {
            title: `Documentation Feedback: ${feedbackData.issue_type}`,
            body: `
**Page:** ${feedbackData.page_url}
**Issue Type:** ${feedbackData.issue_type}
**User Role:** ${feedbackData.user_role || 'Not specified'}

**Description:**
${feedbackData.description}

**Expected Behavior:**
${feedbackData.expected_behavior || 'Not specified'}

**Browser:** ${feedbackData.user_agent}
**Timestamp:** ${feedbackData.timestamp}
            `,
            labels: ['documentation', 'user-feedback', feedbackData.issue_type]
        });
    }

    showSuccessMessage() {
        const message = document.createElement('div');
        message.style.cssText = `
            position: fixed;
            top: 20px;
            right: 20px;
            background: #27ae60;
            color: white;
            padding: 15px 20px;
            border-radius: 6px;
            z-index: 3000;
            box-shadow: 0 2px 10px rgba(0,0,0,0.2);
        `;
        message.textContent = 'Thank you for your feedback!';
        document.body.appendChild(message);

        setTimeout(() => {
            message.remove();
        }, 3000);
    }

    trackUserBehavior() {
        // Track scroll depth
        let maxScroll = 0;
        window.addEventListener('scroll', () => {
            const scrollPercent = Math.round((window.scrollY / (document.body.scrollHeight - window.innerHeight)) * 100);
            if (scrollPercent > maxScroll) {
                maxScroll = scrollPercent;
            }
        });

        // Send scroll depth on page unload
        window.addEventListener('beforeunload', () => {
            if (typeof gtag !== 'undefined') {
                gtag('event', 'scroll_depth', {
                    'value': maxScroll,
                    'page_url': window.location.href,
                    'documentation_section': this.getDocumentationSection()
                });
            }
        });

        // Track search usage
        this.trackSearchUsage();
    }

    trackSearchUsage() {
        const searchInput = document.querySelector('input[name="q"]');
        if (searchInput) {
            let searchTimeout;
            searchInput.addEventListener('input', (e) => {
                clearTimeout(searchTimeout);
                searchTimeout = setTimeout(() => {
                    if (e.target.value.length > 2) {
                        if (typeof gtag !== 'undefined') {
                            gtag('event', 'search', {
                                'search_term': e.target.value,
                                'documentation_section': this.getDocumentationSection()
                            });
                        }
                    }
                }, 1000);
            });
        }
    }

    getDocumentationSection() {
        const path = window.location.pathname;
        if (path.includes('/getting_started/')) return 'getting_started';
        if (path.includes('/user_guide/')) return 'user_guide';
        if (path.includes('/tutorials/')) return 'tutorials';
        if (path.includes('/api/')) return 'api';
        if (path.includes('/methods/')) return 'methods';
        if (path.includes('/theory/')) return 'theory';
        if (path.includes('/development/')) return 'development';
        if (path.includes('/about/')) return 'about';
        return 'homepage';
    }
}

// Initialize feedback system when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    new DocumentationFeedback();
});
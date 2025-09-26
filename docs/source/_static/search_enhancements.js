/**
 * Enhanced search functionality for CAIS documentation
 * Provides search result filtering, highlighting, and improved UX
 */

(function() {
    'use strict';

    // Search enhancement configuration
    const SEARCH_CONFIG = {
        highlightClass: 'search-highlight',
        filterClass: 'search-filter',
        resultClass: 'search-result-enhanced',
        minQueryLength: 2,
        maxResults: 50,
        debounceDelay: 300
    };

    // Search categories for filtering
    const SEARCH_CATEGORIES = {
        'api': {
            name: 'API Reference',
            patterns: ['/api/', 'class ', 'function ', 'method '],
            icon: '📚'
        },
        'methods': {
            name: 'Causal Methods',
            patterns: ['/methods/', 'causal', 'inference', 'treatment', 'effect'],
            icon: '🔬'
        },
        'tutorials': {
            name: 'Tutorials',
            patterns: ['/tutorials/', 'notebook', 'example', 'walkthrough'],
            icon: '📖'
        },
        'theory': {
            name: 'Theory',
            patterns: ['/theory/', 'background', 'concept', 'principle'],
            icon: '🧠'
        },
        'development': {
            name: 'Development',
            patterns: ['/development/', 'architecture', 'extending', 'contributing'],
            icon: '⚙️'
        }
    };

    let searchTimeout;
    let currentResults = [];

    /**
     * Initialize search enhancements when DOM is ready
     */
    function initializeSearchEnhancements() {
        // Wait for RTD search to be available
        if (typeof Search === 'undefined') {
            setTimeout(initializeSearchEnhancements, 100);
            return;
        }

        enhanceSearchInterface();
        addSearchFilters();
        improveSearchResults();
        addKeyboardNavigation();
        addSearchAnalytics();
    }

    /**
     * Enhance the search interface with better UX
     */
    function enhanceSearchInterface() {
        const searchBox = document.getElementById('rtd-search-form');
        if (!searchBox) return;

        // Add search suggestions container
        const suggestionsContainer = document.createElement('div');
        suggestionsContainer.id = 'search-suggestions';
        suggestionsContainer.className = 'search-suggestions hidden';
        searchBox.appendChild(suggestionsContainer);

        // Add loading indicator
        const loadingIndicator = document.createElement('div');
        loadingIndicator.id = 'search-loading';
        loadingIndicator.className = 'search-loading hidden';
        loadingIndicator.innerHTML = '<span class="spinner"></span> Searching...';
        searchBox.appendChild(loadingIndicator);

        // Enhance search input
        const searchInput = searchBox.querySelector('input[name="q"]');
        if (searchInput) {
            searchInput.setAttribute('placeholder', 'Search documentation... (e.g., "decision tree", "RCT method")');
            searchInput.setAttribute('autocomplete', 'off');
            searchInput.addEventListener('input', debounce(handleSearchInput, SEARCH_CONFIG.debounceDelay));
            searchInput.addEventListener('focus', showSearchSuggestions);
            searchInput.addEventListener('blur', hideSearchSuggestions);
        }
    }

    /**
     * Add category-based search filters
     */
    function addSearchFilters() {
        const searchResults = document.getElementById('search-results');
        if (!searchResults) return;

        // Create filter container
        const filterContainer = document.createElement('div');
        filterContainer.id = 'search-filters';
        filterContainer.className = 'search-filters';
        filterContainer.innerHTML = `
            <div class="filter-header">
                <h4>Filter Results</h4>
                <button class="clear-filters" onclick="clearSearchFilters()">Clear All</button>
            </div>
            <div class="filter-options">
                <button class="filter-btn active" data-category="all">
                    All Results <span class="count" id="count-all">0</span>
                </button>
                ${Object.entries(SEARCH_CATEGORIES).map(([key, category]) => `
                    <button class="filter-btn" data-category="${key}">
                        ${category.icon} ${category.name} <span class="count" id="count-${key}">0</span>
                    </button>
                `).join('')}
            </div>
        `;

        // Insert filters before search results
        searchResults.parentNode.insertBefore(filterContainer, searchResults);

        // Add filter event listeners
        filterContainer.addEventListener('click', handleFilterClick);
    }

    /**
     * Improve search results display and functionality
     */
    function improveSearchResults() {
        // Override the default search result display
        if (typeof Search !== 'undefined' && Search.displaySearchResults) {
            const originalDisplayResults = Search.displaySearchResults;
            Search.displaySearchResults = function(results, searchterms) {
                currentResults = results;
                enhancedDisplayResults(results, searchterms);
                updateFilterCounts(results);
                addResultEnhancements();
            };
        }
    }

    /**
     * Enhanced display of search results with better formatting
     */
    function enhancedDisplayResults(results, searchterms) {
        const searchResults = document.getElementById('search-results');
        if (!searchResults) return;

        if (results.length === 0) {
            searchResults.innerHTML = `
                <div class="no-results">
                    <h3>No results found</h3>
                    <p>Try different keywords or check out our popular sections:</p>
                    <ul class="popular-links">
                        <li><a href="getting_started/">Getting Started</a></li>
                        <li><a href="methods/">Causal Methods</a></li>
                        <li><a href="tutorials/">Tutorials</a></li>
                        <li><a href="api/">API Reference</a></li>
                    </ul>
                </div>
            `;
            return;
        }

        const resultsHtml = results.map((result, index) => {
            const category = categorizeResult(result);
            const snippet = createResultSnippet(result, searchterms);
            const relevanceScore = calculateRelevanceScore(result, searchterms);
            
            return `
                <div class="search-result-item ${SEARCH_CONFIG.resultClass}" 
                     data-category="${category}" 
                     data-relevance="${relevanceScore}"
                     data-index="${index}">
                    <div class="result-header">
                        <h3 class="result-title">
                            <a href="${result[0]}" class="result-link">
                                ${highlightSearchTerms(result[2], searchterms)}
                            </a>
                            <span class="result-category">${SEARCH_CATEGORIES[category]?.icon || '📄'}</span>
                        </h3>
                        <div class="result-meta">
                            <span class="result-path">${result[0]}</span>
                            <span class="result-score">Relevance: ${Math.round(relevanceScore * 100)}%</span>
                        </div>
                    </div>
                    <div class="result-snippet">
                        ${snippet}
                    </div>
                </div>
            `;
        }).join('');

        searchResults.innerHTML = `
            <div class="search-results-header">
                <h2>Search Results</h2>
                <div class="results-info">
                    Found ${results.length} result${results.length !== 1 ? 's' : ''} for "${searchterms.join(' ')}"
                </div>
            </div>
            <div class="search-results-list">
                ${resultsHtml}
            </div>
        `;
    }

    /**
     * Add keyboard navigation for search results
     */
    function addKeyboardNavigation() {
        let selectedIndex = -1;
        
        document.addEventListener('keydown', function(e) {
            const searchInput = document.querySelector('#rtd-search-form input[name="q"]');
            const results = document.querySelectorAll('.search-result-item');
            
            if (!searchInput || results.length === 0) return;
            
            switch(e.key) {
                case 'ArrowDown':
                    e.preventDefault();
                    selectedIndex = Math.min(selectedIndex + 1, results.length - 1);
                    updateSelection(results, selectedIndex);
                    break;
                case 'ArrowUp':
                    e.preventDefault();
                    selectedIndex = Math.max(selectedIndex - 1, -1);
                    updateSelection(results, selectedIndex);
                    break;
                case 'Enter':
                    if (selectedIndex >= 0 && results[selectedIndex]) {
                        const link = results[selectedIndex].querySelector('.result-link');
                        if (link) {
                            window.location.href = link.href;
                        }
                    }
                    break;
                case 'Escape':
                    selectedIndex = -1;
                    updateSelection(results, selectedIndex);
                    searchInput.blur();
                    break;
            }
        });
    }

    /**
     * Add search analytics tracking
     */
    function addSearchAnalytics() {
        // Track search queries and popular results
        const searchInput = document.querySelector('#rtd-search-form input[name="q"]');
        if (searchInput) {
            searchInput.addEventListener('search', function(e) {
                if (typeof gtag !== 'undefined') {
                    gtag('event', 'search', {
                        'search_term': e.target.value,
                        'event_category': 'documentation'
                    });
                }
            });
        }

        // Track result clicks
        document.addEventListener('click', function(e) {
            if (e.target.matches('.result-link')) {
                if (typeof gtag !== 'undefined') {
                    gtag('event', 'click', {
                        'event_category': 'search_result',
                        'event_label': e.target.href
                    });
                }
            }
        });
    }

    // Utility functions
    function debounce(func, wait) {
        return function executedFunction(...args) {
            const later = () => {
                clearTimeout(searchTimeout);
                func(...args);
            };
            clearTimeout(searchTimeout);
            searchTimeout = setTimeout(later, wait);
        };
    }

    function handleSearchInput(e) {
        const query = e.target.value.trim();
        if (query.length >= SEARCH_CONFIG.minQueryLength) {
            showLoadingIndicator();
            // Trigger search after debounce
            setTimeout(() => hideLoadingIndicator(), 500);
        }
    }

    function showSearchSuggestions() {
        const suggestions = document.getElementById('search-suggestions');
        if (suggestions) {
            suggestions.classList.remove('hidden');
        }
    }

    function hideSearchSuggestions() {
        setTimeout(() => {
            const suggestions = document.getElementById('search-suggestions');
            if (suggestions) {
                suggestions.classList.add('hidden');
            }
        }, 200);
    }

    function showLoadingIndicator() {
        const loading = document.getElementById('search-loading');
        if (loading) {
            loading.classList.remove('hidden');
        }
    }

    function hideLoadingIndicator() {
        const loading = document.getElementById('search-loading');
        if (loading) {
            loading.classList.add('hidden');
        }
    }

    function categorizeResult(result) {
        const url = result[0].toLowerCase();
        const title = result[2].toLowerCase();
        const content = (result[3] || '').toLowerCase();
        const text = `${url} ${title} ${content}`;

        for (const [category, config] of Object.entries(SEARCH_CATEGORIES)) {
            if (config.patterns.some(pattern => text.includes(pattern.toLowerCase()))) {
                return category;
            }
        }
        return 'other';
    }

    function createResultSnippet(result, searchterms) {
        let snippet = result[3] || result[2] || '';
        if (snippet.length > 200) {
            snippet = snippet.substring(0, 200) + '...';
        }
        return highlightSearchTerms(snippet, searchterms);
    }

    function calculateRelevanceScore(result, searchterms) {
        let score = 0;
        const title = result[2].toLowerCase();
        const content = (result[3] || '').toLowerCase();
        
        searchterms.forEach(term => {
            const termLower = term.toLowerCase();
            // Title matches are more important
            if (title.includes(termLower)) score += 0.5;
            // Content matches
            if (content.includes(termLower)) score += 0.3;
            // Exact matches get bonus
            if (title === termLower || content.includes(` ${termLower} `)) score += 0.2;
        });
        
        return Math.min(score, 1.0);
    }

    function highlightSearchTerms(text, searchterms) {
        let highlightedText = text;
        searchterms.forEach(term => {
            const regex = new RegExp(`(${escapeRegExp(term)})`, 'gi');
            highlightedText = highlightedText.replace(regex, `<mark class="${SEARCH_CONFIG.highlightClass}">$1</mark>`);
        });
        return highlightedText;
    }

    function escapeRegExp(string) {
        return string.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
    }

    function handleFilterClick(e) {
        if (e.target.matches('.filter-btn')) {
            const category = e.target.dataset.category;
            filterResults(category);
            
            // Update active filter
            document.querySelectorAll('.filter-btn').forEach(btn => btn.classList.remove('active'));
            e.target.classList.add('active');
        }
    }

    function filterResults(category) {
        const results = document.querySelectorAll('.search-result-item');
        results.forEach(result => {
            if (category === 'all' || result.dataset.category === category) {
                result.style.display = 'block';
            } else {
                result.style.display = 'none';
            }
        });
    }

    function updateFilterCounts(results) {
        const counts = { all: results.length };
        
        // Initialize category counts
        Object.keys(SEARCH_CATEGORIES).forEach(category => {
            counts[category] = 0;
        });
        
        // Count results by category
        results.forEach(result => {
            const category = categorizeResult(result);
            if (counts[category] !== undefined) {
                counts[category]++;
            }
        });
        
        // Update count displays
        Object.entries(counts).forEach(([category, count]) => {
            const countElement = document.getElementById(`count-${category}`);
            if (countElement) {
                countElement.textContent = count;
            }
        });
    }

    function updateSelection(results, selectedIndex) {
        results.forEach((result, index) => {
            if (index === selectedIndex) {
                result.classList.add('selected');
                result.scrollIntoView({ block: 'nearest' });
            } else {
                result.classList.remove('selected');
            }
        });
    }

    function addResultEnhancements() {
        // Add click tracking and other enhancements to results
        document.querySelectorAll('.search-result-item').forEach(result => {
            result.addEventListener('mouseenter', function() {
                this.classList.add('hovered');
            });
            
            result.addEventListener('mouseleave', function() {
                this.classList.remove('hovered');
            });
        });
    }

    // Global functions for filter controls
    window.clearSearchFilters = function() {
        filterResults('all');
        document.querySelectorAll('.filter-btn').forEach(btn => btn.classList.remove('active'));
        document.querySelector('.filter-btn[data-category="all"]').classList.add('active');
    };

    // Initialize when DOM is ready
    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', initializeSearchEnhancements);
    } else {
        initializeSearchEnhancements();
    }

})();
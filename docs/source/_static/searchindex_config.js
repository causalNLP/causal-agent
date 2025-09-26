/**
 * Enhanced search index configuration for CAIS documentation
 * Improves search relevance and provides better categorization
 */

// Search index weights for different content types
const SEARCH_WEIGHTS = {
    title: 10,
    heading: 8,
    content: 5,
    code: 3,
    comment: 1
};

// Content type patterns for better search categorization
const CONTENT_PATTERNS = {
    api: {
        patterns: ['class ', 'def ', 'function ', 'method ', 'parameter', 'return', 'raises'],
        boost: 1.2
    },
    tutorial: {
        patterns: ['tutorial', 'example', 'walkthrough', 'step', 'guide'],
        boost: 1.1
    },
    method: {
        patterns: ['causal', 'inference', 'treatment', 'effect', 'estimator', 'diagnostic'],
        boost: 1.3
    },
    theory: {
        patterns: ['theory', 'concept', 'principle', 'assumption', 'background'],
        boost: 1.0
    }
};

// Stop words to exclude from search (in addition to default)
const CUSTOM_STOP_WORDS = [
    'cais', 'causal', 'agent', 'ai', 'scientist',  // Too common in our docs
    'documentation', 'docs', 'page', 'section',
    'example', 'note', 'warning', 'see', 'also'
];

// Synonyms for better search matching
const SEARCH_SYNONYMS = {
    'rct': ['randomized controlled trial', 'randomized experiment'],
    'did': ['difference in differences', 'diff in diff'],
    'iv': ['instrumental variable', 'instrumental variables'],
    'rdd': ['regression discontinuity', 'regression discontinuity design'],
    'ps': ['propensity score', 'propensity matching'],
    'ate': ['average treatment effect'],
    'att': ['average treatment effect on treated'],
    'llm': ['large language model', 'language model'],
    'ml': ['machine learning'],
    'ai': ['artificial intelligence']
};

// Export configuration for use by search enhancement script
if (typeof window !== 'undefined') {
    window.CAIS_SEARCH_CONFIG = {
        weights: SEARCH_WEIGHTS,
        patterns: CONTENT_PATTERNS,
        stopWords: CUSTOM_STOP_WORDS,
        synonyms: SEARCH_SYNONYMS
    };
}
/**
 * Interactive Decision Tree for CAIS Method Selection
 * Provides interactive tools and visualizations for understanding the decision process
 */

(function() {
    'use strict';

    // Decision tree state management
    let decisionState = {};
    let currentStep = 1;
    let maxSteps = 4;

    // Method information database
    const METHOD_INFO = {
        'instrumental_variables': {
            name: 'Instrumental Variables',
            category: 'quasi-experimental',
            priority: 1,
            assumptions: [
                'Instrument relevance (F-stat > 10)',
                'Exclusion restriction (instrument affects outcome only through treatment)',
                'Independence (instrument uncorrelated with unobserved confounders)'
            ],
            examples: [
                'Distance to college (affects education, not wages directly)',
                'Random encouragement in experiments',
                'Policy changes affecting treatment eligibility'
            ]
        },
        'linear_regression': {
            name: 'Linear Regression with Covariates',
            category: 'experimental',
            priority: 2,
            assumptions: [
                'Randomization validity (if RCT)',
                'Correct model specification',
                'Linear relationships between variables'
            ],
            examples: [
                'RCT with baseline measurements',
                'A/B test with user characteristics',
                'Clinical trial with patient covariates'
            ]
        },
        'difference_in_means': {
            name: 'Difference in Means',
            category: 'experimental',
            priority: 3,
            assumptions: [
                'Randomization validity',
                'No spillover effects',
                'Stable unit treatment value assumption (SUTVA)'
            ],
            examples: [
                'Pure A/B test without covariates',
                'Simple randomized experiment',
                'Clinical trial with binary outcome'
            ]
        },
        'difference_in_differences': {
            name: 'Difference-in-Differences',
            category: 'quasi-experimental',
            priority: 1,
            assumptions: [
                'Parallel trends between treatment and control groups',
                'No anticipation effects before treatment',
                'Stable composition of treatment and control groups'
            ],
            examples: [
                'Policy evaluation with panel data',
                'State-level interventions over time',
                'Before-after analysis with control group'
            ]
        },
        'regression_discontinuity': {
            name: 'Regression Discontinuity Design',
            category: 'quasi-experimental',
            priority: 2,
            assumptions: [
                'Continuity of potential outcomes at cutoff',
                'No manipulation of running variable around cutoff',
                'No other changes occurring precisely at cutoff'
            ],
            examples: [
                'College admission based on test score cutoff',
                'Program eligibility based on income threshold',
                'Age-based policy interventions'
            ]
        },
        'propensity_score_matching': {
            name: 'Propensity Score Matching',
            category: 'observational',
            priority: 1,
            assumptions: [
                'Unconfoundedness (no unmeasured confounders)',
                'Common support (sufficient overlap)',
                'Correct propensity score model specification'
            ],
            examples: [
                'Medical treatment effectiveness with patient records',
                'Job training program evaluation',
                'Educational intervention with rich student data'
            ]
        },
        'propensity_score_weighting': {
            name: 'Propensity Score Weighting',
            category: 'observational',
            priority: 2,
            assumptions: [
                'Unconfoundedness (no unmeasured confounders)',
                'Correct propensity score model specification',
                'Appropriate weight specification (ATE, ATT, etc.)'
            ],
            examples: [
                'Survey data with sampling weights',
                'Observational study with poor overlap',
                'Large-scale administrative data analysis'
            ]
        }
    };

    // Decision tree logic
    function getMethodRecommendation(state) {
        let method, justification, category;

        if (state.rct === 'yes') {
            // RCT branch
            if (state.instrument === 'yes') {
                method = 'instrumental_variables';
                justification = 'You have an RCT with an instrument different from treatment assignment. This encouragement design allows for strong causal identification.';
            } else if (state.covariates === 'yes') {
                method = 'linear_regression';
                justification = 'You have an RCT with covariates. Including covariates improves precision while maintaining causal identification from randomization.';
            } else {
                method = 'difference_in_means';
                justification = 'You have a pure RCT without covariates. Simple difference in means provides unbiased causal estimates.';
            }
        } else {
            // Observational branch
            if (state.structure === 'panel') {
                method = 'difference_in_differences';
                justification = 'You have panel data with treatment timing variation. DiD can control for time-invariant confounders.';
            } else if (state.structure === 'rdd') {
                method = 'regression_discontinuity';
                justification = 'You have a running variable with sharp cutoff. RDD provides credible causal identification around the cutoff.';
            } else if (state.instrument === 'yes') {
                method = 'instrumental_variables';
                justification = 'You have an instrumental variable available. IV can handle unmeasured confounding.';
            } else if (state.covariate_richness === 'rich') {
                if (state.overlap === 'good') {
                    method = 'propensity_score_matching';
                    justification = 'You have rich covariates with good overlap. Matching creates balanced comparison groups.';
                } else {
                    method = 'propensity_score_weighting';
                    justification = 'You have rich covariates but poor overlap. Weighting can handle limited overlap better than matching.';
                }
            } else {
                method = 'linear_regression';
                justification = 'You have limited covariates available. Linear regression provides a baseline approach with available controls.';
            }
        }

        const methodInfo = METHOD_INFO[method];
        return {
            method: methodInfo.name,
            category: methodInfo.category,
            priority: methodInfo.priority,
            justification: justification,
            assumptions: methodInfo.assumptions,
            examples: methodInfo.examples
        };
    }

    // Interactive decision tree functions
    function selectOption(key, value) {
        decisionState[key] = value;
        
        // Add visual feedback
        const button = event.target;
        const siblings = button.parentNode.querySelectorAll('.option-btn');
        siblings.forEach(btn => btn.classList.remove('selected'));
        button.classList.add('selected');
        
        updateDecisionPath();
        
        // Auto-advance after short delay
        setTimeout(() => {
            nextStep();
        }, 500);
    }

    function nextStep() {
        // Hide current step
        const currentStepEl = document.getElementById(`step-${currentStep}`);
        if (currentStepEl) {
            currentStepEl.style.display = 'none';
            currentStepEl.classList.remove('active');
        }

        currentStep++;

        // Show next step based on previous answers
        if (currentStep === 2) {
            showStep2();
        } else if (currentStep === 3) {
            showStep3();
        } else if (currentStep === 4) {
            showStep4();
        } else {
            showResults();
        }
    }

    function showStep2() {
        const step2 = document.getElementById('step-2');
        if (!step2) return;
        
        step2.style.display = 'block';
        step2.classList.add('active');

        if (decisionState.rct === 'yes') {
            const rctQuestions = document.getElementById('rct-questions');
            const obsQuestions = document.getElementById('obs-questions');
            if (rctQuestions) rctQuestions.style.display = 'block';
            if (obsQuestions) obsQuestions.style.display = 'none';
        } else {
            const rctQuestions = document.getElementById('rct-questions');
            const obsQuestions = document.getElementById('obs-questions');
            if (rctQuestions) rctQuestions.style.display = 'none';
            if (obsQuestions) obsQuestions.style.display = 'block';
        }
    }

    function showStep3() {
        const step3 = document.getElementById('step-3');
        if (!step3) return;
        
        step3.style.display = 'block';
        step3.classList.add('active');

        // Show relevant questions based on previous answers
        const instrumentQuestion = document.getElementById('instrument-question');
        const treatmentTypeQuestion = document.getElementById('treatment-type-question');
        
        if (decisionState.rct === 'yes' || decisionState.structure === 'cross') {
            if (instrumentQuestion) instrumentQuestion.style.display = 'block';
        }

        if (decisionState.structure === 'cross') {
            if (treatmentTypeQuestion) treatmentTypeQuestion.style.display = 'block';
        }

        // Skip to results if no questions needed
        if (decisionState.structure === 'panel' || decisionState.structure === 'rdd') {
            currentStep++;
            showResults();
            return;
        }
    }

    function showStep4() {
        const step4 = document.getElementById('step-4');
        
        // Only show step 4 for cross-sectional observational data
        if (decisionState.structure === 'cross' && decisionState.instrument === 'no') {
            if (step4) {
                step4.style.display = 'block';
                step4.classList.add('active');
            }
            
            const covariateRichnessQuestion = document.getElementById('covariate-richness-question');
            const overlapQuestion = document.getElementById('overlap-question');
            
            if (covariateRichnessQuestion) {
                covariateRichnessQuestion.style.display = 'block';
            }

            if (decisionState.covariate_richness === 'rich' && overlapQuestion) {
                overlapQuestion.style.display = 'block';
            }
        } else {
            currentStep++;
            showResults();
        }
    }

    function showResults() {
        const resultsStep = document.getElementById('results');
        if (!resultsStep) return;
        
        resultsStep.style.display = 'block';
        resultsStep.classList.add('active');

        const recommendation = getMethodRecommendation(decisionState);
        const recommendationEl = document.getElementById('method-recommendation');
        
        if (recommendationEl) {
            recommendationEl.innerHTML = generateRecommendationHTML(recommendation);
        }
    }

    function generateRecommendationHTML(recommendation) {
        const categoryClass = recommendation.category.replace('_', '-');
        
        return `
            <div class="method-result">
                <div class="method-name">
                    <h3>🎯 ${recommendation.method}</h3>
                    <span class="priority-badge ${categoryClass}">
                        ${recommendation.category.replace('_', ' ').toUpperCase()} - Priority ${recommendation.priority}
                    </span>
                </div>
                <div class="method-details">
                    <div class="justification">
                        <h4>Why this method?</h4>
                        <p>${recommendation.justification}</p>
                    </div>
                    <div class="assumptions">
                        <h4>Key Assumptions</h4>
                        <ul class="assumptions-list">
                            ${recommendation.assumptions.map(assumption => `<li>${assumption}</li>`).join('')}
                        </ul>
                    </div>
                    <div class="examples">
                        <h4>Typical Use Cases</h4>
                        <ul class="examples-list">
                            ${recommendation.examples.map(example => `<li class="positive">${example}</li>`).join('')}
                        </ul>
                    </div>
                </div>
            </div>
        `;
    }

    function updateDecisionPath() {
        const pathDisplay = document.getElementById('path-display');
        if (!pathDisplay) return;
        
        let pathSteps = [];

        if (decisionState.rct) {
            pathSteps.push(`Study Design: ${decisionState.rct === 'yes' ? 'Randomized Controlled Trial' : 'Observational Study'}`);
        }

        if (decisionState.covariates) {
            pathSteps.push(`Covariates: ${decisionState.covariates === 'yes' ? 'Available' : 'Not Available'}`);
        }

        if (decisionState.structure) {
            const structureMap = {
                'panel': 'Panel Data Structure',
                'rdd': 'Regression Discontinuity Design',
                'cross': 'Cross-sectional Data'
            };
            pathSteps.push(`Data Structure: ${structureMap[decisionState.structure]}`);
        }

        if (decisionState.instrument) {
            pathSteps.push(`Instrumental Variable: ${decisionState.instrument === 'yes' ? 'Available' : 'Not Available'}`);
        }

        if (decisionState.treatment_type) {
            pathSteps.push(`Treatment Type: ${decisionState.treatment_type.charAt(0).toUpperCase() + decisionState.treatment_type.slice(1)}`);
        }

        if (decisionState.covariate_richness) {
            pathSteps.push(`Covariate Richness: ${decisionState.covariate_richness.charAt(0).toUpperCase() + decisionState.covariate_richness.slice(1)}`);
        }

        if (decisionState.overlap) {
            pathSteps.push(`Covariate Overlap: ${decisionState.overlap.charAt(0).toUpperCase() + decisionState.overlap.slice(1)}`);
        }

        pathDisplay.innerHTML = pathSteps.length > 0 ? 
            `<ol>${pathSteps.map(step => `<li>${step}</li>`).join('')}</ol>` :
            '<p>Start answering questions to see your decision path...</p>';
    }

    function resetDecisionTree() {
        decisionState = {};
        currentStep = 1;

        // Hide all steps except first
        for (let i = 2; i <= maxSteps; i++) {
            const step = document.getElementById(`step-${i}`);
            if (step) {
                step.style.display = 'none';
                step.classList.remove('active');
            }
        }
        
        const results = document.getElementById('results');
        if (results) {
            results.style.display = 'none';
            results.classList.remove('active');
        }

        // Show first step
        const step1 = document.getElementById('step-1');
        if (step1) {
            step1.classList.add('active');
        }

        // Reset path display
        const pathDisplay = document.getElementById('path-display');
        if (pathDisplay) {
            pathDisplay.innerHTML = '<p>Start answering questions to see your decision path...</p>';
        }

        // Reset button states
        document.querySelectorAll('.option-btn').forEach(btn => {
            btn.classList.remove('selected');
        });

        // Hide all sub-questions
        const subQuestions = [
            'rct-questions', 'obs-questions', 'instrument-question', 
            'treatment-type-question', 'covariate-richness-question', 'overlap-question'
        ];
        
        subQuestions.forEach(id => {
            const element = document.getElementById(id);
            if (element) {
                element.style.display = 'none';
            }
        });
    }

    // Method comparison tool
    function createMethodComparisonTool() {
        const comparisonContainer = document.getElementById('method-comparison-tool');
        if (!comparisonContainer) return;

        const methods = Object.keys(METHOD_INFO);
        const comparisonHTML = `
            <div class="comparison-header">
                <h3>🔍 Method Comparison Tool</h3>
                <p>Compare different causal inference methods side by side:</p>
            </div>
            <div class="method-selector">
                <label for="method1">Method 1:</label>
                <select id="method1" onchange="updateComparison()">
                    <option value="">Select a method...</option>
                    ${methods.map(method => `<option value="${method}">${METHOD_INFO[method].name}</option>`).join('')}
                </select>
                
                <label for="method2">Method 2:</label>
                <select id="method2" onchange="updateComparison()">
                    <option value="">Select a method...</option>
                    ${methods.map(method => `<option value="${method}">${METHOD_INFO[method].name}</option>`).join('')}
                </select>
            </div>
            <div id="comparison-results"></div>
        `;
        
        comparisonContainer.innerHTML = comparisonHTML;
    }

    function updateComparison() {
        const method1 = document.getElementById('method1')?.value;
        const method2 = document.getElementById('method2')?.value;
        const resultsContainer = document.getElementById('comparison-results');
        
        if (!method1 || !method2 || !resultsContainer) return;
        
        const info1 = METHOD_INFO[method1];
        const info2 = METHOD_INFO[method2];
        
        resultsContainer.innerHTML = `
            <div class="comparison-grid">
                <div class="method-comparison-card ${info1.category}">
                    <h4>${info1.name}</h4>
                    <div class="method-category">Category: ${info1.category.replace('_', ' ')}</div>
                    <div class="method-priority">Priority: ${info1.priority}</div>
                    <div class="method-assumptions">
                        <h5>Assumptions:</h5>
                        <ul>
                            ${info1.assumptions.map(assumption => `<li>${assumption}</li>`).join('')}
                        </ul>
                    </div>
                </div>
                
                <div class="method-comparison-card ${info2.category}">
                    <h4>${info2.name}</h4>
                    <div class="method-category">Category: ${info2.category.replace('_', ' ')}</div>
                    <div class="method-priority">Priority: ${info2.priority}</div>
                    <div class="method-assumptions">
                        <h5>Assumptions:</h5>
                        <ul>
                            ${info2.assumptions.map(assumption => `<li>${assumption}</li>`).join('')}
                        </ul>
                    </div>
                </div>
            </div>
        `;
    }

    // Diagnostic tool for method validation
    function createDiagnosticTool() {
        const diagnosticContainer = document.getElementById('diagnostic-tool');
        if (!diagnosticContainer) return;

        diagnosticContainer.innerHTML = `
            <div class="diagnostic-header">
                <h3>🔧 Method Diagnostic Tool</h3>
                <p>Check if your data meets the assumptions for specific methods:</p>
            </div>
            <div class="diagnostic-form">
                <div class="form-group">
                    <label for="diagnostic-method">Select Method to Validate:</label>
                    <select id="diagnostic-method" onchange="showDiagnosticChecks()">
                        <option value="">Choose a method...</option>
                        ${Object.keys(METHOD_INFO).map(method => 
                            `<option value="${method}">${METHOD_INFO[method].name}</option>`
                        ).join('')}
                    </select>
                </div>
                <div id="diagnostic-checks"></div>
            </div>
        `;
    }

    function showDiagnosticChecks() {
        const selectedMethod = document.getElementById('diagnostic-method')?.value;
        const checksContainer = document.getElementById('diagnostic-checks');
        
        if (!selectedMethod || !checksContainer) return;
        
        const methodInfo = METHOD_INFO[selectedMethod];
        
        checksContainer.innerHTML = `
            <div class="diagnostic-checklist">
                <h4>Assumption Checklist for ${methodInfo.name}</h4>
                <p>Review each assumption and check if your data satisfies it:</p>
                ${methodInfo.assumptions.map((assumption, index) => `
                    <div class="assumption-check">
                        <input type="checkbox" id="assumption-${index}" />
                        <label for="assumption-${index}">${assumption}</label>
                        <div class="assumption-help">
                            <button type="button" onclick="showAssumptionHelp('${selectedMethod}', ${index})">
                                ❓ How to check this?
                            </button>
                        </div>
                    </div>
                `).join('')}
                <div class="diagnostic-result">
                    <button type="button" onclick="evaluateDiagnostic('${selectedMethod}')">
                        Evaluate Method Suitability
                    </button>
                </div>
            </div>
        `;
    }

    function evaluateDiagnostic(method) {
        const methodInfo = METHOD_INFO[method];
        const checkboxes = document.querySelectorAll('#diagnostic-checks input[type="checkbox"]');
        const checkedCount = Array.from(checkboxes).filter(cb => cb.checked).length;
        const totalCount = checkboxes.length;
        
        let resultHTML = '';
        let resultClass = '';
        
        if (checkedCount === totalCount) {
            resultHTML = `
                <div class="diagnostic-success">
                    ✅ <strong>Method Suitable</strong><br>
                    Your data appears to meet all key assumptions for ${methodInfo.name}.
                    You can proceed with confidence using this method.
                </div>
            `;
        } else if (checkedCount >= totalCount * 0.7) {
            resultHTML = `
                <div class="diagnostic-warning">
                    ⚠️ <strong>Method Partially Suitable</strong><br>
                    Your data meets most assumptions for ${methodInfo.name}, but some concerns remain.
                    Consider sensitivity analyses or alternative methods.
                </div>
            `;
        } else {
            resultHTML = `
                <div class="diagnostic-error">
                    ❌ <strong>Method Not Recommended</strong><br>
                    Your data does not meet key assumptions for ${methodInfo.name}.
                    Consider alternative methods or additional data collection.
                </div>
            `;
        }
        
        const resultContainer = document.querySelector('.diagnostic-result');
        if (resultContainer) {
            resultContainer.innerHTML += resultHTML;
        }
    }

    // Global functions for HTML onclick handlers
    window.selectOption = selectOption;
    window.resetDecisionTree = resetDecisionTree;
    window.updateComparison = updateComparison;
    window.showDiagnosticChecks = showDiagnosticChecks;
    window.evaluateDiagnostic = evaluateDiagnostic;

    // Initialize when DOM is ready
    function initializeDecisionTree() {
        // Create additional tools if containers exist
        createMethodComparisonTool();
        createDiagnosticTool();
        
        // Add keyboard navigation
        document.addEventListener('keydown', function(e) {
            if (e.key === 'Escape') {
                resetDecisionTree();
            }
        });
        
        console.log('Decision Tree Interactive Tools Initialized');
    }

    // Initialize when DOM is ready
    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', initializeDecisionTree);
    } else {
        initializeDecisionTree();
    }

})();
Method Selection Decision Tree
==============================

CAIS uses a sophisticated decision tree algorithm to automatically select the most appropriate causal inference method based on your data characteristics and research design. This page provides comprehensive documentation of the decision logic, interactive tools, and step-by-step walkthroughs to help you understand why CAIS recommends specific methods.

.. contents:: Table of Contents
   :local:
   :depth: 3

Overview of the Decision Process
--------------------------------

The CAIS decision tree follows a hierarchical approach that prioritizes methods with stronger causal identification:

1. **Experimental Methods** (Highest Priority): When randomization is present
2. **Quasi-Experimental Methods** (High Priority): Natural experiments and design-based identification
3. **Observational Methods** (Medium Priority): Statistical adjustment methods

The algorithm considers multiple data characteristics simultaneously and provides both primary recommendations and alternative methods for robustness checking.

Complete Decision Tree Algorithm
--------------------------------

.. mermaid::

   flowchart TD
       A[Start: Causal Analysis] --> B{Is this a randomized<br/>controlled trial?}
       
       %% RCT Branch
       B -->|Yes| C{Are covariates<br/>available?}
       C -->|Yes| D{Is there an instrument<br/>different from treatment?}
       C -->|No| E{Is there an instrument<br/>different from treatment?}
       
       D -->|Yes| F[Instrumental Variables<br/>Encouragement Design<br/>🏆 Priority: 1]
       D -->|No| G[Linear Regression<br/>with Covariates<br/>🏆 Priority: 2]
       
       E -->|Yes| F
       E -->|No| H[Difference in Means<br/>Pure RCT<br/>🏆 Priority: 3]
       
       %% Observational Branch
       B -->|No| I{What is the data<br/>structure?}
       
       %% Temporal Structure Check
       I --> J{Panel data with<br/>treatment timing variation?}
       J -->|Yes| K[Difference-in-Differences<br/>Check parallel trends<br/>🥈 Priority: 1]
       
       %% Discontinuity Check
       I --> L{Running variable<br/>with sharp cutoff?}
       L -->|Yes| M[Regression Discontinuity<br/>Check continuity<br/>🥈 Priority: 2]
       
       %% Cross-sectional Analysis
       I --> N{Cross-sectional<br/>observational data?}
       N -->|Yes| O{What is treatment<br/>variable type?}
       
       %% Binary Treatment Path
       O -->|Binary| P{Instrumental variable<br/>available?}
       P -->|Yes| Q[Instrumental Variables<br/>Binary Treatment<br/>🥈 Priority: 3]
       P -->|No| R{Rich covariates<br/>available?}
       
       R -->|Yes| S{Covariate overlap<br/>assessment}
       S -->|Good overlap<br/>score ≥ 0.1| T[Propensity Score<br/>Matching<br/>🥉 Priority: 1]
       S -->|Poor overlap<br/>score < 0.1| U[Propensity Score<br/>Weighting<br/>🥉 Priority: 2]
       
       R -->|No| V[Linear Regression<br/>with Available Controls<br/>🥉 Priority: 4]
       
       %% Continuous Treatment Path
       O -->|Continuous| W{Instrumental variable<br/>available?}
       W -->|Yes| X[Instrumental Variables<br/>Continuous Treatment<br/>🥈 Priority: 3]
       W -->|No| Y{Rich covariates<br/>available?}
       
       Y -->|Yes| Z[Generalized Propensity Score<br/>Continuous Treatment<br/>🥉 Priority: 3]
       Y -->|No| AA[Linear Regression<br/>Continuous Treatment<br/>🥉 Priority: 4]
       
       %% Categorical Treatment Path
       O -->|Categorical| BB{Instrumental variable<br/>available?}
       BB -->|Yes| CC[Instrumental Variables<br/>Multiple Treatments<br/>🥈 Priority: 3]
       BB -->|No| DD[Multinomial Methods<br/>Multiple Treatments<br/>🥉 Priority: 5]
       
       %% Special Cases
       I --> EE{Front-door criterion<br/>satisfied?}
       EE -->|Yes| FF[Front-door Adjustment<br/>Mediation Analysis<br/>🥈 Priority: 4]
       
       %% Styling
       classDef experimental fill:#e1f5fe,stroke:#01579b,stroke-width:3px
       classDef quasiExp fill:#fff3e0,stroke:#e65100,stroke-width:3px
       classDef observational fill:#f3e5f5,stroke:#4a148c,stroke-width:2px
       classDef decision fill:#f5f5f5,stroke:#424242,stroke-width:2px
       classDef special fill:#e8f5e8,stroke:#2e7d32,stroke-width:2px
       
       class F,G,H experimental
       class K,M,Q,X,CC,FF quasiExp
       class T,U,V,Z,AA,DD observational
       class B,C,D,E,I,J,L,N,O,P,R,S,W,Y,BB,EE decision
       class A special

Interactive Method Selection Tool
---------------------------------

Use this interactive tool to walk through the decision tree logic with your own data characteristics:

.. raw:: html

   <div id="interactive-decision-tree">
       <div class="decision-tool-header">
           <h3>🔍 Interactive Method Selection</h3>
           <p>Answer the questions below to see how CAIS would select a method for your data:</p>
       </div>
       
       <div class="decision-steps">
           <div class="step active" id="step-1">
               <h4>Step 1: Study Design</h4>
               <div class="question">
                   <label>Is your data from a randomized controlled trial (RCT)?</label>
                   <div class="options">
                       <button class="option-btn" data-value="yes" onclick="selectOption('rct', 'yes')">
                           ✅ Yes - Participants were randomly assigned to treatment/control
                       </button>
                       <button class="option-btn" data-value="no" onclick="selectOption('rct', 'no')">
                           ❌ No - Observational data or natural experiment
                       </button>
                   </div>
               </div>
           </div>
           
           <div class="step" id="step-2" style="display: none;">
               <h4>Step 2: Data Structure</h4>
               <div class="question" id="rct-questions" style="display: none;">
                   <label>Do you have covariates (pre-treatment variables)?</label>
                   <div class="options">
                       <button class="option-btn" data-value="yes" onclick="selectOption('covariates', 'yes')">
                           ✅ Yes - I have background variables
                       </button>
                       <button class="option-btn" data-value="no" onclick="selectOption('covariates', 'no')">
                           ❌ No - Only treatment and outcome
                       </button>
                   </div>
               </div>
               
               <div class="question" id="obs-questions" style="display: none;">
                   <label>What type of data structure do you have?</label>
                   <div class="options">
                       <button class="option-btn" data-value="panel" onclick="selectOption('structure', 'panel')">
                           📊 Panel data - Multiple time periods
                       </button>
                       <button class="option-btn" data-value="rdd" onclick="selectOption('structure', 'rdd')">
                           📈 Running variable - Sharp cutoff for treatment
                       </button>
                       <button class="option-btn" data-value="cross" onclick="selectOption('structure', 'cross')">
                           📋 Cross-sectional - Single time period
                       </button>
                   </div>
               </div>
           </div>
           
           <div class="step" id="step-3" style="display: none;">
               <h4>Step 3: Additional Variables</h4>
               <div class="question" id="instrument-question" style="display: none;">
                   <label>Do you have an instrumental variable?</label>
                   <div class="options">
                       <button class="option-btn" data-value="yes" onclick="selectOption('instrument', 'yes')">
                           ✅ Yes - Variable that affects treatment but not outcome directly
                       </button>
                       <button class="option-btn" data-value="no" onclick="selectOption('instrument', 'no')">
                           ❌ No - No valid instrument available
                       </button>
                   </div>
               </div>
               
               <div class="question" id="treatment-type-question" style="display: none;">
                   <label>What type of treatment variable do you have?</label>
                   <div class="options">
                       <button class="option-btn" data-value="binary" onclick="selectOption('treatment_type', 'binary')">
                           🔘 Binary - Treatment vs Control (0/1)
                       </button>
                       <button class="option-btn" data-value="continuous" onclick="selectOption('treatment_type', 'continuous')">
                           📏 Continuous - Dose or intensity level
                       </button>
                       <button class="option-btn" data-value="categorical" onclick="selectOption('treatment_type', 'categorical')">
                           📂 Categorical - Multiple treatment types
                       </button>
                   </div>
               </div>
           </div>
           
           <div class="step" id="step-4" style="display: none;">
               <h4>Step 4: Covariate Assessment</h4>
               <div class="question" id="covariate-richness-question" style="display: none;">
                   <label>How many pre-treatment covariates do you have?</label>
                   <div class="options">
                       <button class="option-btn" data-value="rich" onclick="selectOption('covariate_richness', 'rich')">
                           📚 Rich - Many relevant background variables
                       </button>
                       <button class="option-btn" data-value="limited" onclick="selectOption('covariate_richness', 'limited')">
                           📖 Limited - Few background variables
                       </button>
                   </div>
               </div>
               
               <div class="question" id="overlap-question" style="display: none;">
                   <label>How is the covariate overlap between treatment groups?</label>
                   <div class="options">
                       <button class="option-btn" data-value="good" onclick="selectOption('overlap', 'good')">
                           ✅ Good - Similar distributions across groups
                       </button>
                       <button class="option-btn" data-value="poor" onclick="selectOption('overlap', 'poor')">
                           ⚠️ Poor - Different distributions across groups
                       </button>
                   </div>
               </div>
           </div>
           
           <div class="step" id="results" style="display: none;">
               <h4>🎯 Recommended Method</h4>
               <div id="method-recommendation">
                   <!-- Results will be populated here -->
               </div>
               <button class="reset-btn" onclick="resetDecisionTree()">🔄 Start Over</button>
           </div>
       </div>
       
       <div class="decision-path">
           <h4>📍 Your Decision Path</h4>
           <div id="path-display">
               <p>Start answering questions to see your decision path...</p>
           </div>
       </div>
   </div>

   <script>
   // Interactive Decision Tree Logic
   let decisionState = {};
   let currentStep = 1;
   
   function selectOption(key, value) {
       decisionState[key] = value;
       updateDecisionPath();
       nextStep();
   }
   
   function nextStep() {
       // Hide current step
       document.getElementById(`step-${currentStep}`).style.display = 'none';
       document.getElementById(`step-${currentStep}`).classList.remove('active');
       
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
       step2.style.display = 'block';
       step2.classList.add('active');
       
       if (decisionState.rct === 'yes') {
           document.getElementById('rct-questions').style.display = 'block';
           document.getElementById('obs-questions').style.display = 'none';
       } else {
           document.getElementById('rct-questions').style.display = 'none';
           document.getElementById('obs-questions').style.display = 'block';
       }
   }
   
   function showStep3() {
       const step3 = document.getElementById('step-3');
       step3.style.display = 'block';
       step3.classList.add('active');
       
       // Show relevant questions based on previous answers
       if (decisionState.rct === 'yes' || decisionState.structure === 'cross') {
           document.getElementById('instrument-question').style.display = 'block';
       }
       
       if (decisionState.structure === 'cross') {
           document.getElementById('treatment-type-question').style.display = 'block';
       }
       
       // Skip to step 4 if no questions needed
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
           step4.style.display = 'block';
           step4.classList.add('active');
           document.getElementById('covariate-richness-question').style.display = 'block';
           
           if (decisionState.covariate_richness === 'rich') {
               document.getElementById('overlap-question').style.display = 'block';
           }
       } else {
           currentStep++;
           showResults();
       }
   }
   
   function showResults() {
       const resultsStep = document.getElementById('results');
       resultsStep.style.display = 'block';
       resultsStep.classList.add('active');
       
       const recommendation = getMethodRecommendation();
       document.getElementById('method-recommendation').innerHTML = recommendation;
   }
   
   function getMethodRecommendation() {
       let method, justification, assumptions, alternatives;
       
       if (decisionState.rct === 'yes') {
           if (decisionState.instrument === 'yes') {
               method = 'Instrumental Variables (Encouragement Design)';
               justification = 'You have an RCT with an instrument different from treatment assignment. This encouragement design allows for strong causal identification.';
               assumptions = ['Instrument relevance', 'Exclusion restriction', 'Monotonicity'];
               alternatives = ['Linear Regression with Covariates'];
           } else if (decisionState.covariates === 'yes') {
               method = 'Linear Regression with Covariates';
               justification = 'You have an RCT with covariates. Including covariates improves precision while maintaining causal identification from randomization.';
               assumptions = ['Randomization validity', 'Correct model specification'];
               alternatives = ['Difference in Means'];
           } else {
               method = 'Difference in Means';
               justification = 'You have a pure RCT without covariates. Simple difference in means provides unbiased causal estimates.';
               assumptions = ['Randomization validity', 'No spillover effects'];
               alternatives = ['Linear Regression'];
           }
       } else {
           if (decisionState.structure === 'panel') {
               method = 'Difference-in-Differences';
               justification = 'You have panel data with treatment timing variation. DiD can control for time-invariant confounders.';
               assumptions = ['Parallel trends', 'No anticipation effects', 'Stable composition'];
               alternatives = ['Linear Regression with Fixed Effects'];
           } else if (decisionState.structure === 'rdd') {
               method = 'Regression Discontinuity Design';
               justification = 'You have a running variable with sharp cutoff. RDD provides credible causal identification around the cutoff.';
               assumptions = ['Continuity at cutoff', 'No manipulation of running variable'];
               alternatives = ['Linear Regression'];
           } else if (decisionState.instrument === 'yes') {
               method = 'Instrumental Variables';
               justification = 'You have an instrumental variable available. IV can handle unmeasured confounding.';
               assumptions = ['Instrument relevance', 'Exclusion restriction', 'Independence'];
               alternatives = ['Propensity Score Methods'];
           } else if (decisionState.covariate_richness === 'rich') {
               if (decisionState.overlap === 'good') {
                   method = 'Propensity Score Matching';
                   justification = 'You have rich covariates with good overlap. Matching creates balanced comparison groups.';
                   assumptions = ['Unconfoundedness', 'Common support', 'Correct propensity score model'];
                   alternatives = ['Propensity Score Weighting'];
               } else {
                   method = 'Propensity Score Weighting';
                   justification = 'You have rich covariates but poor overlap. Weighting can handle limited overlap better than matching.';
                   assumptions = ['Unconfoundedness', 'Correct propensity score model', 'Appropriate weights'];
                   alternatives = ['Linear Regression with Controls'];
               }
           } else {
               method = 'Linear Regression with Controls';
               justification = 'You have limited covariates available. Linear regression provides a baseline approach with available controls.';
               assumptions = ['No unmeasured confounders', 'Correct model specification', 'Linear relationships'];
               alternatives = ['Correlation Analysis'];
           }
       }
       
       return `
           <div class="method-result">
               <div class="method-name">
                   <h3>🎯 ${method}</h3>
               </div>
               <div class="method-details">
                   <div class="justification">
                       <h4>Why this method?</h4>
                       <p>${justification}</p>
                   </div>
                   <div class="assumptions">
                       <h4>Key Assumptions</h4>
                       <ul>
                           ${assumptions.map(assumption => `<li>${assumption}</li>`).join('')}
                       </ul>
                   </div>
                   <div class="alternatives">
                       <h4>Alternative Methods</h4>
                       <p>${alternatives.join(', ')}</p>
                   </div>
               </div>
           </div>
       `;
   }
   
   function updateDecisionPath() {
       const pathDisplay = document.getElementById('path-display');
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
       
       // Hide all steps
       for (let i = 2; i <= 4; i++) {
           document.getElementById(`step-${i}`).style.display = 'none';
           document.getElementById(`step-${i}`).classList.remove('active');
       }
       document.getElementById('results').style.display = 'none';
       document.getElementById('results').classList.remove('active');
       
       // Show first step
       document.getElementById('step-1').classList.add('active');
       
       // Reset path display
       document.getElementById('path-display').innerHTML = '<p>Start answering questions to see your decision path...</p>';
       
       // Reset button states
       document.querySelectorAll('.option-btn').forEach(btn => {
           btn.classList.remove('selected');
       });
   }
   
   // Add click styling
   document.addEventListener('click', function(e) {
       if (e.target.classList.contains('option-btn')) {
           // Remove selected class from siblings
           e.target.parentNode.querySelectorAll('.option-btn').forEach(btn => {
               btn.classList.remove('selected');
           });
           // Add selected class to clicked button
           e.target.classList.add('selected');
       }
   });
   </script>

   <style>
   #interactive-decision-tree {
       background: #f8f9fa;
       border: 1px solid #dee2e6;
       border-radius: 8px;
       padding: 20px;
       margin: 20px 0;
   }
   
   .decision-tool-header {
       text-align: center;
       margin-bottom: 30px;
   }
   
   .decision-tool-header h3 {
       color: #2c3e50;
       margin-bottom: 10px;
   }
   
   .step {
       background: white;
       border: 1px solid #e9ecef;
       border-radius: 6px;
       padding: 20px;
       margin-bottom: 20px;
   }
   
   .step.active {
       border-color: #007bff;
       box-shadow: 0 0 0 0.2rem rgba(0,123,255,.25);
   }
   
   .step h4 {
       color: #495057;
       margin-bottom: 15px;
       border-bottom: 2px solid #e9ecef;
       padding-bottom: 10px;
   }
   
   .question {
       margin-bottom: 20px;
   }
   
   .question label {
       display: block;
       font-weight: 600;
       margin-bottom: 15px;
       color: #343a40;
   }
   
   .options {
       display: flex;
       flex-direction: column;
       gap: 10px;
   }
   
   .option-btn {
       background: white;
       border: 2px solid #dee2e6;
       border-radius: 6px;
       padding: 15px;
       text-align: left;
       cursor: pointer;
       transition: all 0.2s ease;
       font-size: 14px;
   }
   
   .option-btn:hover {
       border-color: #007bff;
       background: #f8f9ff;
   }
   
   .option-btn.selected {
       border-color: #007bff;
       background: #e7f3ff;
       color: #0056b3;
   }
   
   .decision-path {
       background: white;
       border: 1px solid #e9ecef;
       border-radius: 6px;
       padding: 20px;
       margin-top: 20px;
   }
   
   .decision-path h4 {
       color: #495057;
       margin-bottom: 15px;
   }
   
   .decision-path ol {
       margin: 0;
       padding-left: 20px;
   }
   
   .decision-path li {
       margin-bottom: 8px;
       color: #6c757d;
   }
   
   .method-result {
       background: white;
       border: 1px solid #28a745;
       border-radius: 6px;
       padding: 20px;
   }
   
   .method-name h3 {
       color: #28a745;
       margin-bottom: 20px;
   }
   
   .method-details > div {
       margin-bottom: 20px;
   }
   
   .method-details h4 {
       color: #495057;
       margin-bottom: 10px;
       font-size: 16px;
   }
   
   .justification p {
       color: #6c757d;
       line-height: 1.6;
   }
   
   .assumptions ul {
       margin: 0;
       padding-left: 20px;
   }
   
   .assumptions li {
       color: #6c757d;
       margin-bottom: 5px;
   }
   
   .alternatives p {
       color: #6c757d;
       font-style: italic;
   }
   
   .reset-btn {
       background: #6c757d;
       color: white;
       border: none;
       border-radius: 4px;
       padding: 10px 20px;
       cursor: pointer;
       margin-top: 20px;
   }
   
   .reset-btn:hover {
       background: #5a6268;
   }
   
   @media (max-width: 768px) {
       #interactive-decision-tree {
           padding: 15px;
       }
       
       .step {
           padding: 15px;
       }
       
       .option-btn {
           padding: 12px;
           font-size: 13px;
       }
   }
   </style>

Dataset Property Influence Visualization
----------------------------------------

The following diagram shows how different dataset properties influence method selection:

.. mermaid::

   graph LR
       subgraph "Dataset Properties"
           A[Randomization Status]
           B[Temporal Structure]
           C[Running Variable]
           D[Instrumental Variable]
           E[Treatment Type]
           F[Covariate Richness]
           G[Covariate Overlap]
       end
       
       subgraph "Method Categories"
           H[Experimental Methods]
           I[Quasi-Experimental]
           J[Observational Methods]
       end
       
       subgraph "Specific Methods"
           K[Difference in Means]
           L[Linear Regression]
           M[Instrumental Variables]
           N[Difference-in-Differences]
           O[Regression Discontinuity]
           P[Propensity Score Matching]
           Q[Propensity Score Weighting]
       end
       
       A -->|RCT = Yes| H
       A -->|RCT = No| I
       A -->|RCT = No| J
       
       B -->|Panel Data| N
       C -->|Sharp Cutoff| O
       D -->|Available| M
       
       E -->|Binary| P
       E -->|Binary| Q
       E -->|Continuous| R[Generalized PS]
       
       F -->|Rich| P
       F -->|Rich| Q
       F -->|Limited| L
       
       G -->|Good| P
       G -->|Poor| Q
       
       H --> K
       H --> L
       H --> M
       
       I --> N
       I --> O
       I --> M
       
       J --> P
       J --> Q
       J --> L

Decision Criteria Explained
---------------------------

1. Randomized Experiment Check
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Question**: Is this data from a randomized controlled trial?

**Why it matters**: Randomization is the gold standard for causal inference because it eliminates confounding by design. If you have randomized data, you can use simpler methods with stronger causal identification.

**How CAIS detects**:
- User specification of ``is_rct=True``
- Analysis of treatment assignment patterns
- Detection of balanced covariates across treatment groups

**Next steps**:
- **Yes** → Use experimental methods (RCT analysis)
- **No** → Continue to observational methods

2. Data Structure Analysis
~~~~~~~~~~~~~~~~~~~~~~~~~~

**Panel Data with Treatment Timing**

**Question**: Do you have repeated observations over time with variation in when units receive treatment?

**Indicators**:
- Time variable present
- Treatment varies within units over time
- Clear before/after treatment periods

**Method**: Difference-in-Differences (DiD)
**Key assumption**: Parallel trends between treatment and control groups

**Running Variable with Cutoff**

**Question**: Is treatment assigned based on a continuous variable crossing a threshold?

**Indicators**:
- Continuous assignment variable (running variable)
- Sharp cutoff determining treatment
- Units just above/below cutoff are similar

**Method**: Regression Discontinuity Design (RDD)
**Key assumption**: Continuity of potential outcomes at cutoff

3. Instrumental Variable Assessment
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Question**: Is there a variable that affects treatment assignment but not the outcome directly?

**Valid instruments must satisfy**:
1. **Relevance**: Instrument predicts treatment assignment
2. **Exclusion**: Instrument affects outcome only through treatment
3. **Exogeneity**: Instrument is uncorrelated with unobserved confounders

**Common instruments**:
- Policy changes affecting treatment eligibility
- Random encouragement in experiments
- Geographic or temporal variation in treatment access

**Method**: Instrumental Variables (IV)
**Strength**: Can handle unmeasured confounding

4. Treatment Variable Type
~~~~~~~~~~~~~~~~~~~~~~~~~~

**Binary Treatment**

Most causal inference methods are designed for binary (0/1) treatments:
- Propensity score methods
- Most matching approaches
- Standard difference-in-differences

**Continuous Treatment**

Requires specialized methods:
- Generalized propensity score
- Dose-response functions
- Instrumental variables with continuous endogenous variables

**Categorical Treatment**

Multiple treatment levels:
- Multinomial propensity scores
- Multiple treatment IV
- Generalized difference-in-differences

5. Covariate Assessment
~~~~~~~~~~~~~~~~~~~~~~~

**Rich Covariates Available**

**Question**: Do you have many pre-treatment variables that predict both treatment and outcome?

**Propensity Score Methods**:
- **Matching**: Pair similar units with different treatments
- **Weighting**: Weight observations to balance covariate distributions
- **Stratification**: Analyze within covariate strata

**Covariate Overlap Check**:
- **Good overlap** → Propensity Score Matching preferred
- **Poor overlap** → Propensity Score Weighting or trimming

**Limited Covariates**

**Method**: Linear regression with available controls
**Assumption**: No unmeasured confounders (strong assumption)

Step-by-Step Decision Walkthroughs
-----------------------------------

This section provides detailed walkthroughs of the decision process for different types of datasets, showing exactly how CAIS analyzes data characteristics and selects methods.

Walkthrough 1: Randomized Controlled Trial
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Scenario**: A/B test for a new website feature

**Dataset Characteristics**:
- 10,000 users randomly assigned to treatment (new feature) or control (old feature)
- Outcome: conversion rate
- Available covariates: user age, previous purchases, account type

**Decision Process**:

.. mermaid::

   flowchart TD
       A[Dataset Analysis Begins] --> B{Is this an RCT?}
       B -->|✅ Yes - Random assignment confirmed| C{Are covariates available?}
       C -->|✅ Yes - Age, purchases, account type| D{Is there an instrument?}
       D -->|❌ No - Treatment assignment is the intervention| E[Selected: Linear Regression with Covariates]
       
       style A fill:#e8f5e8
       style E fill:#e1f5fe
       
       F[Decision Reasoning:<br/>• Randomization ensures causal identification<br/>• Covariates improve precision<br/>• No instrument needed with direct randomization]
       E --> F

**Step-by-step Analysis**:

1. **Randomization Check**: ✅ PASS
   - Users were randomly assigned using a randomization algorithm
   - Treatment assignment is independent of user characteristics
   - **Result**: Experimental methods are available

2. **Covariate Assessment**: ✅ AVAILABLE
   - User demographics and behavior history collected
   - Pre-treatment variables that can improve precision
   - **Result**: Include covariates in analysis

3. **Instrument Assessment**: ❌ NOT APPLICABLE
   - Treatment assignment itself is the intervention
   - No separate encouragement or instrument needed
   - **Result**: Direct treatment effect estimation

4. **Final Selection**: **Linear Regression with Covariates**
   - **Priority Score**: 2 (High - Experimental method)
   - **Justification**: "RCT with covariates—use OLS for precision"
   - **Assumptions**: Randomization validity, correct model specification

**Alternative Methods Considered**:
- Difference in Means (lower precision without covariates)
- Propensity Score methods (unnecessary with randomization)

Walkthrough 2: Panel Data Analysis
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Scenario**: State-level policy evaluation

**Dataset Characteristics**:
- 50 states observed over 10 years (2010-2020)
- Treatment: Policy implemented in different states at different times
- Outcome: Economic indicator (e.g., unemployment rate)
- Time-varying covariates: GDP, population, other policies

**Decision Process**:

.. mermaid::

   flowchart TD
       A[Dataset Analysis Begins] --> B{Is this an RCT?}
       B -->|❌ No - Observational policy data| C{What data structure?}
       C -->|✅ Panel - States × Years| D{Treatment timing varies?}
       D -->|✅ Yes - Staggered implementation| E[Selected: Difference-in-Differences]
       
       style A fill:#e8f5e8
       style E fill:#fff3e0
       
       F[Decision Reasoning:<br/>• Panel structure with treatment timing<br/>• Can control for state and time fixed effects<br/>• Parallel trends assumption testable]
       E --> F

**Step-by-step Analysis**:

1. **Study Design Check**: ❌ NOT RCT
   - Policy implementation was not randomized
   - States chose when to implement based on political/economic factors
   - **Result**: Observational methods required

2. **Data Structure Assessment**: ✅ PANEL DATA
   - Multiple units (states) observed over time
   - Treatment varies within units over time
   - Clear before/after periods for each state
   - **Result**: Temporal methods available

3. **Treatment Timing Check**: ✅ STAGGERED
   - Different states implemented at different times
   - Creates natural comparison groups
   - **Result**: Difference-in-Differences applicable

4. **Parallel Trends Assessment**: ⚠️ REQUIRES TESTING
   - Key assumption: treated and control states would follow similar trends
   - Can be tested using pre-treatment periods
   - **Result**: Diagnostic tests required

5. **Final Selection**: **Difference-in-Differences**
   - **Priority Score**: 1 (High - Quasi-experimental)
   - **Justification**: "Temporal structure via time variable—consider DiD"
   - **Assumptions**: Parallel trends, no anticipation, stable composition

**Alternative Methods Considered**:
- Linear regression with fixed effects (similar but less robust)
- Synthetic control (if few treated units)

**Diagnostic Tests Required**:
- Parallel trends test using pre-treatment periods
- Event study plots to check for anticipation effects
- Balance tests for time-varying covariates

Walkthrough 6: Complex Multi-Treatment Scenario
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Scenario**: Educational intervention with multiple treatment arms

**Dataset Characteristics**:
- 15,000 students across 200 schools
- Three treatment conditions: online tutoring, in-person tutoring, hybrid approach
- Control group receives standard instruction
- Rich student and school-level covariates
- Outcome: standardized test scores

**Decision Process**:

.. mermaid::

   flowchart TD
       A[Dataset Analysis Begins] --> B{Is this an RCT?}
       B -->|❌ No - Schools self-selected programs| C{What data structure?}
       C -->|Cross-sectional with multiple treatments| D{Treatment type?}
       D -->|Categorical - 4 treatment arms| E{Instrumental variable?}
       E -->|❌ No - No valid instrument identified| F{Rich covariates?}
       F -->|✅ Yes - Student and school characteristics| G[Selected: Multinomial Propensity Score Methods]
       
       style A fill:#e8f5e8
       style G fill:#f3e5f5
       
       H[Decision Reasoning:<br/>• Multiple treatment categories require specialized methods<br/>• Rich covariates enable propensity score approach<br/>• Multinomial treatment assignment modeling needed]
       G --> H

**Step-by-step Analysis**:

1. **Study Design Check**: ❌ NOT RCT
   - Schools chose their preferred intervention approach
   - Selection based on resources, preferences, student needs
   - **Result**: Confounding likely present

2. **Treatment Type Assessment**: ✅ CATEGORICAL (4 LEVELS)
   - Control, online tutoring, in-person tutoring, hybrid
   - No natural ordering between treatment types
   - **Result**: Multinomial treatment methods required

3. **Covariate Assessment**: ✅ RICH COVARIATES
   - Student demographics, prior achievement, socioeconomic status
   - School characteristics, resources, teacher quality
   - **Result**: Sufficient variables for adjustment

4. **Method Selection Logic**:
   - Standard propensity score methods designed for binary treatment
   - Multinomial logistic regression needed for treatment assignment
   - Generalized propensity score approach required

5. **Final Selection**: **Multinomial Propensity Score Methods**
   - **Priority Score**: 5 (Medium - Observational, complex)
   - **Justification**: "Multiple treatment categories with rich covariates"
   - **Implementation**: Multinomial logistic regression → inverse probability weighting

**Special Considerations**:
- Balance checking across all treatment pairs
- Common support assessment for each treatment comparison
- Multiple comparison adjustments for pairwise effects

Walkthrough 7: Weak Instrument Scenario
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Scenario**: Returns to education with questionable instrument

**Dataset Characteristics**:
- Survey data on wages and education (n=5,000)
- Proposed instrument: Month of birth (affects school starting age)
- Outcome: Log hourly wages
- Treatment: Years of education (continuous)

**Decision Process**:

.. mermaid::

   flowchart TD
       A[Dataset Analysis Begins] --> B{Is this an RCT?}
       B -->|❌ No - Observational survey data| C{Treatment type?}
       C -->|Continuous - Years of education| D{Instrumental variable?}
       D -->|⚠️ Potentially - Month of birth| E{Instrument validation}
       E -->|❌ Fails relevance test F < 10| F{Rich covariates?}
       F -->|✅ Yes - Demographics, family background| G[Selected: Linear Regression with Controls]
       
       style A fill:#e8f5e8
       style G fill:#f3e5f5
       
       H[Decision Reasoning:<br/>• Weak instrument fails first-stage test<br/>• IV methods inappropriate with F < 10<br/>• Fallback to regression with available controls]
       G --> H

**Instrument Validation Process**:

1. **Relevance Test**: ❌ FAILED
   - First-stage F-statistic: 3.2 (< 10 threshold)
   - Month of birth weakly predicts education
   - **Result**: Weak instrument problem

2. **Exclusion Restriction**: ⚠️ QUESTIONABLE
   - Month of birth might affect wages through other channels
   - Age effects, seasonal labor market conditions
   - **Result**: Exclusion restriction violated

3. **Independence Test**: ✅ LIKELY SATISFIED
   - Month of birth appears random
   - No correlation with family background
   - **Result**: Independence assumption met

4. **Overall Assessment**: ❌ INVALID INSTRUMENT
   - Weak relevance dominates other considerations
   - IV estimates would be severely biased
   - **Result**: Reject instrumental variable approach

**Final Method Selection**: **Linear Regression with Controls**
- Acknowledge limitations of causal interpretation
- Include extensive controls for ability proxies
- Sensitivity analysis with different specifications
- Transparent reporting of assumptions

Edge Cases and Troubleshooting
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Case 1: Perfect Separation in Propensity Scores**

*Problem*: Some covariate combinations perfectly predict treatment assignment

*Detection*: Propensity scores of exactly 0 or 1 for some observations

*Solution*: 
- Switch from matching to weighting with trimming
- Use regularized propensity score models (LASSO, Ridge)
- Consider covariate balancing propensity scores

**Case 2: Insufficient Temporal Variation for DiD**

*Problem*: Treatment occurs simultaneously across all units

*Detection*: No staggered treatment timing in panel data

*Solution*:
- Cannot use difference-in-differences
- Fall back to cross-sectional methods
- Consider synthetic control if few treated units

**Case 3: Discontinuous Covariates at RDD Cutoff**

*Problem*: Other variables jump discontinuously at the cutoff

*Detection*: Significant discontinuities in covariates at threshold

*Solution*:
- Include discontinuous covariates as controls
- Use local linear regression with narrow bandwidth
- Consider alternative identification strategies

**Case 4: Multiple Instruments with Conflicting Results**

*Problem*: Different instruments give different treatment effect estimates

*Detection*: Overidentification tests reject null hypothesis

*Solution*:
- Test each instrument individually for validity
- Use robust inference methods (Anderson-Rubin)
- Report results from most credible instrument

**Case 5: Time-Varying Treatment Intensity**

*Problem*: Treatment intensity changes over time within units

*Detection*: Continuous treatment variable varies within units over time

*Solution*:
- Use dose-response DiD methods
- Model treatment intensity explicitly
- Consider dynamic treatment effect models

Algorithm Robustness and Validation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Cross-Validation of Method Selection**:

The decision tree algorithm includes built-in validation:

.. code-block:: python

   def validate_method_selection(data, selected_method, alternatives):
       """Validate method selection through cross-checks"""
       
       validation_results = {}
       
       # 1. Assumption checking
       assumptions_met = check_method_assumptions(data, selected_method)
       validation_results['assumptions'] = assumptions_met
       
       # 2. Alternative method comparison
       alternative_estimates = []
       for method in alternatives:
           if method_applicable(data, method):
               estimate = run_method(data, method)
               alternative_estimates.append((method, estimate))
       
       validation_results['alternatives'] = alternative_estimates
       
       # 3. Sensitivity analysis
       sensitivity_results = run_sensitivity_analysis(data, selected_method)
       validation_results['sensitivity'] = sensitivity_results
       
       return validation_results

**Confidence Scoring**:

Each method recommendation includes a confidence score:

- **High Confidence (90-100%)**: Strong identification, assumptions clearly met
- **Medium Confidence (70-89%)**: Good identification, some assumption concerns
- **Low Confidence (50-69%)**: Weak identification, major assumption violations

**Recommendation Uncertainty**:

When multiple methods have similar priority scores, CAIS provides:

1. **Primary recommendation** with highest score
2. **Alternative methods** for robustness checking  
3. **Sensitivity analysis** guidance
4. **Assumption testing** protocols

Walkthrough 3: Regression Discontinuity
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Scenario**: College admission cutoff analysis

**Dataset Characteristics**:
- Students with test scores around admission cutoff
- Treatment: College admission (based on score ≥ cutoff)
- Outcome: Future earnings
- Running variable: Test score (continuous)
- Cutoff: Score = 1200

**Decision Process**:

.. mermaid::

   flowchart TD
       A[Dataset Analysis Begins] --> B{Is this an RCT?}
       B -->|❌ No - Admission based on test scores| C{What data structure?}
       C -->|✅ Running variable with cutoff| D{Is cutoff sharp?}
       D -->|✅ Yes - Score ≥ 1200 determines admission| E[Selected: Regression Discontinuity]
       
       style A fill:#e8f5e8
       style E fill:#fff3e0
       
       F[Decision Reasoning:<br/>• Sharp cutoff creates quasi-randomization<br/>• Students just above/below cutoff are similar<br/>• Local causal identification at cutoff]
       E --> F

**Step-by-step Analysis**:

1. **Study Design Check**: ❌ NOT RCT
   - Admission determined by test score, not randomization
   - Selection into treatment is systematic
   - **Result**: Observational methods required

2. **Running Variable Check**: ✅ IDENTIFIED
   - Test score is continuous assignment variable
   - Clear relationship between score and treatment
   - **Result**: RDD potentially applicable

3. **Cutoff Assessment**: ✅ SHARP CUTOFF
   - Score ≥ 1200 deterministically assigns treatment
   - No exceptions or fuzzy assignment around cutoff
   - **Result**: Sharp RDD design

4. **Continuity Check**: ⚠️ REQUIRES TESTING
   - Assumption: potential outcomes continuous at cutoff
   - No other changes occurring at score = 1200
   - **Result**: Diagnostic tests required

5. **Manipulation Check**: ⚠️ REQUIRES TESTING
   - Students shouldn't precisely control scores around cutoff
   - Test for bunching or discontinuities in score density
   - **Result**: Validation tests required

6. **Final Selection**: **Regression Discontinuity Design**
   - **Priority Score**: 2 (High - Quasi-experimental)
   - **Justification**: "Running variable with cutoff—consider RDD"
   - **Assumptions**: Continuity at cutoff, no manipulation

**Alternative Methods Considered**:
- Linear regression (ignores discontinuity structure)
- Propensity score methods (inappropriate with deterministic assignment)

Walkthrough 4: Observational Study with Rich Covariates
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Scenario**: Medical treatment effectiveness

**Dataset Characteristics**:
- 5,000 patients from electronic health records
- Treatment: New medication vs standard care
- Outcome: Recovery time
- Rich covariates: Demographics, medical history, comorbidities, lab values

**Decision Process**:

.. mermaid::

   flowchart TD
       A[Dataset Analysis Begins] --> B{Is this an RCT?}
       B -->|❌ No - Observational medical data| C{What data structure?}
       C -->|Cross-sectional - Single treatment decision| D{Treatment type?}
       D -->|Binary - New vs standard medication| E{Instrumental variable?}
       E -->|❌ No - No valid instrument identified| F{Rich covariates?}
       F -->|✅ Yes - Extensive patient data| G{Covariate overlap?}
       G -->|✅ Good - Similar patients in both groups| H[Selected: Propensity Score Matching]
       
       style A fill:#e8f5e8
       style H fill:#f3e5f5
       
       I[Decision Reasoning:<br/>• Rich covariates enable matching<br/>• Good overlap supports valid comparisons<br/>• Creates balanced treatment groups]
       H --> I

**Step-by-step Analysis**:

1. **Study Design Check**: ❌ NOT RCT
   - Treatment assignment based on physician decisions
   - Patient characteristics influence treatment choice
   - **Result**: Confounding likely present

2. **Data Structure Assessment**: ✅ CROSS-SECTIONAL
   - Single time point for treatment decision
   - No temporal variation to exploit
   - **Result**: Cross-sectional methods required

3. **Treatment Type Check**: ✅ BINARY
   - Clear treatment vs control comparison
   - No dose-response relationship
   - **Result**: Binary treatment methods applicable

4. **Instrumental Variable Search**: ❌ NOT AVAILABLE
   - No policy changes or random encouragement
   - Physician preferences correlated with patient outcomes
   - **Result**: IV methods not applicable

5. **Covariate Assessment**: ✅ RICH COVARIATES
   - Extensive patient characteristics available
   - Variables likely capture major confounders
   - **Result**: Propensity score methods feasible

6. **Overlap Assessment**: ✅ GOOD OVERLAP
   - Similar patients receive both treatments
   - Common support region is substantial
   - **Result**: Matching preferred over weighting

7. **Final Selection**: **Propensity Score Matching**
   - **Priority Score**: 1 (Medium - Observational)
   - **Justification**: "Covariates observed; PS method chosen based on overlap"
   - **Assumptions**: Unconfoundedness, common support, correct PS model

**Alternative Methods Considered**:
- Propensity Score Weighting (if overlap was poor)
- Linear regression with controls (simpler but stronger assumptions)

Walkthrough 5: Instrumental Variables Analysis
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Scenario**: Education returns analysis

**Dataset Characteristics**:
- Survey data on wages and education
- Treatment: Years of education (continuous)
- Outcome: Log wages
- Instrument: Distance to nearest college (affects education but not wages directly)
- Covariates: Demographics, family background

**Decision Process**:

.. mermaid::

   flowchart TD
       A[Dataset Analysis Begins] --> B{Is this an RCT?}
       B -->|❌ No - Observational survey data| C{What data structure?}
       C -->|Cross-sectional survey| D{Treatment type?}
       D -->|Continuous - Years of education| E{Instrumental variable?}
       E -->|✅ Yes - Distance to college| F{Instrument validity?}
       F -->|✅ Passes relevance and exclusion tests| G[Selected: Instrumental Variables]
       
       style A fill:#e8f5e8
       style G fill:#fff3e0
       
       H[Decision Reasoning:<br/>• Instrument handles unobserved ability bias<br/>• Distance affects education but not wages directly<br/>• Continuous treatment IV estimation]
       G --> H

**Step-by-step Analysis**:

1. **Study Design Check**: ❌ NOT RCT
   - Education choices are endogenous
   - Unobserved ability affects both education and wages
   - **Result**: Confounding present

2. **Treatment Type Check**: ✅ CONTINUOUS
   - Years of education is continuous variable
   - Dose-response relationship expected
   - **Result**: Continuous treatment methods needed

3. **Instrumental Variable Assessment**: ✅ AVAILABLE
   - Distance to college varies geographically
   - Affects education access but not wages directly
   - **Result**: IV methods applicable

4. **Instrument Validation**:
   - **Relevance**: ✅ Distance strongly predicts education (F-stat > 10)
   - **Exclusion**: ✅ Distance doesn't affect wages except through education
   - **Exogeneity**: ✅ Distance uncorrelated with unobserved ability
   - **Result**: Valid instrument confirmed

5. **Final Selection**: **Instrumental Variables**
   - **Priority Score**: 3 (High - Quasi-experimental)
   - **Justification**: "Instrument available for continuous treatment"
   - **Assumptions**: Relevance, exclusion restriction, independence

**Alternative Methods Considered**:
- Linear regression (biased due to omitted ability)
- Generalized propensity score (requires unconfoundedness assumption)

Method Selection Examples
-------------------------

Example 1: A/B Test Analysis
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Data characteristics**:
- Users randomly assigned to treatment/control
- Outcome measured post-treatment
- Some user characteristics available

**Decision path**:
1. Randomized experiment? **Yes**
2. Covariates available? **Yes** 
3. **Selected method**: Linear regression with covariates

**Why**: Randomization ensures causal identification; covariates improve precision

Example 2: Policy Evaluation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Data characteristics**:
- Policy implemented in some states but not others
- Data before and after policy implementation
- State-level panel data

**Decision path**:
1. Randomized experiment? **No**
2. Panel data with treatment timing? **Yes**
3. **Selected method**: Difference-in-Differences

**Why**: Exploits timing variation; controls for time-invariant confounders

Example 3: Observational Study
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Data characteristics**:
- Cross-sectional survey data
- Binary treatment (college attendance)
- Rich set of background characteristics

**Decision path**:
1. Randomized experiment? **No**
2. Panel data? **No**
3. Running variable? **No**
4. Binary treatment? **Yes**
5. Instrumental variable? **No**
6. Rich covariates? **Yes**
7. Good overlap? **Yes**
8. **Selected method**: Propensity Score Matching

**Why**: Rich covariates allow credible matching; good overlap ensures valid comparisons

Decision Node Documentation
----------------------------

This section provides detailed documentation of each decision node in the algorithm, including the specific criteria, thresholds, and examples that guide method selection.

Node 1: Randomization Assessment
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Decision Question**: "Is this data from a randomized controlled trial?"

**Detection Criteria**:

1. **User Specification**: ``is_rct=True`` parameter
2. **Automatic Detection Patterns**:
   - Balanced covariates across treatment groups (p-value > 0.05 for all covariates)
   - Treatment assignment uncorrelated with pre-treatment variables
   - Equal group sizes (within 10% tolerance)

**Code Implementation**:

.. code-block:: python

   def detect_randomization(df, treatment, covariates):
       """Detect if data appears to be from an RCT"""
       balance_tests = []
       
       for covariate in covariates:
           # T-test for continuous, chi-square for categorical
           if df[covariate].dtype in ['float64', 'int64']:
               stat, p_value = ttest_ind(
                   df[df[treatment]==1][covariate],
                   df[df[treatment]==0][covariate]
               )
           else:
               contingency = pd.crosstab(df[treatment], df[covariate])
               stat, p_value, _, _ = chi2_contingency(contingency)
           
           balance_tests.append(p_value > 0.05)
       
       # Majority of covariates should be balanced
       return sum(balance_tests) / len(balance_tests) > 0.8

**Examples**:
- ✅ **RCT**: A/B test with random assignment algorithm
- ✅ **RCT**: Clinical trial with randomization protocol
- ❌ **Not RCT**: Survey data with self-selected treatment
- ❌ **Not RCT**: Policy evaluation with non-random implementation

**Next Steps**:
- **If RCT**: → Node 2A (Covariate Assessment for RCT)
- **If Not RCT**: → Node 2B (Data Structure Assessment)

Node 2A: RCT Covariate Assessment
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Decision Question**: "Are covariates available in the RCT?"

**Detection Criteria**:
- Pre-treatment variables identified in dataset
- Variables measured before treatment assignment
- Sufficient variation (not constant across observations)

**Threshold**: At least one valid covariate with non-zero variance

**Examples**:
- ✅ **Covariates Available**: User demographics, baseline measurements
- ❌ **No Covariates**: Only treatment and outcome variables

**Next Steps**:
- **If Covariates Available**: → Node 3A (Instrument Check for RCT)
- **If No Covariates**: → Node 3B (Pure RCT Analysis)

Node 2B: Data Structure Assessment
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Decision Question**: "What type of data structure is present?"

**Detection Criteria**:

1. **Panel Data Detection**:
   - Time variable present and varies
   - Unit identifier present
   - Treatment varies within units over time
   - Multiple observations per unit

2. **Regression Discontinuity Detection**:
   - Running variable identified (continuous)
   - Cutoff value specified or detectable
   - Treatment assignment based on running variable threshold

3. **Cross-sectional Detection**:
   - Single time period or no time variation
   - No clear discontinuity structure

**Code Implementation**:

.. code-block:: python

   def detect_data_structure(df, time_var, unit_var, treatment, running_var, cutoff):
       """Detect the data structure type"""
       
       # Panel data check
       if time_var and unit_var:
           time_periods = df[time_var].nunique()
           units = df[unit_var].nunique()
           treatment_variation = df.groupby(unit_var)[treatment].nunique().mean()
           
           if time_periods > 1 and treatment_variation > 1.1:
               return "panel"
       
       # RDD check
       if running_var and cutoff is not None:
           # Check if treatment assignment follows cutoff rule
           above_cutoff = df[running_var] >= cutoff
           treatment_above = df[above_cutoff][treatment].mean()
           treatment_below = df[~above_cutoff][treatment].mean()
           
           if abs(treatment_above - treatment_below) > 0.8:
               return "rdd"
       
       return "cross_sectional"

**Next Steps**:
- **If Panel**: → Difference-in-Differences (Priority 1)
- **If RDD**: → Regression Discontinuity (Priority 2)
- **If Cross-sectional**: → Node 3C (Treatment Type Assessment)

Node 3C: Treatment Variable Type Assessment
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Decision Question**: "What type of treatment variable is present?"

**Detection Criteria**:

1. **Binary Treatment**:
   - Only two unique values (typically 0/1)
   - Clear treatment vs control distinction

2. **Continuous Treatment**:
   - Many unique values (>10% of sample size)
   - Numeric variable with meaningful ordering
   - Dose-response relationship expected

3. **Categorical Treatment**:
   - Multiple discrete categories (3-10 typical)
   - No natural ordering between categories

**Code Implementation**:

.. code-block:: python

   def detect_treatment_type(df, treatment):
       """Detect treatment variable type"""
       unique_values = df[treatment].nunique()
       total_obs = len(df)
       
       if unique_values == 2:
           return "binary"
       elif unique_values > 0.1 * total_obs and df[treatment].dtype in ['float64', 'int64']:
           return "continuous"
       elif unique_values <= 10:
           return "categorical"
       else:
           return "continuous"  # Many categories treated as continuous

**Next Steps**:
- **If Binary**: → Node 4A (Instrumental Variable Check - Binary)
- **If Continuous**: → Node 4B (Instrumental Variable Check - Continuous)
- **If Categorical**: → Node 4C (Instrumental Variable Check - Categorical)

Node 4A: Instrumental Variable Assessment (Binary Treatment)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Decision Question**: "Is a valid instrumental variable available?"

**Validation Criteria**:

1. **Relevance Test**:
   - F-statistic > 10 in first-stage regression
   - Instrument significantly predicts treatment (p < 0.05)

2. **Exclusion Restriction** (not directly testable):
   - Instrument affects outcome only through treatment
   - Domain knowledge and theory support exclusion

3. **Independence Test**:
   - Instrument uncorrelated with observed confounders
   - Balance tests show instrument is "as good as random"

**Code Implementation**:

.. code-block:: python

   def validate_instrument(df, treatment, instrument, outcome, covariates):
       """Validate instrumental variable"""
       
       # Relevance test (first stage)
       X = df[covariates + [instrument]]
       y = df[treatment]
       first_stage = sm.OLS(y, sm.add_constant(X)).fit()
       f_stat = first_stage.fvalue
       
       # Weak instrument test
       if f_stat < 10:
           return False, "Weak instrument (F < 10)"
       
       # Balance tests (independence)
       balance_scores = []
       for covariate in covariates:
           corr = df[instrument].corr(df[covariate])
           balance_scores.append(abs(corr) < 0.1)
       
       if sum(balance_scores) / len(balance_scores) < 0.8:
           return False, "Instrument correlated with covariates"
       
       return True, "Valid instrument"

**Examples**:
- ✅ **Valid IV**: Distance to college (affects education, not wages directly)
- ✅ **Valid IV**: Random encouragement in experiment
- ❌ **Invalid IV**: Parental income (affects both education and wages)

**Next Steps**:
- **If Valid IV**: → Instrumental Variables (Priority 3)
- **If No Valid IV**: → Node 5A (Covariate Richness Assessment)

Node 5A: Covariate Richness Assessment
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Decision Question**: "Are rich covariates available for adjustment?"

**Assessment Criteria**:

1. **Quantity**: Number of covariates relative to sample size
   - Rich: > 5 covariates or > 1% of sample size
   - Limited: ≤ 5 covariates and < 1% of sample size

2. **Quality**: Predictive power for treatment and outcome
   - R² > 0.1 in treatment prediction model
   - R² > 0.1 in outcome prediction model

3. **Relevance**: Domain knowledge of confounding variables
   - Variables known to affect both treatment and outcome
   - Pre-treatment measurements

**Code Implementation**:

.. code-block:: python

   def assess_covariate_richness(df, treatment, outcome, covariates):
       """Assess richness of available covariates"""
       n_covariates = len(covariates)
       n_obs = len(df)
       
       # Quantity check
       quantity_rich = n_covariates > 5 or n_covariates > 0.01 * n_obs
       
       # Quality check - predictive power
       X = df[covariates]
       
       # Treatment prediction
       treatment_model = LogisticRegression().fit(X, df[treatment])
       treatment_r2 = treatment_model.score(X, df[treatment])
       
       # Outcome prediction
       outcome_model = LinearRegression().fit(X, df[outcome])
       outcome_r2 = outcome_model.score(X, df[outcome])
       
       quality_rich = treatment_r2 > 0.1 and outcome_r2 > 0.1
       
       return quantity_rich and quality_rich

**Next Steps**:
- **If Rich Covariates**: → Node 6A (Covariate Overlap Assessment)
- **If Limited Covariates**: → Linear Regression with Controls (Priority 4)

Node 6A: Covariate Overlap Assessment
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Decision Question**: "Is there good covariate overlap between treatment groups?"

**Assessment Criteria**:

1. **Propensity Score Overlap**:
   - Common support region covers >80% of observations
   - Propensity scores not too close to 0 or 1

2. **Standardized Mean Differences**:
   - SMD < 0.25 for all covariates (good balance)
   - SMD > 0.25 indicates poor overlap

3. **Overlap Score Calculation**:
   - Score = 1 - (proportion of observations outside common support)
   - Threshold: Score ≥ 0.1 indicates good overlap

**Code Implementation**:

.. code-block:: python

   def assess_covariate_overlap(df, treatment, covariates):
       """Assess covariate overlap between treatment groups"""
       
       # Estimate propensity scores
       X = df[covariates]
       ps_model = LogisticRegression().fit(X, df[treatment])
       propensity_scores = ps_model.predict_proba(X)[:, 1]
       
       # Common support assessment
       treated_ps = propensity_scores[df[treatment] == 1]
       control_ps = propensity_scores[df[treatment] == 0]
       
       min_treated = treated_ps.min()
       max_treated = treated_ps.max()
       min_control = control_ps.min()
       max_control = control_ps.max()
       
       # Overlap region
       overlap_min = max(min_treated, min_control)
       overlap_max = min(max_treated, max_control)
       
       # Proportion in common support
       in_support = ((propensity_scores >= overlap_min) & 
                     (propensity_scores <= overlap_max))
       overlap_score = in_support.mean()
       
       return overlap_score

**Examples**:
- ✅ **Good Overlap**: Similar patient populations receive both treatments
- ❌ **Poor Overlap**: Sickest patients only get experimental treatment

**Next Steps**:
- **If Good Overlap (≥0.1)**: → Propensity Score Matching (Priority 1)
- **If Poor Overlap (<0.1)**: → Propensity Score Weighting (Priority 2)

Priority Ordering and Method Selection
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Final Selection Algorithm**:

1. **Collect all applicable methods** with their priority scores
2. **Filter out excluded methods** (if any specified)
3. **Sort by priority score** (lower = higher priority)
4. **Select method with highest priority** (lowest score)
5. **Identify alternative methods** for robustness checking

**Priority Hierarchy**:

.. list-table:: Method Priority Scores
   :header-rows: 1
   :widths: 30 20 20 30

   * - Method Category
     - Priority Score
     - Identification Strength
     - Example Methods
   * - Experimental
     - 1-3
     - Strongest
     - IV (Encouragement), Linear Reg + Covariates, Diff in Means
   * - Quasi-Experimental
     - 1-4
     - Strong
     - DiD, RDD, IV, Front-door
   * - Observational
     - 1-5
     - Moderate
     - PS Matching, PS Weighting, Linear Reg, GPS, Multinomial

Understanding Method Recommendations
------------------------------------

Priority Ordering
~~~~~~~~~~~~~~~~~

When multiple methods are applicable, CAIS prioritizes based on:

1. **Strength of identification**: Methods with weaker assumptions ranked higher
2. **Data requirements**: Methods that fully utilize available data structure
3. **Robustness**: Methods less sensitive to specification choices

**Experimental Methods** (Highest Priority):
- Instrumental Variables (encouragement designs)
- Linear Regression with covariates
- Difference in means

**Quasi-Experimental Methods** (High Priority):
- Difference-in-Differences
- Regression Discontinuity
- Instrumental Variables

**Observational Methods** (Medium Priority):
- Propensity Score methods
- Backdoor adjustment
- Linear regression with controls

Alternative Methods
~~~~~~~~~~~~~~~~~~~

CAIS also suggests alternative methods for robustness checking:

**Primary recommendation**: Best method given data and assumptions
**Alternatives**: Other plausible methods for sensitivity analysis

**Example output**:
- **Selected**: Propensity Score Matching
- **Alternatives**: Propensity Score Weighting, Linear Regression
- **Justification**: Rich covariates with good overlap support matching

Customizing Method Selection
----------------------------

Excluding Methods
~~~~~~~~~~~~~~~~~

You can exclude specific methods from consideration:

.. code-block:: python

   from causal_agent import CausalAgent
   
   agent = CausalAgent()
   result = agent.analyze(
       data=df,
       treatment="treatment_var",
       outcome="outcome_var",
       excluded_methods=["linear_regression", "diff_in_means"]
   )

Forcing Method Selection
~~~~~~~~~~~~~~~~~~~~~~~~

You can also specify a particular method:

.. code-block:: python

   result = agent.analyze(
       data=df,
       treatment="treatment_var", 
       outcome="outcome_var",
       method="propensity_score_matching"
   )

Validating Method Choice
------------------------

After method selection, CAIS provides:

**Assumption Checking**:
- Tests of key identifying assumptions
- Diagnostic plots and statistics
- Sensitivity analysis recommendations

**Balance Assessment**:
- Covariate balance before/after adjustment
- Overlap diagnostics
- Common support analysis

**Robustness Checks**:
- Alternative method comparisons
- Specification sensitivity
- Placebo tests where applicable

Interactive Tools and Utilities
--------------------------------

Method Comparison Tool
~~~~~~~~~~~~~~~~~~~~~~

Compare different causal inference methods side by side to understand their trade-offs:

.. raw:: html

   <div id="method-comparison-tool">
       <!-- Method comparison tool will be populated by JavaScript -->
   </div>

   <style>
   .comparison-header {
       text-align: center;
       margin-bottom: 20px;
       padding: 15px;
       background: #f8f9fa;
       border-radius: 6px;
   }
   
   .method-selector {
       display: grid;
       grid-template-columns: 1fr 1fr;
       gap: 20px;
       margin-bottom: 20px;
       padding: 15px;
       background: white;
       border: 1px solid #dee2e6;
       border-radius: 6px;
   }
   
   .method-selector label {
       font-weight: 600;
       color: #495057;
       margin-bottom: 5px;
       display: block;
   }
   
   .method-selector select {
       width: 100%;
       padding: 8px 12px;
       border: 1px solid #ced4da;
       border-radius: 4px;
       font-size: 14px;
   }
   
   .comparison-grid {
       display: grid;
       grid-template-columns: 1fr 1fr;
       gap: 20px;
   }
   
   .method-comparison-card {
       background: white;
       border: 2px solid #e9ecef;
       border-radius: 8px;
       padding: 20px;
   }
   
   .method-comparison-card.experimental {
       border-left-color: #007bff;
   }
   
   .method-comparison-card.quasi-experimental {
       border-left-color: #fd7e14;
   }
   
   .method-comparison-card.observational {
       border-left-color: #6f42c1;
   }
   
   .method-category {
       color: #6c757d;
       font-size: 14px;
       margin-bottom: 5px;
   }
   
   .method-priority {
       color: #495057;
       font-weight: 600;
       margin-bottom: 15px;
   }
   
   .method-assumptions h5 {
       color: #495057;
       margin-bottom: 10px;
   }
   
   .method-assumptions ul {
       list-style: none;
       padding: 0;
       margin: 0;
   }
   
   .method-assumptions li {
       padding: 5px 0;
       color: #6c757d;
       font-size: 14px;
       position: relative;
       padding-left: 20px;
   }
   
   .method-assumptions li:before {
       content: "•";
       position: absolute;
       left: 0;
       color: #007bff;
       font-weight: bold;
   }
   
   @media (max-width: 768px) {
       .method-selector {
           grid-template-columns: 1fr;
       }
       
       .comparison-grid {
           grid-template-columns: 1fr;
       }
   }
   </style>

Method Diagnostic Tool
~~~~~~~~~~~~~~~~~~~~~~

Validate whether your data meets the assumptions for specific methods:

.. raw:: html

   <div id="diagnostic-tool">
       <!-- Diagnostic tool will be populated by JavaScript -->
   </div>

   <style>
   .diagnostic-header {
       text-align: center;
       margin-bottom: 20px;
       padding: 15px;
       background: #f8f9fa;
       border-radius: 6px;
   }
   
   .diagnostic-form {
       background: white;
       border: 1px solid #dee2e6;
       border-radius: 6px;
       padding: 20px;
   }
   
   .form-group {
       margin-bottom: 20px;
   }
   
   .form-group label {
       font-weight: 600;
       color: #495057;
       margin-bottom: 8px;
       display: block;
   }
   
   .form-group select {
       width: 100%;
       padding: 8px 12px;
       border: 1px solid #ced4da;
       border-radius: 4px;
       font-size: 14px;
   }
   
   .diagnostic-checklist {
       margin-top: 20px;
   }
   
   .diagnostic-checklist h4 {
       color: #495057;
       margin-bottom: 15px;
   }
   
   .assumption-check {
       display: flex;
       align-items: center;
       gap: 10px;
       padding: 10px;
       margin: 8px 0;
       background: #f8f9fa;
       border-radius: 4px;
   }
   
   .assumption-check input[type="checkbox"] {
       margin: 0;
   }
   
   .assumption-check label {
       flex: 1;
       margin: 0;
       color: #495057;
       font-size: 14px;
   }
   
   .assumption-help button {
       background: #17a2b8;
       color: white;
       border: none;
       padding: 4px 8px;
       border-radius: 3px;
       font-size: 12px;
       cursor: pointer;
   }
   
   .assumption-help button:hover {
       background: #138496;
   }
   
   .diagnostic-result {
       margin-top: 20px;
       text-align: center;
   }
   
   .diagnostic-result button {
       background: #007bff;
       color: white;
       border: none;
       padding: 10px 20px;
       border-radius: 4px;
       font-size: 14px;
       cursor: pointer;
   }
   
   .diagnostic-result button:hover {
       background: #0056b3;
   }
   
   .diagnostic-success {
       background: #d4edda;
       border: 1px solid #c3e6cb;
       color: #155724;
       padding: 15px;
       border-radius: 4px;
       margin-top: 15px;
   }
   
   .diagnostic-warning {
       background: #fff3cd;
       border: 1px solid #ffeaa7;
       color: #856404;
       padding: 15px;
       border-radius: 4px;
       margin-top: 15px;
   }
   
   .diagnostic-error {
       background: #f8d7da;
       border: 1px solid #f5c6cb;
       color: #721c24;
       padding: 15px;
       border-radius: 4px;
       margin-top: 15px;
   }
   </style>

Decision Tree Algorithm Implementation
--------------------------------------

For developers interested in the technical implementation, here's how the decision tree algorithm works in CAIS:

**Core Algorithm Structure**:

.. code-block:: python

   def select_method(dataset_properties, excluded_methods=None):
       """
       Main decision tree algorithm for method selection
       
       Args:
           dataset_properties: Dict containing data characteristics
           excluded_methods: List of methods to exclude
           
       Returns:
           Dict with selected method and justification
       """
       candidates = []  # (method, priority_score) pairs
       
       # 1. Check for randomization
       if dataset_properties.get('is_rct'):
           candidates.extend(get_experimental_methods(dataset_properties))
       
       # 2. Check for quasi-experimental designs
       candidates.extend(get_quasi_experimental_methods(dataset_properties))
       
       # 3. Add observational methods as fallbacks
       candidates.extend(get_observational_methods(dataset_properties))
       
       # 4. Filter excluded methods and select best
       return select_best_method(candidates, excluded_methods)

**Priority Scoring System**:

The algorithm uses a hierarchical priority system:

1. **Experimental Methods** (Priority 1-3): Strongest causal identification
2. **Quasi-Experimental Methods** (Priority 1-4): Natural experiments
3. **Observational Methods** (Priority 1-5): Statistical adjustment

**Method Selection Logic**:

.. mermaid::

   graph TD
       A[Dataset Properties] --> B{RCT?}
       B -->|Yes| C[Experimental Methods]
       B -->|No| D[Check Data Structure]
       
       D --> E{Panel Data?}
       D --> F{RDD Design?}
       D --> G{Cross-sectional?}
       
       E -->|Yes| H[Difference-in-Differences]
       F -->|Yes| I[Regression Discontinuity]
       G -->|Yes| J[Observational Methods]
       
       C --> K[Priority Ranking]
       H --> K
       I --> K
       J --> K
       
       K --> L[Filter Excluded]
       L --> M[Select Best Method]

**Assumption Validation Framework**:

Each method comes with a set of testable assumptions:

.. code-block:: python

   METHOD_ASSUMPTIONS = {
       'difference_in_differences': [
           'parallel_trends',      # Testable with pre-treatment data
           'no_anticipation',      # Check treatment timing
           'stable_composition'    # Verify unit consistency
       ],
       'regression_discontinuity': [
           'continuity_at_cutoff', # Density tests
           'no_manipulation',      # McCrary test
           'no_other_changes'      # Domain knowledge
       ],
       # ... other methods
   }

Next Steps
----------

1. **Understand your selected method**: Read the detailed method documentation
2. **Check assumptions**: Use the diagnostic tool above to validate assumptions
3. **Compare alternatives**: Use the comparison tool to understand trade-offs
4. **Interpret results**: Understand what your causal estimate means
5. **Run robustness checks**: Test with alternative methods

**Method-Specific Documentation**:

* :doc:`experimental/index` - RCT and experimental methods
* :doc:`quasi_experimental/index` - DiD, IV, and RDD methods
* :doc:`observational/index` - Propensity score and matching methods

**Interactive Resources**:

* Use the **Interactive Decision Tree** above to walk through method selection
* Try the **Method Comparison Tool** to understand differences between approaches
* Validate your choice with the **Diagnostic Tool**

**Tutorials and Examples**:

* :doc:`../tutorials/index` - Step-by-step examples with real data
* :doc:`../tutorials/examples/decision_path_comparisons` - Compare decision paths for different datasets
* :doc:`../getting_started/quickstart` - Quick start guide with decision tree examples
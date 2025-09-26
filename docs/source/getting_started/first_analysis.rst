Your First Causal Analysis
=========================

This guide provides a comprehensive walkthrough of conducting your first causal analysis with CAIS, helping you understand both the technical steps and the causal inference concepts behind them.

.. contents:: Contents
   :local:
   :depth: 3

Understanding Causal Questions
------------------------------

Before diving into the analysis, it's important to understand what makes a good causal question and how CAIS approaches causal inference.

What is Causal Inference?
~~~~~~~~~~~~~~~~~~~~~~~~~~

Causal inference goes beyond correlation to answer questions like:

* **Does X cause Y?** (e.g., "Does education cause higher income?")
* **What would happen if...?** (e.g., "What would happen to crime rates if we increased police presence?")
* **How much of an effect?** (e.g., "By how much does job training increase earnings?")

**Key Concept:** Correlation ≠ Causation

.. code-block:: python

   # Example: Ice cream sales and drowning deaths are correlated
   # But ice cream doesn't cause drowning - both increase in summer!
   
   # CAIS helps identify true causal relationships by:
   # 1. Selecting appropriate methods
   # 2. Controlling for confounding variables
   # 3. Testing assumptions

Good vs. Poor Causal Questions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Good Causal Questions:**

.. code-block:: text

   ✓ "Does participating in job training increase income?"
   ✓ "What is the effect of class size on student performance?"
   ✓ "How does a new drug affect patient recovery time?"

**Poor Causal Questions:**

.. code-block:: text

   ✗ "What factors are associated with income?" (descriptive, not causal)
   ✗ "Is there a relationship between X and Y?" (correlation question)
   ✗ "What predicts outcome Z?" (prediction, not causation)

Step-by-Step Analysis Walkthrough
----------------------------------

Let's work through a complete analysis using a real-world example: the effect of education on earnings.

Step 1: Problem Setup
~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   import pandas as pd
   from causal_agent import run_causal_analysis
   import os
   
   # Set up your environment
   os.environ['OPENAI_API_KEY'] = 'your-api-key-here'
   
   # Define our causal question
   causal_question = "What is the effect of college education on annual income?"
   
   print(f"Research Question: {causal_question}")

Step 2: Data Preparation
~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Load sample education-income dataset
   # (In practice, you'd load your own data)
   data = pd.read_csv('data/all_data/education_income.csv')
   
   # Examine the data structure
   print("Dataset Overview:")
   print(f"Shape: {data.shape}")
   print(f"Columns: {list(data.columns)}")
   print("\nFirst few rows:")
   print(data.head())
   
   # Check for missing values
   print("\nMissing values:")
   print(data.isnull().sum())

**Expected Output:**

.. code-block:: text

   Dataset Overview:
   Shape: (2500, 12)
   Columns: ['person_id', 'college_degree', 'annual_income', 'age', 'gender', 
            'work_experience', 'industry', 'region', 'family_background', 
            'cognitive_ability', 'motivation', 'health_status']

Step 3: Running the Analysis
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Provide a detailed dataset description
   dataset_description = """
   This dataset contains information about working adults and their educational and 
   economic outcomes. Each row represents one individual with the following key variables:
   
   - college_degree: Whether the person has a college degree (treatment variable)
   - annual_income: Person's annual income in dollars (outcome variable)
   - age, gender, work_experience: Demographic controls
   - industry, region: Economic context variables
   - family_background: Socioeconomic background (potential confounder)
   - cognitive_ability, motivation: Individual characteristics (potential confounders)
   - health_status: Health-related factors
   
   The data comes from a longitudinal survey of working adults aged 25-65.
   """
   
   # Run the causal analysis
   print("Running causal analysis...")
   result = run_causal_analysis(
       query=causal_question,
       dataset_path='education_income.csv',
       dataset_description=dataset_description
   )
   
   print("Analysis completed!")

Step 4: Understanding the Results
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Extract key results
   method_used = result['results']['method_used']
   treatment_var = result['results']['treatment_variable']
   outcome_var = result['results']['outcome_variable']
   effect_estimate = result['results']['effect_estimate']
   std_error = result['results']['standard_error']
   p_value = result['results']['p_value']
   confidence_interval = result['results'].get('confidence_interval', 'Not available')
   
   print("=== ANALYSIS RESULTS ===")
   print(f"Method Selected: {method_used}")
   print(f"Treatment Variable: {treatment_var}")
   print(f"Outcome Variable: {outcome_var}")
   print(f"Causal Effect Estimate: ${effect_estimate:,.2f}")
   print(f"Standard Error: ${std_error:,.2f}")
   print(f"P-value: {p_value:.4f}")
   print(f"95% Confidence Interval: {confidence_interval}")
   
   # Statistical significance
   is_significant = p_value < 0.05
   print(f"Statistically Significant: {'Yes' if is_significant else 'No'}")

**Sample Output:**

.. code-block:: text

   === ANALYSIS RESULTS ===
   Method Selected: Propensity Score Matching
   Treatment Variable: college_degree
   Outcome Variable: annual_income
   Causal Effect Estimate: $18,450.00
   Standard Error: $2,340.00
   P-value: 0.0001
   95% Confidence Interval: [$13,864, $23,036]
   Statistically Significant: Yes

Step 5: Interpreting the Results
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Get the AI interpretation
   interpretation = result['results']['interpretation']
   print("\n=== AI INTERPRETATION ===")
   print(interpretation)
   
   # Manual interpretation guide
   print("\n=== INTERPRETATION GUIDE ===")
   
   if is_significant:
       print(f"✓ The analysis suggests that having a college degree CAUSES an increase")
       print(f"  in annual income of approximately ${effect_estimate:,.0f}.")
       print(f"✓ This effect is statistically significant (p = {p_value:.4f} < 0.05).")
       print(f"✓ We can be 95% confident the true effect is between {confidence_interval}.")
   else:
       print(f"✗ No statistically significant causal effect was found.")
       print(f"✗ The estimated effect of ${effect_estimate:,.0f} could be due to chance.")
   
   print(f"\n📊 Method Used: {method_used}")
   print("   This method was selected based on the characteristics of your data.")

Step 6: Examining Method Selection
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Understanding why CAIS selected a particular method helps you trust and validate the results:

.. code-block:: python

   # Check method selection reasoning (if available)
   method_reasoning = result['results'].get('method_reasoning', 'Not provided')
   print(f"\n=== WHY THIS METHOD? ===")
   print(f"Selected Method: {method_used}")
   print(f"Reasoning: {method_reasoning}")
   
   # Common methods and when they're used:
   method_guide = {
       'Randomized Controlled Trial': 'Data appears to be from a randomized experiment',
       'Propensity Score Matching': 'Observational data with good covariate balance possible',
       'Difference-in-Differences': 'Panel data with treatment timing variation',
       'Instrumental Variables': 'Strong instrumental variable detected',
       'Regression Discontinuity': 'Sharp cutoff/threshold detected in treatment assignment',
       'Linear Regression': 'Simple observational data with controls'
   }
   
   if method_used in method_guide:
       print(f"Typical use case: {method_guide[method_used]}")

Step 7: Validating Results
~~~~~~~~~~~~~~~~~~~~~~~~~~

Always validate your causal analysis results:

.. code-block:: python

   # Check for potential issues
   print("\n=== VALIDATION CHECKLIST ===")
   
   # 1. Effect size reasonableness
   print(f"1. Effect Size: ${effect_estimate:,.0f}")
   if abs(effect_estimate) > 100000:
       print("   ⚠️  Large effect - double-check data and units")
   else:
       print("   ✓ Reasonable effect size")
   
   # 2. Statistical significance
   print(f"2. P-value: {p_value:.4f}")
   if p_value < 0.05:
       print("   ✓ Statistically significant")
   else:
       print("   ⚠️  Not statistically significant")
   
   # 3. Sample size
   sample_size = len(data)
   print(f"3. Sample Size: {sample_size:,}")
   if sample_size < 100:
       print("   ⚠️  Small sample size - results may be unreliable")
   else:
       print("   ✓ Adequate sample size")
   
   # 4. Missing data
   missing_pct = (data.isnull().sum().sum() / (len(data) * len(data.columns))) * 100
   print(f"4. Missing Data: {missing_pct:.1f}%")
   if missing_pct > 10:
       print("   ⚠️  High missing data percentage")
   else:
       print("   ✓ Low missing data")

Common Patterns and What They Mean
-----------------------------------

Different Types of Results
~~~~~~~~~~~~~~~~~~~~~~~~~~

**Strong Positive Effect:**

.. code-block:: python

   # Example result interpretation
   if effect_estimate > 0 and p_value < 0.01:
       print("Strong evidence that treatment INCREASES the outcome")
       print("Effect is both large and highly statistically significant")

**No Effect:**

.. code-block:: python

   if abs(effect_estimate) < std_error and p_value > 0.05:
       print("No evidence of causal effect")
       print("Treatment appears to have no impact on outcome")

**Negative Effect:**

.. code-block:: python

   if effect_estimate < 0 and p_value < 0.05:
       print("Evidence that treatment DECREASES the outcome")
       print("This could be beneficial or harmful depending on context")

Understanding Confidence Intervals
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Interpreting confidence intervals
   def interpret_confidence_interval(ci_lower, ci_upper, effect):
       print(f"95% Confidence Interval: [{ci_lower:,.0f}, {ci_upper:,.0f}]")
       
       if ci_lower > 0:
           print("✓ We're confident the effect is positive")
       elif ci_upper < 0:
           print("✓ We're confident the effect is negative")
       else:
           print("⚠️  Confidence interval includes zero - effect uncertain")
       
       width = ci_upper - ci_lower
       precision = width / abs(effect) if effect != 0 else float('inf')
       
       if precision < 0.5:
           print("✓ Precise estimate (narrow confidence interval)")
       else:
           print("⚠️  Imprecise estimate (wide confidence interval)")

Troubleshooting Common Issues
-----------------------------

Data Quality Issues
~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Check for common data problems
   def diagnose_data_issues(data):
       issues = []
       
       # Check for constant variables
       for col in data.columns:
           if data[col].nunique() <= 1:
               issues.append(f"Variable '{col}' has no variation")
       
       # Check for extreme outliers
       numeric_cols = data.select_dtypes(include=['number']).columns
       for col in numeric_cols:
           q99 = data[col].quantile(0.99)
           q01 = data[col].quantile(0.01)
           if (q99 / q01) > 1000:  # Very large range
               issues.append(f"Variable '{col}' has extreme outliers")
       
       # Check sample size
       if len(data) < 50:
           issues.append("Sample size very small (< 50 observations)")
       
       return issues
   
   # Run diagnostics
   issues = diagnose_data_issues(data)
   if issues:
       print("⚠️  Data Quality Issues Found:")
       for issue in issues:
           print(f"   - {issue}")
   else:
       print("✓ No major data quality issues detected")

Method Selection Issues
~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # If you disagree with method selection
   print("\n=== IF YOU DISAGREE WITH METHOD SELECTION ===")
   print("CAIS selected:", method_used)
   print("\nConsider these factors:")
   print("1. Is your data from a randomized experiment? → RCT methods")
   print("2. Do you have before/after data? → Difference-in-Differences")
   print("3. Is there a sharp cutoff in treatment? → Regression Discontinuity")
   print("4. Do you have a valid instrument? → Instrumental Variables")
   print("5. Is this observational data? → Matching or Regression methods")

Result Interpretation Issues
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Common interpretation mistakes
   print("\n=== COMMON INTERPRETATION MISTAKES ===")
   print("❌ 'X predicts Y' → ✅ 'X causes Y'")
   print("❌ 'X is associated with Y' → ✅ 'X has a causal effect on Y'")
   print("❌ 'The model shows...' → ✅ 'The causal analysis suggests...'")
   print("❌ Ignoring confidence intervals → ✅ Report uncertainty")
   print("❌ Over-interpreting small effects → ✅ Consider practical significance")

Next Steps and Advanced Topics
-------------------------------

Now that you've completed your first analysis, here are suggested next steps:

**Immediate Next Steps:**

1. **Try different datasets:** Experiment with various domains and data types
2. **Compare methods:** Run the same analysis with different method specifications
3. **Explore batch processing:** Analyze multiple datasets systematically

**Advanced Learning:**

1. **Method deep-dives:** :doc:`../methods/index` - Learn about each causal inference method
2. **Domain tutorials:** :doc:`../tutorials/index` - See examples in your field
3. **Configuration options:** :doc:`../user_guide/advanced_usage` - Customize CAIS behavior

**Best Practices:**

.. code-block:: python

   # Template for systematic causal analysis
   def systematic_causal_analysis(query, data_path, description):
       """Template for thorough causal analysis"""
       
       # 1. Pre-analysis
       print("=== PRE-ANALYSIS ===")
       data = pd.read_csv(data_path)
       print(f"Data shape: {data.shape}")
       print(f"Missing data: {data.isnull().sum().sum()}")
       
       # 2. Run analysis
       print("\n=== ANALYSIS ===")
       result = run_causal_analysis(query, data_path, description)
       
       # 3. Validate results
       print("\n=== VALIDATION ===")
       effect = result['results']['effect_estimate']
       p_val = result['results']['p_value']
       print(f"Effect: {effect}, P-value: {p_val}")
       
       # 4. Interpret
       print("\n=== INTERPRETATION ===")
       print(result['results']['interpretation'])
       
       return result

**Research Workflow Integration:**

.. code-block:: python

   # Integrate CAIS into your research workflow
   research_questions = [
       "Does treatment A improve outcome X?",
       "What is the effect of policy B on metric Y?",
       "How does intervention C affect behavior Z?"
   ]
   
   results = {}
   for question in research_questions:
       print(f"\nAnalyzing: {question}")
       # Run your analysis here
       # results[question] = run_causal_analysis(...)
   
   # Compare results across questions
   # Generate research report
   # Plan follow-up studies

Congratulations! You've completed your first comprehensive causal analysis with CAIS. You're now ready to tackle more complex analyses and explore advanced features.

Continue to :doc:`../user_guide/index` for advanced usage patterns, or explore :doc:`../tutorials/index` for domain-specific examples!
Batch Processing
================

CAIS provides powerful batch processing capabilities for analyzing multiple datasets efficiently. This guide covers command-line batch processing, programmatic batch analysis, and best practices for large-scale causal inference workflows.

Command Line Batch Processing
------------------------------

Basic Batch Analysis
~~~~~~~~~~~~~~~~~~~~

The simplest way to process multiple datasets is using the CLI batch command with a metadata CSV file:

.. code-block:: bash

    causal_agent batch metadata.csv data_folder/ results.json

Metadata CSV Format
~~~~~~~~~~~~~~~~~~~

Create a CSV file with the following columns:

.. code-block:: text

    natural_language_query,data_description,data_files,method,answer
    "What is the effect of job training on earnings?","LaLonde job training data","lalonde_data.csv","propensity_score","positive"
    "What is the effect of education on income?","Survey data with demographics","survey_data.csv","regression","positive"
    "What was the impact of the policy change?","Panel data pre/post policy","policy_data.csv","difference_in_differences","significant"

Required columns:
- ``natural_language_query``: The causal question to analyze
- ``data_description``: Description of the dataset
- ``data_files``: Filename of the dataset (relative to data folder)

Optional columns:
- ``method``: Expected method (for validation)
- ``answer``: Expected result (for comparison)

Example Batch Command
~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

    # Basic batch processing
    causal_agent batch studies_metadata.csv /path/to/data/ batch_results.json
    
    # With specific LLM configuration
    causal_agent batch studies_metadata.csv /path/to/data/ batch_results.json \
        --llm-provider anthropic \
        --llm-name claude-3-5-sonnet-latest
    
    # Process subset of studies
    head -10 studies_metadata.csv > subset_metadata.csv
    causal_agent batch subset_metadata.csv /path/to/data/ subset_results.json

Batch Results Format
~~~~~~~~~~~~~~~~~~~~

The output JSON file contains structured results for each analysis:

.. code-block:: json

    {
      "0": {
        "query": "What is the effect of job training on earnings?",
        "method": "propensity_score",
        "answer": "positive",
        "dataset_description": "LaLonde job training data",
        "dataset_path": "/path/to/data/lalonde_data.csv",
        "final_result": {
          "method": "propensity_score_matching",
          "causal_effect": 1794.34,
          "standard_deviation": 632.85,
          "treatment_variable": "treat",
          "outcome_variable": "re78",
          "covariates": ["age", "education", "black", "hispanic", "married"]
        }
      },
      "1": {
        "query": "What is the effect of education on income?",
        "error": "Dataset file not found"
      }
    }

Programmatic Batch Processing
-----------------------------

Python Batch Analysis
~~~~~~~~~~~~~~~~~~~~~

For more control over batch processing, use the Python API:

.. code-block:: python

    import pandas as pd
    import json
    from pathlib import Path
    from causal_agent import run_causal_analysis
    from typing import Dict, List, Any
    
    def batch_causal_analysis(metadata_path: str, 
                             data_folder: str,
                             output_path: str = None) -> Dict[int, Dict[str, Any]]:
        """Run batch causal analysis from metadata CSV."""
        
        # Load metadata
        metadata_df = pd.read_csv(metadata_path)
        results = {}
        
        for idx, row in metadata_df.iterrows():
            print(f"Processing analysis {idx + 1}/{len(metadata_df)}: {row['natural_language_query'][:50]}...")
            
            dataset_path = Path(data_folder) / row['data_files']
            
            try:
                result = run_causal_analysis(
                    query=row['natural_language_query'],
                    dataset_path=str(dataset_path),
                    dataset_description=row.get('data_description')
                )
                
                # Extract key results
                results[idx] = {
                    'query': row['natural_language_query'],
                    'expected_method': row.get('method'),
                    'expected_answer': row.get('answer'),
                    'dataset_description': row.get('data_description'),
                    'dataset_path': str(dataset_path),
                    'success': True,
                    'final_result': {
                        'method': result['results']['results']['method_used'],
                        'causal_effect': result['results']['results']['effect_estimate'],
                        'standard_error': result['results']['results']['standard_error'],
                        'p_value': result['results']['results']['p_value'],
                        'confidence_interval': result['results']['results']['confidence_interval'],
                        'treatment_variable': result['results']['variables']['treatment_variable'],
                        'outcome_variable': result['results']['variables']['outcome_variable'],
                        'covariates': result['results']['variables'].get('covariates', [])
                    }
                }
                
            except Exception as e:
                results[idx] = {
                    'query': row['natural_language_query'],
                    'dataset_path': str(dataset_path),
                    'success': False,
                    'error': str(e)
                }
        
        # Save results if output path provided
        if output_path:
            with open(output_path, 'w') as f:
                json.dump(results, f, indent=2)
            print(f"Results saved to {output_path}")
        
        return results
    
    # Usage
    results = batch_causal_analysis(
        metadata_path="studies_metadata.csv",
        data_folder="data/",
        output_path="batch_results.json"
    )

Advanced Batch Processing
~~~~~~~~~~~~~~~~~~~~~~~~~

Process batches with custom configurations and error handling:

.. code-block:: python

    import os
    import time
    from concurrent.futures import ThreadPoolExecutor, as_completed
    from typing import Optional, Callable
    
    class BatchProcessor:
        """Advanced batch processor for causal analysis."""
        
        def __init__(self, 
                     llm_provider: str = "openai",
                     llm_model: str = None,
                     max_workers: int = 4,
                     retry_attempts: int = 3):
            self.llm_provider = llm_provider
            self.llm_model = llm_model
            self.max_workers = max_workers
            self.retry_attempts = retry_attempts
            
            # Set environment variables
            os.environ["LLM_PROVIDER"] = llm_provider
            if llm_model:
                os.environ["LLM_MODEL"] = llm_model
        
        def process_single_analysis(self, 
                                  analysis_config: Dict[str, Any],
                                  progress_callback: Optional[Callable] = None) -> Dict[str, Any]:
            """Process a single analysis with retry logic."""
            
            for attempt in range(self.retry_attempts):
                try:
                    start_time = time.time()
                    
                    result = run_causal_analysis(
                        query=analysis_config['query'],
                        dataset_path=analysis_config['dataset_path'],
                        dataset_description=analysis_config.get('description')
                    )
                    
                    processing_time = time.time() - start_time
                    
                    if progress_callback:
                        progress_callback(analysis_config['id'], 'completed', processing_time)
                    
                    return {
                        'id': analysis_config['id'],
                        'success': True,
                        'processing_time': processing_time,
                        'attempt': attempt + 1,
                        'result': result
                    }
                    
                except Exception as e:
                    if attempt == self.retry_attempts - 1:
                        if progress_callback:
                            progress_callback(analysis_config['id'], 'failed', 0)
                        
                        return {
                            'id': analysis_config['id'],
                            'success': False,
                            'error': str(e),
                            'attempts': self.retry_attempts
                        }
                    
                    time.sleep(2 ** attempt)  # Exponential backoff
        
        def process_batch(self, 
                         analysis_configs: List[Dict[str, Any]],
                         progress_callback: Optional[Callable] = None) -> List[Dict[str, Any]]:
            """Process multiple analyses in parallel."""
            
            results = []
            
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                # Submit all analyses
                future_to_config = {
                    executor.submit(self.process_single_analysis, config, progress_callback): config
                    for config in analysis_configs
                }
                
                # Collect results as they complete
                for future in as_completed(future_to_config):
                    result = future.result()
                    results.append(result)
            
            # Sort results by original order
            results.sort(key=lambda x: x['id'])
            return results
    
    # Usage with progress tracking
    def progress_callback(analysis_id: str, status: str, processing_time: float):
        if status == 'completed':
            print(f"✓ Analysis {analysis_id} completed in {processing_time:.2f}s")
        elif status == 'failed':
            print(f"✗ Analysis {analysis_id} failed")
    
    # Create analysis configurations
    configs = [
        {
            'id': 'study_001',
            'query': 'What is the effect of treatment on outcome?',
            'dataset_path': 'data/study1.csv',
            'description': 'RCT data from study 1'
        },
        {
            'id': 'study_002', 
            'query': 'What is the effect of intervention on metric?',
            'dataset_path': 'data/study2.csv',
            'description': 'Observational data from study 2'
        }
    ]
    
    # Process batch
    processor = BatchProcessor(
        llm_provider="anthropic",
        max_workers=2,
        retry_attempts=3
    )
    
    results = processor.process_batch(configs, progress_callback)

Result Analysis and Reporting
-----------------------------

Analyzing Batch Results
~~~~~~~~~~~~~~~~~~~~~~

Generate summary reports from batch processing results:

.. code-block:: python

    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    from typing import Dict, Any
    
    def analyze_batch_results(results: Dict[int, Dict[str, Any]]) -> pd.DataFrame:
        """Analyze and summarize batch processing results."""
        
        summary_data = []
        
        for idx, result in results.items():
            if result.get('success', False):
                final_result = result['final_result']
                summary_data.append({
                    'analysis_id': idx,
                    'query': result['query'][:50] + '...',
                    'method_used': final_result['method'],
                    'effect_estimate': final_result['causal_effect'],
                    'standard_error': final_result['standard_error'],
                    'p_value': final_result['p_value'],
                    'significant': final_result['p_value'] < 0.05,
                    'treatment_var': final_result['treatment_variable'],
                    'outcome_var': final_result['outcome_variable'],
                    'n_covariates': len(final_result.get('covariates', [])),
                    'processing_time': result.get('processing_time', 0)
                })
            else:
                summary_data.append({
                    'analysis_id': idx,
                    'query': result['query'][:50] + '...',
                    'error': result.get('error', 'Unknown error'),
                    'success': False
                })
        
        return pd.DataFrame(summary_data)
    
    def create_batch_report(results_df: pd.DataFrame, output_path: str = None):
        """Create comprehensive batch analysis report."""
        
        # Success rate
        success_rate = results_df['success'].mean() if 'success' in results_df.columns else 1.0
        
        # Method distribution
        if 'method_used' in results_df.columns:
            method_counts = results_df['method_used'].value_counts()
        
        # Effect size distribution
        if 'effect_estimate' in results_df.columns:
            effects_df = results_df.dropna(subset=['effect_estimate'])
        
        # Create visualizations
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Success rate pie chart
        success_counts = [success_rate, 1 - success_rate]
        axes[0, 0].pie(success_counts, labels=['Success', 'Failed'], autopct='%1.1f%%')
        axes[0, 0].set_title('Analysis Success Rate')
        
        # Method distribution
        if 'method_used' in results_df.columns:
            method_counts.plot(kind='bar', ax=axes[0, 1])
            axes[0, 1].set_title('Methods Used')
            axes[0, 1].tick_params(axis='x', rotation=45)
        
        # Effect size distribution
        if 'effect_estimate' in results_df.columns and len(effects_df) > 0:
            axes[1, 0].hist(effects_df['effect_estimate'], bins=20, alpha=0.7)
            axes[1, 0].set_title('Distribution of Effect Estimates')
            axes[1, 0].set_xlabel('Effect Estimate')
        
        # Significance vs effect size
        if all(col in results_df.columns for col in ['effect_estimate', 'significant']):
            significant_df = effects_df.dropna(subset=['significant'])
            colors = ['red' if not sig else 'green' for sig in significant_df['significant']]
            axes[1, 1].scatter(significant_df['effect_estimate'], 
                             significant_df['standard_error'], c=colors, alpha=0.6)
            axes[1, 1].set_xlabel('Effect Estimate')
            axes[1, 1].set_ylabel('Standard Error')
            axes[1, 1].set_title('Effect Size vs Standard Error (Green=Significant)')
        
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        # Print summary statistics
        print(f"Batch Analysis Summary")
        print(f"=" * 50)
        print(f"Total analyses: {len(results_df)}")
        print(f"Success rate: {success_rate:.1%}")
        
        if 'method_used' in results_df.columns:
            print(f"\nMethods used:")
            for method, count in method_counts.items():
                print(f"  {method}: {count}")
        
        if 'effect_estimate' in results_df.columns:
            effects_df = results_df.dropna(subset=['effect_estimate'])
            if len(effects_df) > 0:
                print(f"\nEffect estimates:")
                print(f"  Mean: {effects_df['effect_estimate'].mean():.3f}")
                print(f"  Median: {effects_df['effect_estimate'].median():.3f}")
                print(f"  Std: {effects_df['effect_estimate'].std():.3f}")
                
                significant_count = effects_df['significant'].sum() if 'significant' in effects_df.columns else 0
                print(f"  Significant results: {significant_count}/{len(effects_df)} ({significant_count/len(effects_df):.1%})")
    
    # Usage
    with open('batch_results.json', 'r') as f:
        batch_results = json.load(f)
    
    # Convert to DataFrame and analyze
    results_df = analyze_batch_results(batch_results)
    create_batch_report(results_df, 'batch_analysis_report.png')

Performance Optimization
------------------------

Efficient Resource Usage
~~~~~~~~~~~~~~~~~~~~~~~~

Optimize batch processing for large datasets:

.. code-block:: python

    import psutil
    import gc
    from typing import Iterator
    
    def memory_efficient_batch_processing(metadata_path: str,
                                        data_folder: str,
                                        batch_size: int = 10,
                                        memory_threshold: float = 0.8) -> Iterator[Dict[str, Any]]:
        """Process batches with memory management."""
        
        metadata_df = pd.read_csv(metadata_path)
        
        for i in range(0, len(metadata_df), batch_size):
            # Check memory usage
            memory_percent = psutil.virtual_memory().percent / 100
            if memory_percent > memory_threshold:
                print(f"Memory usage high ({memory_percent:.1%}), running garbage collection...")
                gc.collect()
            
            batch_df = metadata_df.iloc[i:i + batch_size]
            batch_results = {}
            
            for idx, row in batch_df.iterrows():
                try:
                    result = run_causal_analysis(
                        query=row['natural_language_query'],
                        dataset_path=f"{data_folder}/{row['data_files']}",
                        dataset_description=row.get('data_description')
                    )
                    batch_results[idx] = {'success': True, 'result': result}
                    
                except Exception as e:
                    batch_results[idx] = {'success': False, 'error': str(e)}
            
            yield batch_results
            
            # Clear memory after each batch
            gc.collect()
    
    # Usage for large datasets
    all_results = {}
    for batch_results in memory_efficient_batch_processing(
        "large_metadata.csv", 
        "data/",
        batch_size=5
    ):
        all_results.update(batch_results)
        print(f"Processed batch, total results: {len(all_results)}")

Monitoring and Logging
~~~~~~~~~~~~~~~~~~~~~~

Implement comprehensive monitoring for batch jobs:

.. code-block:: python

    import logging
    import time
    from datetime import datetime
    from pathlib import Path
    
    def setup_batch_logging(log_dir: str = "batch_logs"):
        """Set up logging for batch processing."""
        Path(log_dir).mkdir(exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = Path(log_dir) / f"batch_processing_{timestamp}.log"
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        
        return logging.getLogger(__name__)
    
    def monitored_batch_processing(metadata_path: str, data_folder: str):
        """Batch processing with comprehensive monitoring."""
        logger = setup_batch_logging()
        
        start_time = time.time()
        metadata_df = pd.read_csv(metadata_path)
        total_analyses = len(metadata_df)
        
        logger.info(f"Starting batch processing of {total_analyses} analyses")
        logger.info(f"Metadata file: {metadata_path}")
        logger.info(f"Data folder: {data_folder}")
        
        results = {}
        successful = 0
        failed = 0
        
        for idx, row in metadata_df.iterrows():
            analysis_start = time.time()
            
            try:
                logger.info(f"Processing analysis {idx + 1}/{total_analyses}: {row['natural_language_query'][:50]}...")
                
                result = run_causal_analysis(
                    query=row['natural_language_query'],
                    dataset_path=f"{data_folder}/{row['data_files']}",
                    dataset_description=row.get('data_description')
                )
                
                analysis_time = time.time() - analysis_start
                successful += 1
                
                results[idx] = {
                    'success': True,
                    'result': result,
                    'processing_time': analysis_time
                }
                
                logger.info(f"✓ Analysis {idx + 1} completed in {analysis_time:.2f}s")
                
            except Exception as e:
                analysis_time = time.time() - analysis_start
                failed += 1
                
                results[idx] = {
                    'success': False,
                    'error': str(e),
                    'processing_time': analysis_time
                }
                
                logger.error(f"✗ Analysis {idx + 1} failed after {analysis_time:.2f}s: {e}")
        
        total_time = time.time() - start_time
        success_rate = successful / total_analyses
        
        logger.info(f"Batch processing completed in {total_time:.2f}s")
        logger.info(f"Success rate: {successful}/{total_analyses} ({success_rate:.1%})")
        logger.info(f"Average time per analysis: {total_time/total_analyses:.2f}s")
        
        return results

Best Practices
--------------

Data Organization
~~~~~~~~~~~~~~~~~

1. **Consistent File Structure**: Organize datasets in a clear folder hierarchy
2. **Descriptive Filenames**: Use meaningful names that indicate study/dataset content
3. **Metadata Completeness**: Provide comprehensive dataset descriptions
4. **Version Control**: Track metadata files and analysis configurations

Error Handling
~~~~~~~~~~~~~~

1. **Graceful Degradation**: Continue processing other analyses when one fails
2. **Detailed Error Logging**: Capture specific error messages and context
3. **Retry Logic**: Implement exponential backoff for transient failures
4. **Validation**: Check data format and completeness before processing

Performance
~~~~~~~~~~~

1. **Parallel Processing**: Use appropriate number of workers based on system resources
2. **Memory Management**: Monitor and manage memory usage for large batches
3. **Caching**: Cache intermediate results to avoid recomputation
4. **Progress Tracking**: Provide clear progress indicators for long-running jobs

Quality Assurance
~~~~~~~~~~~~~~~~~

1. **Result Validation**: Check for reasonable effect sizes and statistical significance
2. **Method Consistency**: Verify that appropriate methods are being selected
3. **Comparative Analysis**: Compare results across similar studies
4. **Manual Review**: Sample and manually review a subset of results

Next Steps
----------

- For LLM provider configuration and optimization, see :doc:`configuration`
- For understanding method selection and validation, see :doc:`../methods/index`
- For advanced customization options, see :doc:`advanced_usage`
- For integration into production workflows, see :doc:`../deployment/index`
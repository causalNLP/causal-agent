import pytest
import os
import pandas as pd

# Import the refactored parse_input function
from auto_causal.components import input_parser

# Check if OpenAI API key is available, skip if not
api_key_present = bool(os.environ.get("OPENAI_API_KEY"))
skip_if_no_key = pytest.mark.skipif(not api_key_present, reason="OPENAI_API_KEY environment variable not set")

@skip_if_no_key
def test_parse_input_with_real_llm():
    """Tests the parse_input function invoking the actual LLM.
    
    Note: This test requires the OPENAI_API_KEY environment variable to be set 
    and will make a real API call.
    """
    # --- Test Case 1: Effect query with dataset and constraint ---
    query1 = "analyze the effect of 'Minimum Wage Increase' on 'Unemployment Rate' using data/county_data.csv where year > 2010"
    
    # Provide some dummy dataset context
    dataset_info1 = {
        'columns': ['County', 'Year', 'Minimum Wage Increase', 'Unemployment Rate', 'Population'],
        'column_types': {'County': 'object', 'Year': 'int64', 'Minimum Wage Increase': 'int64', 'Unemployment Rate': 'float64', 'Population': 'int64'},
        'sample_rows': [
            {'County': 'A', 'Year': 2009, 'Minimum Wage Increase': 0, 'Unemployment Rate': 5.5, 'Population': 10000},
            {'County': 'A', 'Year': 2011, 'Minimum Wage Increase': 1, 'Unemployment Rate': 6.0, 'Population': 10200}
        ]
    }
    
    # Create a dummy data file for path checking (relative to workspace root)
    dummy_file_path = "data/county_data.csv"
    os.makedirs(os.path.dirname(dummy_file_path), exist_ok=True)
    with open(dummy_file_path, 'w') as f:
        f.write("County,Year,Minimum Wage Increase,Unemployment Rate,Population\n")
        f.write("A,2009,0,5.5,10000\n")
        f.write("A,2011,1,6.0,10200\n")
        
    result1 = input_parser.parse_input(query=query1, dataset_info=dataset_info1)
    
    # Clean up dummy file
    if os.path.exists(dummy_file_path):
        os.remove(dummy_file_path)
        # Try removing the directory if empty
        try:
            os.rmdir(os.path.dirname(dummy_file_path))
        except OSError:
            pass # Ignore if directory is not empty or other error

    # Assertions for Test Case 1
    assert result1 is not None
    assert result1['original_query'] == query1
    assert result1['query_type'] == "EFFECT_ESTIMATION"
    assert result1['dataset_path'] == dummy_file_path # Check if path extraction worked
    
    # Check variables (allowing for some LLM interpretation flexibility)
    assert 'treatment' in result1['extracted_variables']
    assert 'outcome' in result1['extracted_variables']
    # Check if the core variable names are present in the extracted lists
    assert any('Minimum Wage Increase' in t for t in result1['extracted_variables'].get('treatment', []))
    assert any('Unemployment Rate' in o for o in result1['extracted_variables'].get('outcome', []))
    
    # Check constraints
    assert isinstance(result1['constraints'], list)
    # Check if a constraint related to 'year > 2010' was captured (LLM might phrase it differently)
    assert any('year' in c.lower() and '2010' in c for c in result1.get('constraints', [])), "Constraint 'year > 2010' not found or not parsed correctly."

    # --- Test Case 2: Counterfactual without dataset path ---
    query2 = "What would sales have been if we hadn't run the 'Summer Sale' campaign?"
    dataset_info2 = {
        'columns': ['Date', 'Sales', 'Summer Sale', 'Competitor Activity'],
        'column_types': { 'Date': 'datetime64[ns]', 'Sales': 'float64', 'Summer Sale': 'int64', 'Competitor Activity': 'float64'}
    }
    
    result2 = input_parser.parse_input(query=query2, dataset_info=dataset_info2)
    
    # Assertions for Test Case 2
    assert result2 is not None
    assert result2['query_type'] == "COUNTERFACTUAL"
    assert result2['dataset_path'] is None # No path mentioned or inferrable here
    assert any('Summer Sale' in t for t in result2['extracted_variables'].get('treatment', []))
    assert any('Sales' in o for o in result2['extracted_variables'].get('outcome', []))
    assert not result2['constraints'] # No constraints expected

    # --- Test Case 3: Simple query, LLM might fail validation? ---
    # This tests if the retry/failure mechanism logs warnings but doesn't crash
    # (Assuming LLM might struggle to extract treatment/outcome from just "sales vs ads")
    query3 = "sales vs ads"
    dataset_info3 = {
        'columns': ['sales', 'ads'],
        'column_types': {'sales': 'float', 'ads': 'float'}
    }
    result3 = input_parser.parse_input(query=query3, dataset_info=dataset_info3)
    assert result3 is not None
    # LLM might fail extraction; check default/fallback values
    # Query type might default to OTHER or CORRELATION/DESCRIPTIVE
    # Variables might be empty or partially filled
    # This mainly checks that the function completes without error even if LLM fails
    print(f"Result for ambiguous query: {result3}") 